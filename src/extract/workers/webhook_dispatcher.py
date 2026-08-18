"""Webhook delivery dispatcher + reconciler.

Runs as a coroutine group inside the batch worker. Same claim discipline as
items: ``FOR UPDATE SKIP LOCKED`` over due pending rows OR delivering rows
whose lease expired; all completion writes are lease-guarded.

Log lines carry ids, mode, attempt numbers, status classes, and finite
error codes — never destination URLs, bodies, metadata, or secrets. "What
exactly did we send, where" lives in the webhook_deliveries table
(``python -m extract.webhooks deliveries``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime

from extract.clients import webhook_transport
from extract.clients.webhook_secrets import WebhookSecretCrypto
from extract.clients.webhook_secrets import from_settings as crypto_from_settings
from extract.config import settings
from extract.core.webhooks import signature_headers
from extract.repos.webhooks import ClaimedDelivery, WebhookRepo
from extract.repos.webhooks import from_settings as webhook_repo_from_settings

logger = logging.getLogger("webhook")

# Decrypted-secret cache, keyed by ciphertext (rotation naturally invalidates).
_MAX_SECRET_CACHE = 256


class WebhookDispatcher:
    def __init__(
        self,
        *,
        repo: WebhookRepo | None = None,
        crypto: WebhookSecretCrypto | None = None,
    ) -> None:
        self.repo = repo or webhook_repo_from_settings()
        self.crypto = crypto or crypto_from_settings()
        self._client = webhook_transport._new_client()
        self._secret_cache: dict[str, str] = {}
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # --- lifecycle -------------------------------------------------------

    def start(self) -> list[asyncio.Task]:
        n = max(int(settings.EXTRACT_WEBHOOK_DISPATCH_CONCURRENCY), 1)
        self._tasks = [asyncio.create_task(self._dispatch_loop(i)) for i in range(n)]
        self._tasks.append(asyncio.create_task(self._reconcile_loop()))
        return self._tasks

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._client.aclose()
        await self.repo.aclose()

    # --- delivery --------------------------------------------------------

    async def _dispatch_loop(self, slot: int) -> None:
        poll = max(settings.EXTRACT_WEBHOOK_POLL_INTERVAL_MS, 100) / 1000.0
        backoff = poll
        while not self._stop.is_set():
            try:
                claimed = await self.repo.claim_next_delivery(
                    lease_seconds=settings.EXTRACT_WEBHOOK_LEASE_SECONDS
                )
            except Exception:  # noqa: BLE001 — never let the slot die
                logger.exception("webhook.loop.error slot=%d", slot)
                backoff = min(backoff * 2, 30.0)
                await asyncio.sleep(backoff)
                continue
            if claimed is None:
                await asyncio.sleep(poll)
                backoff = poll
                continue
            backoff = poll
            try:
                await self._deliver_one(claimed)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "webhook.loop.error delivery_id=%s event_id=%s",
                    claimed.delivery_id,
                    claimed.event_id,
                )
                # Best-effort: schedule a retry instead of stranding the
                # lease; if this write fails too, lease expiry reclaims it.
                with contextlib.suppress(Exception):
                    await self.repo.mark_attempt_failed(
                        delivery_id=claimed.delivery_id,
                        lease_token=claimed.lease_token,
                        status_code=None,
                        error=f"dispatcher error: {type(e).__name__}",
                    )

    async def _deliver_one(self, claimed: ClaimedDelivery) -> None:
        mode = "svix" if claimed.endpoint_id is not None else "direct"

        # Kill switch (§3.6): endpoint disabled or soft-deleted since enqueue.
        if claimed.endpoint_id is not None and not claimed.endpoint_enabled:
            await self.repo.mark_cancelled(
                delivery_id=claimed.delivery_id, lease_token=claimed.lease_token
            )
            logger.info(
                "webhook.delivery.cancelled delivery_id=%s endpoint_id=%s",
                claimed.delivery_id,
                claimed.endpoint_id,
            )
            return

        headers = {"content-type": "application/json", "user-agent": "extract-webhooks/1.0"}
        if claimed.signed:
            secrets = await self._signing_secrets(claimed)
            headers.update(
                signature_headers(secrets, msg_id=claimed.event_id, body=claimed.body)
            )

        # Prefer the endpoint's current URL over the enqueue-time snapshot so
        # an admin URL edit applies to queued retries.
        url = claimed.endpoint_url or claimed.url

        started = time.perf_counter()
        result = await webhook_transport.deliver(
            url=url, body=claimed.body, headers=headers, client=self._client
        )
        latency_ms = (time.perf_counter() - started) * 1000
        attempt_number = claimed.attempts + 1
        logger.info(
            "webhook.attempt delivery_id=%s mode=%s attempt=%d ok=%s "
            "status_code=%s error_code=%s latency_ms=%.0f",
            claimed.delivery_id,
            mode,
            attempt_number,
            result.ok,
            result.status_code,
            result.error_code,
            latency_ms,
        )
        if result.ok:
            await self.repo.mark_succeeded(
                delivery_id=claimed.delivery_id,
                lease_token=claimed.lease_token,
                status_code=result.status_code or 200,
            )
            return
        new_status = await self.repo.mark_attempt_failed(
            delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token,
            status_code=result.status_code,
            error=result.error_detail,
        )
        if new_status == "failed":
            logger.warning(
                "webhook.delivery.failed delivery_id=%s mode=%s attempts=%d status_code=%s",
                claimed.delivery_id,
                mode,
                attempt_number,
                result.status_code,
            )

    async def _signing_secrets(self, claimed: ClaimedDelivery) -> list[str]:
        secrets: list[str] = []
        if claimed.secret_ciphertext:
            secrets.append(
                await self._decrypt_cached(claimed.secret_ciphertext, claimed.secret_key_id or "")
            )
        old_valid = (
            claimed.old_secret_ciphertext
            and claimed.old_secret_expires_at is not None
            and claimed.old_secret_expires_at
            > datetime.now(tz=UTC).replace(tzinfo=None)
        )
        if old_valid:
            secrets.append(
                await self._decrypt_cached(
                    claimed.old_secret_ciphertext or "", claimed.secret_key_id or ""
                )
            )
        if not secrets:
            raise RuntimeError("signed delivery has no signing secret")
        return secrets

    async def _decrypt_cached(self, ciphertext: str, key_id: str) -> str:
        cached = self._secret_cache.get(ciphertext)
        if cached is not None:
            return cached
        plaintext = await self.crypto.decrypt(ciphertext, key_id=key_id)
        if len(self._secret_cache) >= _MAX_SECRET_CACHE:
            self._secret_cache.pop(next(iter(self._secret_cache)))
        self._secret_cache[ciphertext] = plaintext
        return plaintext

    # --- reconciler + backlog gauge ---------------------------------------

    async def _reconcile_loop(self) -> None:
        interval = max(int(settings.EXTRACT_WEBHOOK_RECONCILE_INTERVAL_SECONDS), 30)
        while not self._stop.is_set():
            try:
                reconciled = await self.repo.reconcile_missing_events()
                age = await self.repo.oldest_pending_age_seconds()
                if reconciled:
                    # Any reconcile means the live enqueue missed — the
                    # consumer was made whole, but investigate.
                    logger.warning("webhook.reconciled count=%d", reconciled)
                logger.info("webhook.backlog oldest_pending_age_s=%.0f", age or 0.0)
            except Exception:  # noqa: BLE001
                logger.exception("webhook.reconcile.error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                continue


__all__ = ["WebhookDispatcher"]
