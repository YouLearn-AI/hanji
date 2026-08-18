"""Async-batch worker process.

Loops over Postgres for ready items, runs them through the same
``Extractor`` the sync API uses, writes results to the results bucket, and
updates DB rows. Run it as a second long-lived process alongside the API
server::

    python -m extract.workers.batch_worker

Concurrency model:

- One process holds N concurrent extraction tasks (``EXTRACT_WORKER_CONCURRENCY``).
- Each task ``claim_next_item`` -> process -> repeat. The claim query
  folds the ``run_after`` backoff into the WHERE, so this loop is
  straightforward.
- Crash recovery: leases auto-expire after
  ``EXTRACT_WORKER_LEASE_SECONDS``; the next worker reclaims via the
  same query. ``attempts`` is incremented every claim and capped at
  ``MAX_ATTEMPTS`` to prevent infinite loops on poison items.

The completion-webhook dispatcher + reconciler run as coroutines inside
this same process (``EXTRACT_WEBHOOKS_ENABLED``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import time

import httpx

from extract.clients.result_store import ResultStore
from extract.clients.result_store import from_settings as result_store_from_settings
from extract.config import settings
from extract.core import (
    DocumentTooLarge,
    ExtractionFailed,
    Extractor,
    ExtractRequest,
    PageLimitExceeded,
    RemoteFetchError,
    UnsupportedInput,
)
from extract.core.batch import ItemErrorCode, is_retryable, redact_url
from extract.core.io import load_bytes
from extract.core.pdf import MAX_SIZE_BYTES
from extract.observability.timing import StageTimer
from extract.repos.batches import BatchRepo, ClaimedItem
from extract.repos.batches import from_settings as batch_repo_from_settings
from extract.storage.inline import InlineStorage
from extract.workers.webhook_dispatcher import WebhookDispatcher

logger = logging.getLogger("worker")

MAX_ATTEMPTS = 3
HEARTBEAT_INTERVAL_SECONDS = 60


def _backoff_seconds(attempts: int) -> int:
    """Exponential backoff with jitter, capped at 5 minutes."""
    base = min(2 ** max(attempts - 1, 0), 300)
    return base + random.randint(0, base)


class WorkerState:
    """Process-global state. One per `main()` call."""

    def __init__(self) -> None:
        self.repo: BatchRepo | None = None
        self.result_store: ResultStore | None = None
        self.extractor: Extractor | None = None
        self.http: httpx.AsyncClient | None = None
        self.stop_event: asyncio.Event | None = None
        self.in_flight: set[asyncio.Task] = set()
        # Completion-webhook dispatcher + reconciler. Never uses `state.http`
        # — it has its own SSRF-safe transport.
        self.webhook_dispatcher: WebhookDispatcher | None = None


async def _bootstrap(state: WorkerState) -> None:
    state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=30.0),
        follow_redirects=True,
    )
    state.repo = batch_repo_from_settings()
    if not state.repo.configured:
        raise RuntimeError("DATABASE_URL must be set for the async batch worker")
    state.result_store = result_store_from_settings()
    if not state.result_store._results_bucket:  # noqa: SLF001
        raise RuntimeError("EXTRACT_RESULTS_BUCKET must be set for the async batch worker")
    # InlineStorage — image artifacts are inlined into the result blob
    # (base64), so the caller doesn't need a second round-trip per image.
    state.extractor = Extractor(storage=InlineStorage(), download_client=state.http)
    state.stop_event = asyncio.Event()


async def _shutdown(state: WorkerState) -> None:
    if state.stop_event is not None:
        state.stop_event.set()
    if state.in_flight:
        logger.info("worker.shutdown.draining in_flight=%d", len(state.in_flight))
        await asyncio.gather(*state.in_flight, return_exceptions=True)
    if state.webhook_dispatcher is not None:
        await state.webhook_dispatcher.stop()
    if state.http is not None:
        await state.http.aclose()
    if state.repo is not None:
        await state.repo.aclose()


# --- Heartbeat -------------------------------------------------------------


async def _heartbeat_loop(
    *,
    repo: BatchRepo,
    item_id: str,
    lease_token: str,
    stop_event: asyncio.Event,
) -> None:
    """Refresh the lease until the parent task signals stop.

    Bounded by ``EXTRACT_WORKER_LEASE_SECONDS``: if the heartbeat misses
    twice in a row (e.g. DB outage), the lease expires and another worker
    reclaims the item. That's the desired failure mode.
    """
    interval = min(HEARTBEAT_INTERVAL_SECONDS, settings.EXTRACT_WORKER_LEASE_SECONDS // 3)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return  # stop was set
        except TimeoutError:
            pass
        try:
            await repo.heartbeat_lease(
                item_id=item_id,
                lease_token=lease_token,
                lease_seconds=settings.EXTRACT_WORKER_LEASE_SECONDS,
            )
        except Exception:  # noqa: BLE001
            logger.warning("worker.heartbeat.failed item_id=%s", item_id, exc_info=True)


# --- Per-item processing ---------------------------------------------------


async def _process_item(state: WorkerState, claimed: ClaimedItem) -> None:
    """Run one item end-to-end. Never raises; exceptions are translated
    into terminal item statuses or scheduled retries.
    """
    repo = state.repo
    assert repo is not None
    assert state.result_store is not None
    assert state.stop_event is not None

    item_start = time.perf_counter()
    logger.info(
        "worker.item.start item_id=%s batch_id=%s attempts=%d",
        claimed.item_id,
        claimed.batch_id,
        claimed.attempts,
    )

    hb_stop = asyncio.Event()
    hb_task = asyncio.create_task(
        _heartbeat_loop(
            repo=repo,
            item_id=claimed.item_id,
            lease_token=claimed.lease_token,
            stop_event=hb_stop,
        )
    )

    try:
        # 1. Download. url-sourced items are fetched directly here — the bytes
        # live only in this process's memory for the duration of the item; they
        # are never written to the uploads bucket. file-sourced items fetch
        # from the uploads bucket.
        if claimed.url is not None:
            try:
                data = await load_bytes(
                    url=claimed.url,
                    max_size=MAX_SIZE_BYTES,
                    client=state.http,
                    block_private=True,
                )
            except Exception as e:  # noqa: BLE001
                await _handle_url_fetch_failure(state, claimed, e)
                return
        else:
            try:
                data = await state.result_store.fetch_upload(
                    bucket=claimed.file_s3_bucket,
                    key=claimed.file_s3_key,
                )
            except Exception as e:  # noqa: BLE001
                await _handle_fetch_failure(state, claimed, e)
                return

        # 2. Extract.
        timer = StageTimer()
        timer.meta["via"] = "batch"
        timer.meta["table_output_format"] = claimed.table_output_format
        timer.meta["chunking"] = claimed.chunking
        try:
            request = ExtractRequest(
                extract_text=claimed.extract_text,
                extract_images=claimed.extract_images,
                ocr=claimed.ocr,
                table_output_format=claimed.table_output_format,
                chunking=claimed.chunking,
                chunk_size=claimed.chunk_size,
            )
            assert state.extractor is not None
            response = await state.extractor.aextract_from_bytes(
                data,
                # For url items there's no uploaded filename — fall back to the
                # url itself so an extension hint (e.g. `.pdf`) still helps
                # detect_kind, mirroring what the sync url route does.
                filename=claimed.file_filename or claimed.url,
                request=request,
                timer=timer,
            )
            page_count = response.page_count
        except (DocumentTooLarge, PageLimitExceeded) as e:
            await _terminal_failure(
                state,
                claimed,
                error_code=ItemErrorCode.DOCUMENT_TOO_LARGE
                if isinstance(e, DocumentTooLarge)
                else ItemErrorCode.PAGE_LIMIT_EXCEEDED,
                error_message=str(e),
            )
            return
        except UnsupportedInput as e:
            await _terminal_failure(
                state,
                claimed,
                error_code=ItemErrorCode.UNSUPPORTED_INPUT,
                error_message=str(e),
            )
            return
        except ExtractionFailed as e:
            await _terminal_failure(
                state,
                claimed,
                error_code=ItemErrorCode.EXTRACTION_FAILED,
                error_message=str(e),
            )
            return
        except Exception as e:  # noqa: BLE001 - includes model-endpoint errors
            # Model-tier outages are transient, so the item retries as a
            # provider error rather than failing terminally.
            await _maybe_retry(
                state,
                claimed,
                error_code=ItemErrorCode.OCR_PROVIDER_ERROR,
                error_message=str(e),
            )
            return

        # 3. Write result.
        try:
            payload = response.model_dump(mode="json")
            payload["batch_id"] = claimed.batch_id
            payload["item_id"] = claimed.item_id
            payload["file_id"] = claimed.file_id
            payload["url"] = redact_url(claimed.url)
            payload["page_count"] = page_count
            body_bytes = json.dumps(payload).encode("utf-8")
            stored = await state.result_store.write_result(
                batch_id=claimed.batch_id,
                item_id=claimed.item_id,
                body=body_bytes,
            )
        except Exception as e:  # noqa: BLE001
            await _maybe_retry(
                state,
                claimed,
                error_code=ItemErrorCode.INTERNAL_ERROR,
                error_message=f"result store error: {e}",
            )
            return

        # 4. Mark succeeded (same transaction as the webhook enqueue).
        await repo.update_item_succeeded(
            item_id=claimed.item_id,
            lease_token=claimed.lease_token,
            page_count=page_count,
            result_s3_bucket=stored.bucket,
            result_s3_key=stored.key,
        )
        logger.info(
            "worker.item.complete item_id=%s batch_id=%s pages=%d "
            "latency_ms=%.0f result_bytes=%d",
            claimed.item_id,
            claimed.batch_id,
            page_count,
            (time.perf_counter() - item_start) * 1000.0,
            stored.bytes_written,
        )
    finally:
        hb_stop.set()
        with contextlib.suppress(Exception):
            await hb_task


async def _handle_fetch_failure(state: WorkerState, claimed: ClaimedItem, e: Exception) -> None:
    """The caller never finished the upload, or a lifecycle rule ate it.

    We treat 404 as terminal (caller must re-upload) and other errors
    as retryable.
    """
    err = getattr(getattr(e, "response", None), "get", lambda *_: {})("Error", {}).get("Code")
    if err in {"NoSuchKey", "404", "NotFound"}:
        await _terminal_failure(
            state,
            claimed,
            error_code=ItemErrorCode.UPLOAD_MISSING,
            error_message="Upload not found in storage",
        )
        return
    await _maybe_retry(
        state,
        claimed,
        error_code=ItemErrorCode.INTERNAL_ERROR,
        error_message=f"upload fetch error: {e}",
    )


def _redact_url_in_message(message: str, url: str | None) -> str:
    """Strip a presigned signature back out of a fetch-error message.

    `RemoteFetchError`'s message (and the underlying httpx error it wraps)
    embeds the exact url verbatim — often twice. This message is persisted to
    `extract_batch_items.error_message` and returned as-is in `error.message`
    on every poll, so it needs the same redaction `redact_url()` already
    applies to the item's `url` field — otherwise a failed presigned fetch
    (most commonly: the signature expired before the worker got to it) would
    leak the live signature to anyone who can read the batch.
    """
    if not url:
        return message
    redacted = redact_url(url)
    if redacted is None or redacted == url:
        return message
    return message.replace(url, redacted)


async def _handle_url_fetch_failure(state: WorkerState, claimed: ClaimedItem, e: Exception) -> None:
    """A url-sourced item's fetch failed.

    `RemoteFetchError.retryable` (set at the raise site in core/io.py) is
    authoritative: False for scheme/host/SSRF validation failures and 4xx
    responses (bad url, forbidden, not found, expired presigned signature) —
    these fail identically on every attempt, so we fail the item immediately
    instead of burning MAX_ATTEMPTS retries on something that can't recover.
    True for 5xx/408/409/429 and network-level errors, which retry like any
    other transient failure. Any other exception type (a bug, not a fetch
    failure) defaults to retryable, same as the file-fetch path.
    """
    retryable = e.retryable if isinstance(e, RemoteFetchError) else True
    message = _redact_url_in_message(str(e), claimed.url)
    if not retryable:
        await _terminal_failure(
            state,
            claimed,
            error_code=ItemErrorCode.URL_FETCH_FAILED,
            error_message=message,
        )
        return
    await _maybe_retry(
        state,
        claimed,
        error_code=ItemErrorCode.INTERNAL_ERROR,
        error_message=f"url fetch error: {message}",
    )


async def _maybe_retry(
    state: WorkerState,
    claimed: ClaimedItem,
    *,
    error_code: str,
    error_message: str,
) -> None:
    """Retry transient failures up to MAX_ATTEMPTS; otherwise terminal."""
    repo = state.repo
    assert repo is not None
    if claimed.attempts >= MAX_ATTEMPTS or not is_retryable(error_code):
        await _terminal_failure(
            state,
            claimed,
            error_code=error_code,
            error_message=error_message,
        )
        return
    backoff = _backoff_seconds(claimed.attempts)
    logger.warning(
        "worker.item.retry item_id=%s attempts=%d error_code=%s backoff_s=%d",
        claimed.item_id,
        claimed.attempts,
        error_code,
        backoff,
    )
    await repo.reschedule_item(
        item_id=claimed.item_id,
        lease_token=claimed.lease_token,
        backoff_seconds=backoff,
        error_code=error_code,
        error_message=error_message,
    )


async def _terminal_failure(
    state: WorkerState,
    claimed: ClaimedItem,
    *,
    error_code: str,
    error_message: str,
    page_count: int | None = None,
) -> None:
    repo = state.repo
    assert repo is not None
    logger.warning(
        "worker.item.failed item_id=%s batch_id=%s attempts=%d error_code=%s",
        claimed.item_id,
        claimed.batch_id,
        claimed.attempts,
        error_code,
    )
    await repo.update_item_failed(
        item_id=claimed.item_id,
        lease_token=claimed.lease_token,
        error_code=error_code,
        error_message=error_message,
        page_count=page_count,
    )


# --- Main loop -------------------------------------------------------------


async def _queue_depth_loop(state: WorkerState) -> None:
    """Log the claimable backlog periodically (autoscaling signal).

    Failures are logged and swallowed: a metrics gap must never take down a
    worker.
    """
    assert state.repo is not None
    assert state.stop_event is not None
    interval = max(settings.EXTRACT_WORKER_QUEUE_METRIC_INTERVAL_SECONDS, 10)
    while not state.stop_event.is_set():
        try:
            depth = await state.repo.count_claimable_items()
            logger.info("worker.queue.depth depth=%d", depth)
        except Exception as e:  # noqa: BLE001 — never let the gauge kill the worker
            logger.warning("worker.queue.depth.failed type=%s", type(e).__name__)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval)
            return  # stop was set
        except TimeoutError:
            continue


async def _claim_loop(state: WorkerState, slot: int) -> None:
    """One claim slot. Pulls one item, processes, repeats."""
    assert state.repo is not None
    assert state.stop_event is not None
    poll_interval = max(settings.EXTRACT_WORKER_POLL_INTERVAL_MS, 50) / 1000.0
    backoff = poll_interval
    while not state.stop_event.is_set():
        try:
            claimed = await state.repo.claim_next_item(
                lease_seconds=settings.EXTRACT_WORKER_LEASE_SECONDS,
            )
        except Exception:  # noqa: BLE001
            logger.exception("worker.claim.failed slot=%d", slot)
            await asyncio.sleep(min(backoff * 2, 30.0))
            backoff = min(backoff * 2, 30.0)
            continue
        if claimed is None:
            await asyncio.sleep(poll_interval)
            backoff = poll_interval
            continue
        backoff = poll_interval
        try:
            await _process_item(state, claimed)
        except Exception:  # noqa: BLE001 - never let the slot die
            logger.exception(
                "worker.process.crash slot=%d item_id=%s", slot, claimed.item_id
            )


def _install_signal_handlers(state: WorkerState) -> None:
    loop = asyncio.get_running_loop()

    def _handle(signum: int) -> None:
        logger.info("worker.signal signal=%d", signum)
        if state.stop_event is not None:
            state.stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            # Windows / non-Unix: silently skip; SIGTERM still terminates.
            loop.add_signal_handler(sig, _handle, int(sig))


async def run() -> None:
    """Worker entry point."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    state = WorkerState()
    await _bootstrap(state)
    try:
        _install_signal_handlers(state)
        n = max(
            int(os.environ.get("EXTRACT_WORKER_CONCURRENCY", settings.EXTRACT_WORKER_CONCURRENCY)),
            1,
        )
        logger.info(
            "worker.start concurrency=%d lease_seconds=%d poll_interval_ms=%d",
            n,
            settings.EXTRACT_WORKER_LEASE_SECONDS,
            settings.EXTRACT_WORKER_POLL_INTERVAL_MS,
        )
        slots = [asyncio.create_task(_claim_loop(state, i)) for i in range(n)]
        slots.append(asyncio.create_task(_queue_depth_loop(state)))
        state.in_flight.update(slots)
        if settings.EXTRACT_WEBHOOKS_ENABLED and settings.DATABASE_URL:
            state.webhook_dispatcher = WebhookDispatcher()
            state.webhook_dispatcher.start()
        assert state.stop_event is not None
        await state.stop_event.wait()
        for t in slots:
            t.cancel()
        await asyncio.gather(*slots, return_exceptions=True)
    finally:
        await _shutdown(state)


def main() -> None:  # pragma: no cover - process entrypoint
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
