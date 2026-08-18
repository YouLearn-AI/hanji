"""Webhook outbox persistence.

Two halves:

- ``enqueue_batch_event(conn, …)`` runs INSIDE the batch-finalization
  transaction (``BatchRepo._update_item_terminal`` / ``cancel_batch``) so the
  logical event commits atomically with the terminal state change. The
  ``webhook_events UNIQUE(batch_id, event_type)`` key makes it exactly-once
  even under concurrent finishers and the reconciler.
- ``WebhookRepo`` is the dispatcher's side: claim deliveries with
  ``FOR UPDATE SKIP LOCKED`` (due pending rows OR delivering rows whose lease
  expired — the OR is what un-strands a crash mid-delivery), lease-guarded
  completion writes, the retry ladder, and the bounded mode-aware reconciler.

Delivery is at-least-once by nature; consumers dedup on the stable
``svix-id`` (= ``webhook_events.id``).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from extract.config import settings
from extract.core.batch import BatchStatus
from extract.core.webhooks import (
    DELIVERY_STATUS_CANCELLED,
    DELIVERY_STATUS_DELIVERING,
    DELIVERY_STATUS_FAILED,
    DELIVERY_STATUS_PENDING,
    DELIVERY_STATUS_SUCCEEDED,
    EVENT_BATCH_UPDATE,
    WebhookMode,
    build_batch_update_body,
    new_delivery_id,
    new_message_id,
    next_attempt_delay_seconds,
    normalize_mode,
)


def _utcnow() -> datetime:
    """Naive UTC — matches the schema's `timestamp` (no TZ) convention."""
    return datetime.now(tz=UTC).replace(tzinfo=None)


# --- Enqueue (runs inside the caller's transaction) ---------------------------


async def enqueue_batch_event(conn, *, batch_row: Any) -> str | None:
    """Enqueue the ``batch.update`` event for a batch that JUST flipped
    ``completed_at`` null→set (the caller verifies the flip while holding the
    batch row lock). Returns the event id, or None when the batch isn't
    webhook-eligible.

    ``batch_row`` is the asyncpg Record returned by the finalizing UPDATE
    (``returning *``). ``.get()`` keeps a rolling deploy racing migration
    0020 harmless: missing columns mean "not opted in".
    """
    mode = normalize_mode(batch_row.get("webhook_mode"))
    if mode not in (WebhookMode.SVIX, WebhookMode.DIRECT):
        return None  # NULL/omitted/disabled → never (opt-in default)

    customer_id = batch_row["customer_id"]
    direct_url = batch_row.get("webhook_url")
    if mode == WebhookMode.DIRECT and not direct_url:
        return None

    endpoints: list[Any] = []
    if mode == WebhookMode.SVIX:
        endpoints = await conn.fetch(
            """
            select id, url from webhook_endpoints
             where customer_id = $1 and enabled and deleted_at is null
            """,
            customer_id,
        )
        if not endpoints:
            return None  # gate: svix requires ≥1 enabled endpoint

    counts_raw = batch_row["counts_json"]
    counts = json.loads(counts_raw) if isinstance(counts_raw, str) else dict(counts_raw or {})
    metadata_raw = batch_row["metadata_json"]
    metadata = (
        json.loads(metadata_raw)
        if isinstance(metadata_raw, str)
        else (dict(metadata_raw) if metadata_raw is not None else None)
    )
    body = build_batch_update_body(
        batch_id=batch_row["id"],
        status=batch_row["status"],
        counts={k: int(v) for k, v in counts.items()},
        total_items=int(batch_row["total_items"]),
        metadata=metadata,
        completed_at=batch_row["completed_at"],
        # The batch's result-retention deadline — the same `expires_at` the
        # resend path honors. Tells consumers how long they have
        # to fetch results before they're gone.
        results_expires_at=batch_row.get("expires_at"),
    )

    event_id = new_message_id()
    inserted = await conn.fetchval(
        """
        insert into webhook_events (id, customer_id, batch_id, event_type, body, created_at)
        values ($1, $2, $3, $4, $5, $6)
        on conflict (batch_id, event_type) do nothing
        returning id
        """,
        event_id,
        customer_id,
        batch_row["id"],
        EVENT_BATCH_UPDATE,
        body.decode("utf-8"),
        _utcnow(),
    )
    if inserted is None:
        return None  # already enqueued (idempotent under races + reconciler)

    now = _utcnow()
    if mode == WebhookMode.DIRECT:
        await conn.execute(
            """
            insert into webhook_deliveries
                (id, event_id, endpoint_id, customer_id, url, signed,
                 status, attempts, next_attempt_at, created_at)
            values ($1, $2, null, $3, $4, false, $5, 0, $6, $6)
            on conflict do nothing
            """,
            new_delivery_id(),
            event_id,
            customer_id,
            direct_url,
            DELIVERY_STATUS_PENDING,
            now,
        )
    else:
        for ep in endpoints:
            await conn.execute(
                """
                insert into webhook_deliveries
                    (id, event_id, endpoint_id, customer_id, url, signed,
                     status, attempts, next_attempt_at, created_at)
                values ($1, $2, $3, $4, $5, true, $6, 0, $7, $7)
                on conflict (event_id, endpoint_id) do nothing
                """,
                new_delivery_id(),
                event_id,
                ep["id"],
                customer_id,
                ep["url"],
                DELIVERY_STATUS_PENDING,
                now,
            )
    return event_id


# --- Dispatcher side -----------------------------------------------------------


@dataclass(frozen=True)
class ClaimedDelivery:
    delivery_id: str
    event_id: str  # == the svix-id message id
    endpoint_id: str | None
    customer_id: str
    url: str
    signed: bool
    attempts: int
    lease_token: str
    body: bytes
    event_type: str
    # Endpoint fields (None for direct rows). `enabled` is re-read at claim
    # time — the disable kill switch.
    endpoint_enabled: bool | None
    endpoint_url: str | None
    secret_ciphertext: str | None
    secret_key_id: str | None
    old_secret_ciphertext: str | None
    old_secret_expires_at: datetime | None


class WebhookRepo:
    """asyncpg-backed access to the webhook outbox. One per process."""

    def __init__(self, *, dsn: str | None) -> None:
        self._dsn = dsn
        self._pool: Any = None

    @property
    def configured(self) -> bool:
        return bool(self._dsn)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self):
        if not self._dsn:
            raise RuntimeError("DATABASE_URL must be set for the webhook dispatcher.")
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=5)
        return self._pool

    async def claim_next_delivery(self, *, lease_seconds: int) -> ClaimedDelivery | None:
        pool = await self._get_pool()
        lease_token = f"whl_{uuid.uuid4().hex}"
        now = _utcnow()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                select d.id from webhook_deliveries d
                 where (d.status = $1 and d.next_attempt_at <= $2)
                    or (d.status = $3 and d.lease_expires_at < $2)
                 order by d.next_attempt_at
                 for update skip locked
                 limit 1
                """,
                DELIVERY_STATUS_PENDING,
                now,
                DELIVERY_STATUS_DELIVERING,
            )
            if row is None:
                return None
            claimed = await conn.fetchrow(
                """
                update webhook_deliveries d
                   set status = $2, lease_token = $3, lease_expires_at = $4
                 where d.id = $1
                returning d.*
                """,
                row["id"],
                DELIVERY_STATUS_DELIVERING,
                lease_token,
                now + timedelta(seconds=lease_seconds),
            )
            event = await conn.fetchrow(
                "select id, event_type, body from webhook_events where id = $1",
                claimed["event_id"],
            )
            endpoint = None
            if claimed["endpoint_id"] is not None:
                endpoint = await conn.fetchrow(
                    """
                    select enabled, deleted_at, url, secret_ciphertext, secret_key_id,
                           old_secret_ciphertext, old_secret_expires_at
                      from webhook_endpoints where id = $1
                    """,
                    claimed["endpoint_id"],
                )
        enabled: bool | None = None
        if endpoint is not None:
            enabled = bool(endpoint["enabled"]) and endpoint["deleted_at"] is None
        return ClaimedDelivery(
            delivery_id=claimed["id"],
            event_id=claimed["event_id"],
            endpoint_id=claimed["endpoint_id"],
            customer_id=claimed["customer_id"],
            url=claimed["url"],
            signed=bool(claimed["signed"]),
            attempts=int(claimed["attempts"]),
            lease_token=lease_token,
            body=event["body"].encode("utf-8"),
            event_type=event["event_type"],
            endpoint_enabled=enabled,
            endpoint_url=endpoint["url"] if endpoint is not None else None,
            secret_ciphertext=endpoint["secret_ciphertext"] if endpoint is not None else None,
            secret_key_id=endpoint["secret_key_id"] if endpoint is not None else None,
            old_secret_ciphertext=(
                endpoint["old_secret_ciphertext"] if endpoint is not None else None
            ),
            old_secret_expires_at=(
                endpoint["old_secret_expires_at"] if endpoint is not None else None
            ),
        )

    async def mark_succeeded(
        self, *, delivery_id: str, lease_token: str, status_code: int
    ) -> bool:
        pool = await self._get_pool()
        result = await pool.execute(
            """
            update webhook_deliveries
               set status = $3, last_status_code = $4, last_error = null,
                   attempts = attempts + 1, succeeded_at = $5,
                   lease_token = null, lease_expires_at = null
             where id = $1 and lease_token = $2 and status = $6
            """,
            delivery_id,
            lease_token,
            DELIVERY_STATUS_SUCCEEDED,
            status_code,
            _utcnow(),
            DELIVERY_STATUS_DELIVERING,
        )
        return result.endswith(" 1")

    async def mark_attempt_failed(
        self,
        *,
        delivery_id: str,
        lease_token: str,
        status_code: int | None,
        error: str | None,
    ) -> str:
        """Record a failed attempt; schedule the next per the ladder or mark
        the delivery terminally failed. Returns the resulting status."""
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                select attempts from webhook_deliveries
                 where id = $1 and lease_token = $2 and status = $3
                 for update
                """,
                delivery_id,
                lease_token,
                DELIVERY_STATUS_DELIVERING,
            )
            if row is None:
                return "lost_lease"
            attempts_done = int(row["attempts"]) + 1
            delay = next_attempt_delay_seconds(attempts_done)
            new_status = DELIVERY_STATUS_PENDING if delay is not None else DELIVERY_STATUS_FAILED
            next_at = _utcnow() + timedelta(seconds=delay or 0)
            await conn.execute(
                """
                update webhook_deliveries
                   set status = $2, attempts = $3, next_attempt_at = $4,
                       last_status_code = $5, last_error = $6,
                       lease_token = null, lease_expires_at = null
                 where id = $1
                """,
                delivery_id,
                new_status,
                attempts_done,
                next_at,
                status_code,
                (error or "")[:1000] or None,
            )
            return new_status

    async def mark_cancelled(self, *, delivery_id: str, lease_token: str) -> bool:
        """Endpoint disabled/deleted at send time — the kill switch."""
        pool = await self._get_pool()
        result = await pool.execute(
            """
            update webhook_deliveries
               set status = $3, lease_token = null, lease_expires_at = null,
                   last_error = 'endpoint disabled'
             where id = $1 and lease_token = $2 and status = $4
            """,
            delivery_id,
            lease_token,
            DELIVERY_STATUS_CANCELLED,
            DELIVERY_STATUS_DELIVERING,
        )
        return result.endswith(" 1")

    # --- Endpoint management (the `python -m extract.webhooks` CLI) --------

    async def list_endpoints(self, *, customer_id: str) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            select id, url, description, enabled, created_at, updated_at
              from webhook_endpoints
             where customer_id = $1 and deleted_at is null
             order by created_at
            """,
            customer_id,
        )
        return [dict(r) for r in rows]

    async def count_endpoints(self, *, customer_id: str) -> int:
        pool = await self._get_pool()
        return int(
            await pool.fetchval(
                "select count(*) from webhook_endpoints"
                " where customer_id = $1 and deleted_at is null",
                customer_id,
            )
        )

    async def get_endpoint(self, *, endpoint_id: str, customer_id: str) -> Any | None:
        pool = await self._get_pool()
        return await pool.fetchrow(
            """
            select * from webhook_endpoints
             where id = $1 and customer_id = $2 and deleted_at is null
            """,
            endpoint_id,
            customer_id,
        )

    async def create_endpoint(
        self,
        *,
        endpoint_id: str,
        customer_id: str,
        url: str,
        description: str | None,
        secret_ciphertext: str,
        secret_key_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            insert into webhook_endpoints
                (id, customer_id, url, description, secret_ciphertext, secret_key_id,
                 enabled, created_at, updated_at)
            values ($1, $2, $3, $4, $5, $6, true, $7, $7)
            returning id, url, description, enabled, created_at, updated_at
            """,
            endpoint_id,
            customer_id,
            url,
            description,
            secret_ciphertext,
            secret_key_id,
            _utcnow(),
        )
        return dict(row)

    async def update_endpoint(
        self,
        *,
        endpoint_id: str,
        customer_id: str,
        url: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            update webhook_endpoints
               set url = coalesce($3, url),
                   description = coalesce($4, description),
                   enabled = coalesce($5, enabled),
                   updated_at = $6
             where id = $1 and customer_id = $2 and deleted_at is null
            returning id, url, description, enabled, created_at, updated_at
            """,
            endpoint_id,
            customer_id,
            url,
            description,
            enabled,
            _utcnow(),
        )
        return dict(row) if row else None

    async def soft_delete_endpoint(self, *, endpoint_id: str, customer_id: str) -> bool:
        """Soft delete: queued deliveries keep a valid signing key; the
        dispatcher's claim-time enabled/deleted check stops future sends."""
        pool = await self._get_pool()
        result = await pool.execute(
            """
            update webhook_endpoints
               set deleted_at = $3, enabled = false, updated_at = $3
             where id = $1 and customer_id = $2 and deleted_at is null
            """,
            endpoint_id,
            customer_id,
            _utcnow(),
        )
        return result.endswith(" 1")

    async def rotate_endpoint_secret(
        self,
        *,
        endpoint_id: str,
        customer_id: str,
        new_ciphertext: str,
        new_key_id: str,
        overlap_hours: int = 24,
    ) -> bool:
        """New secret takes over; the previous one keeps verifying for the
        dual-signing overlap window."""
        pool = await self._get_pool()
        result = await pool.execute(
            """
            update webhook_endpoints
               set old_secret_ciphertext = secret_ciphertext,
                   old_secret_expires_at = $3,
                   secret_ciphertext = $4,
                   secret_key_id = $5,
                   updated_at = $6
             where id = $1 and customer_id = $2 and deleted_at is null
            """,
            endpoint_id,
            customer_id,
            _utcnow() + timedelta(hours=overlap_hours),
            new_ciphertext,
            new_key_id,
            _utcnow(),
        )
        return result.endswith(" 1")

    # --- Deliveries browse / ping / resend ---------------------------------

    async def create_ping(self, *, customer_id: str, endpoint_id: str) -> str | None:
        """Synthetic webhook.ping event + one delivery through the normal
        outbox (batch_id NULL — NULLs are distinct under the unique key)."""
        from extract.core.webhooks import EVENT_PING, build_ping_body

        pool = await self._get_pool()
        endpoint = await self.get_endpoint(endpoint_id=endpoint_id, customer_id=customer_id)
        if endpoint is None:
            return None
        event_id = new_message_id()
        now = _utcnow()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                insert into webhook_events
                    (id, customer_id, batch_id, event_type, body, created_at)
                values ($1, $2, null, $3, $4, $5)
                """,
                event_id,
                customer_id,
                EVENT_PING,
                build_ping_body(event_id.removeprefix("msg_")).decode("utf-8"),
                now,
            )
            await conn.execute(
                """
                insert into webhook_deliveries
                    (id, event_id, endpoint_id, customer_id, url, signed,
                     status, attempts, next_attempt_at, created_at)
                values ($1, $2, $3, $4, $5, true, $6, 0, $7, $7)
                """,
                new_delivery_id(),
                event_id,
                endpoint_id,
                customer_id,
                endpoint["url"],
                DELIVERY_STATUS_PENDING,
                now,
            )
        return event_id

    async def list_deliveries(
        self, *, customer_id: str, limit: int = 50, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            select d.id, d.event_id, d.endpoint_id, d.signed, d.status,
                   d.attempts, d.next_attempt_at, d.last_status_code,
                   d.last_error, d.created_at, d.succeeded_at,
                   e.event_type, e.batch_id,
                   b.expires_at as batch_expires_at
              from webhook_deliveries d
              join webhook_events e on e.id = d.event_id
              left join extract_batches b on b.id = e.batch_id
             where d.customer_id = $1
               and ($2::timestamp is null or d.created_at < $2)
             order by d.created_at desc
             limit $3
            """,
            customer_id,
            before,
            limit,
        )
        return [dict(r) for r in rows]

    async def resend_delivery(self, *, delivery_id: str, customer_id: str) -> str:
        """Re-queue a settled delivery with a fresh ladder, or pull a pending
        row's next scheduled attempt forward ("retry now"). Refused while an
        attempt is actively in flight or once the batch's results expired."""
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                select d.id, d.status, b.expires_at
                  from webhook_deliveries d
                  join webhook_events e on e.id = d.event_id
                  left join extract_batches b on b.id = e.batch_id
                 where d.id = $1 and d.customer_id = $2
                 for update of d
                """,
                delivery_id,
                customer_id,
            )
            if row is None:
                return "not_found"
            if row["status"] == DELIVERY_STATUS_DELIVERING:
                return "in_flight"
            if row["expires_at"] is not None and row["expires_at"] < _utcnow():
                return "results_expired"
            if row["status"] == DELIVERY_STATUS_PENDING:
                # Retry now: skip the backoff wait but keep the ladder position,
                # so repeated clicks can't keep a dying delivery alive forever.
                # Distinct outcome: the row may have settled between the client's
                # render and the click, and the caller should report which
                # operation actually ran.
                await conn.execute(
                    "update webhook_deliveries set next_attempt_at = $2 where id = $1",
                    delivery_id,
                    _utcnow(),
                )
                return "nudged"
            await conn.execute(
                """
                update webhook_deliveries
                   set status = $2, attempts = 0, next_attempt_at = $3,
                       lease_token = null, lease_expires_at = null,
                       last_error = null
                 where id = $1
                """,
                delivery_id,
                DELIVERY_STATUS_PENDING,
                _utcnow(),
            )
            return "requeued"

    async def oldest_pending_age_seconds(self) -> float | None:
        """Backlog-health gauge: age of the oldest due, undelivered
        row. A dead dispatcher shows up here even while item processing is
        healthy."""
        pool = await self._get_pool()
        oldest = await pool.fetchval(
            """
            select min(next_attempt_at) from webhook_deliveries
             where status = $1
            """,
            DELIVERY_STATUS_PENDING,
        )
        if oldest is None:
            return None
        return max(0.0, (_utcnow() - oldest).total_seconds())

    # --- Reconciler -------------------------

    async def reconcile_missing_events(self) -> int:
        """Enqueue events for opted-in batches that went terminal without one
        (rolling-deploy window, missed-enqueue bugs). Bounded to the lookback
        window W; per-mode gating mirrors ``enqueue_batch_event`` exactly.
        Any nonzero return is alarm-worthy — the live enqueue missed."""
        pool = await self._get_pool()
        lookback = timedelta(hours=settings.EXTRACT_WEBHOOK_RECONCILE_LOOKBACK_HOURS)
        grace = timedelta(minutes=settings.EXTRACT_WEBHOOK_RECONCILE_GRACE_MINUTES)
        now = _utcnow()
        candidates = await pool.fetch(
            """
            select b.id from extract_batches b
             where b.status = any($1::text[])
               and b.completed_at between $2 and $3
               and b.webhook_mode in ('svix', 'registered', 'direct')
               and not exists (
                   select 1 from webhook_events e
                    where e.batch_id = b.id and e.event_type = $4
               )
             limit 100
            """,
            list(BatchStatus.TERMINAL),
            now - lookback,
            now - grace,
            EVENT_BATCH_UPDATE,
        )
        enqueued = 0
        for candidate in candidates:
            async with pool.acquire() as conn, conn.transaction():
                batch = await conn.fetchrow(
                    "select * from extract_batches where id = $1 for update",
                    candidate["id"],
                )
                if batch is None or batch["completed_at"] is None:
                    continue
                if await enqueue_batch_event(conn, batch_row=batch) is not None:
                    enqueued += 1
        return enqueued


def from_settings() -> WebhookRepo:
    return WebhookRepo(dsn=settings.DATABASE_URL)


__all__ = ["ClaimedDelivery", "WebhookRepo", "enqueue_batch_event", "from_settings"]
