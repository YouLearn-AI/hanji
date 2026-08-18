"""Async-batch data access — files, batches, batch items.

All persistent state for the async pipeline lives in three Postgres
tables. Schema is authored in ``migrations/`` (applied by
``python -m extract.migrate``). Here we just talk to the rows.

Design notes:

- One shared asyncpg pool per process, lazy-created on first use.
- All claim / state-change paths are written as single statements that
  return enough context for the caller to decide what's next; we avoid
  multi-statement transactions where one statement plus `RETURNING` does
  the same job.
- The worker claim folds the `run_after` backoff into the WHERE clause.
  Lock-then-skip-then-release is never used.
- `counts_json` on :class:`extract_batches` is denormalized so polling
  can return per-batch counts without GROUPing items. Updates happen in
  the same transaction as the per-item state change, via a CTE — see
  :meth:`BatchRepo.update_item_terminal`.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from extract.config import settings
from extract.core.batch import (
    ZERO_COUNTS,
    BatchStatus,
    ItemStatus,
    derive_batch_status,
)
from extract.repos.webhooks import enqueue_batch_event

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Naive UTC datetime to match the schema's `timestamp` (without TZ).

    The new `extract_*` tables follow the same convention as `phi_*` —
    columns are declared `TIMESTAMP WITHOUT TIME ZONE` and store UTC
    by convention. asyncpg refuses to bind tz-aware datetimes against
    tz-naive columns, so we strip the tzinfo at the boundary.
    """
    return datetime.now(UTC).replace(tzinfo=None)

DEFAULT_LEASE_SECONDS = 300  # 5 min — covers a long extraction
# 3 days — gives callers a working weekend window between upload and
# processing. Keep in sync with any object-lifecycle rule on the buckets.
DEFAULT_FILE_TTL_SECONDS = 3 * 24 * 3600
DEFAULT_BATCH_TTL_SECONDS = 3 * 24 * 3600


# --- ID generation ----------------------------------------------------------
# Mirrors the OpenAI / Anthropic / Stripe `prefix_<random>` style. URL-safe,
# unique, no PII. Long enough that brute-forcing /v1/files/{file_id} is not
# a thing — 22 random chars × ~6 bits ≈ 132 bits of entropy.

def _random_suffix(n: int = 22) -> str:
    return secrets.token_urlsafe(n)[:n]


def new_file_id() -> str:
    return f"file_{_random_suffix()}"


def new_batch_id() -> str:
    return f"batch_{_random_suffix()}"


def new_item_id() -> str:
    return f"item_{_random_suffix()}"


def new_lease_token() -> str:
    return secrets.token_urlsafe(16)


# --- File status (DB-specific, not part of the public API contract) --------

class FileStatus:
    PENDING_UPLOAD = "pending_upload"
    UPLOADED = "uploaded"
    EXPIRED = "expired"


# --- Row dataclasses --------------------------------------------------------

@dataclass(frozen=True)
class FileRow:
    id: str
    customer_id: str
    phi_safe: bool
    s3_bucket: str
    s3_key: str
    filename: str | None
    content_type: str | None
    size_bytes: int
    sha256: str | None
    status: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class BatchRow:
    id: str
    customer_id: str
    phi_safe: bool
    status: str
    idempotency_key: str | None
    total_items: int
    counts: dict[str, int]
    metadata: dict[str, Any] | None
    engine: str | None
    extract_text: bool
    extract_images: bool
    ocr: str
    table_output_format: str
    # Plan 077 opt-in RAG chunking ("none" | "semantic") + target size (chars).
    chunking: str
    chunk_size: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    expires_at: datetime


@dataclass(frozen=True)
class BatchItemRow:
    id: str
    batch_id: str
    customer_id: str
    file_id: str | None
    position: int
    status: str
    page_count: int | None
    error_code: str | None
    error_message: str | None
    result_s3_bucket: str | None
    result_s3_key: str | None
    attempts: int
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime
    # url-sourced items only (mutually exclusive with file_id). Full url incl.
    # query string — a presigned signature lives there — so this is DB/worker
    # internal only; project_item() redacts it before it reaches a response.
    url: str | None = None


@dataclass(frozen=True)
class ClaimedItem:
    """A successfully-claimed batch item, with everything the worker needs.

    The claim query joins to `extract_files` so the worker doesn't have to
    fetch the file row separately. Exactly one of `file_id` (+ the `file_*`
    fields) or `url` is set, mirroring FilesSource/UrlSource on the request.
    """

    item_id: str
    batch_id: str
    customer_id: str
    phi_safe: bool
    file_id: str | None
    file_s3_bucket: str | None
    file_s3_key: str | None
    file_filename: str | None
    file_content_type: str | None
    lease_token: str
    lease_expires_at: datetime
    attempts: int
    extract_text: bool
    extract_images: bool
    ocr: str
    engine: str
    url: str | None = None
    # Opt-in table structure of the batch ("markdown" | "html").
    table_output_format: str = "markdown"
    # Opt-in RAG chunking ("none" | "semantic") + target size (chars).
    chunking: str = "none"
    chunk_size: int = 1000


@dataclass(frozen=True)
class BatchItemSpec:
    """One item to create, from either source type. Exactly one of `file_id`
    or `url` is set — callers (the route handlers) build these from either
    already-uploaded `FileRow`s or raw URLs before calling `create_batch`."""

    file_id: str | None = None
    url: str | None = None


# --- Repository -------------------------------------------------------------


class BatchRepo:
    """asyncpg-backed access to extract_files / extract_batches / extract_batch_items.

    One instance per process; pool is created lazily.
    """

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
            raise RuntimeError("DATABASE_URL must be set for the async batch API.")
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=10)
        return self._pool

    # --- Files ----------------------------------------------------------

    async def create_file(
        self,
        *,
        customer_id: str,
        phi_safe: bool,
        s3_bucket: str,
        s3_key: str,
        filename: str | None,
        content_type: str | None,
        size_bytes: int,
        file_id: str | None = None,
        ttl_seconds: int = DEFAULT_FILE_TTL_SECONDS,
    ) -> FileRow:
        file_id = file_id or new_file_id()
        now = _utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            insert into extract_files (
                id, customer_id, phi_safe, s3_bucket, s3_key,
                filename, content_type, size_bytes, status, created_at, expires_at
            ) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            returning *
            """,
            file_id,
            customer_id,
            phi_safe,
            s3_bucket,
            s3_key,
            filename,
            content_type,
            size_bytes,
            FileStatus.PENDING_UPLOAD,
            now,
            expires_at,
        )
        return _file_row(row)

    async def mark_file_uploaded(
        self,
        *,
        file_id: str,
        customer_id: str,
        sha256: str | None = None,
    ) -> FileRow | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            update extract_files
               set status = $3,
                   sha256 = coalesce($4, sha256)
             where id = $1
               and customer_id = $2
               and status = $5
            returning *
            """,
            file_id,
            customer_id,
            FileStatus.UPLOADED,
            sha256,
            FileStatus.PENDING_UPLOAD,
        )
        return _file_row(row) if row else None

    async def get_file(self, *, file_id: str, customer_id: str) -> FileRow | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "select * from extract_files where id = $1 and customer_id = $2",
            file_id,
            customer_id,
        )
        return _file_row(row) if row else None

    # --- Batches --------------------------------------------------------

    async def find_batch_by_idempotency(
        self,
        *,
        customer_id: str,
        idempotency_key: str,
    ) -> BatchRow | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            select * from extract_batches
             where customer_id = $1 and idempotency_key = $2
            """,
            customer_id,
            idempotency_key,
        )
        return _batch_row(row) if row else None

    async def create_batch(
        self,
        *,
        customer_id: str,
        phi_safe: bool,
        idempotency_key: str | None,
        engine: str | None,
        extract_text: bool,
        extract_images: bool,
        ocr: str,
        metadata: dict[str, Any] | None,
        items: Sequence[BatchItemSpec],
        table_output_format: str = "markdown",
        chunking: str = "none",
        chunk_size: int = 1000,
        webhook_mode: str | None = None,
        webhook_url: str | None = None,
        ttl_seconds: int = DEFAULT_BATCH_TTL_SECONDS,
    ) -> tuple[BatchRow, list[BatchItemRow]]:
        """Create a batch + N items in one transaction.

        Each `items[i]` is a `BatchItemSpec` with either `file_id` (an
        already-uploaded FileRow owned by this customer, in `uploaded`
        status) or `url` set; callers resolve/validate the source before
        calling here.
        """
        if not items:
            raise ValueError("create_batch requires at least one item")

        batch_id = new_batch_id()
        now = _utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        counts = dict(ZERO_COUNTS)
        counts["pending"] = len(items)

        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            # Fully parameterized: the only dynamic values ride asyncpg
            # parameters, never the SQL string.
            batch = await conn.fetchrow(
                """
                insert into extract_batches (
                    id, customer_id, phi_safe, status, idempotency_key,
                    total_items, counts_json, metadata_json,
                    engine, extract_text, extract_images, ocr, table_output_format,
                    chunking, chunk_size,
                    created_at, expires_at, webhook_mode, webhook_url
                ) values ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
                returning *
                """,
                batch_id,
                customer_id,
                phi_safe,
                BatchStatus.PENDING,
                idempotency_key,
                len(items),
                json.dumps(counts),
                json.dumps(metadata) if metadata is not None else None,
                engine,
                extract_text,
                extract_images,
                ocr,
                table_output_format,
                chunking,
                chunk_size,
                now,
                expires_at,
                webhook_mode,
                webhook_url,
            )
            item_rows: list[BatchItemRow] = []
            stmt = await conn.prepare(
                """
                insert into extract_batch_items (
                    id, batch_id, customer_id, file_id, url, position,
                    status, attempts, updated_at
                ) values ($1,$2,$3,$4,$5,$6,$7,0,$8)
                returning *
                """
            )
            for position, spec in enumerate(items):
                row = await stmt.fetchrow(
                    new_item_id(),
                    batch_id,
                    customer_id,
                    spec.file_id,
                    spec.url,
                    position,
                    ItemStatus.PENDING,
                    now,
                )
                item_rows.append(_batch_item_row(row))
        return _batch_row(batch), item_rows

    async def get_batch(self, *, batch_id: str, customer_id: str) -> BatchRow | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "select * from extract_batches where id = $1 and customer_id = $2",
            batch_id,
            customer_id,
        )
        return _batch_row(row) if row else None

    async def list_batches(
        self,
        *,
        customer_id: str,
        status: str | None = None,
        limit: int = 50,
        cursor_created_at: datetime | None = None,
        cursor_id: str | None = None,
    ) -> list[BatchRow]:
        pool = await self._get_pool()
        if cursor_created_at is None:
            rows = await pool.fetch(
                """
                select * from extract_batches
                 where customer_id = $1
                   and ($2::text is null or status = $2)
                 order by created_at desc, id desc
                 limit $3
                """,
                customer_id,
                status,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                select * from extract_batches
                 where customer_id = $1
                   and ($2::text is null or status = $2)
                   and (created_at, id) < ($3, $4)
                 order by created_at desc, id desc
                 limit $5
                """,
                customer_id,
                status,
                cursor_created_at,
                cursor_id,
                limit,
            )
        return [_batch_row(r) for r in rows]

    # --- Batch items ----------------------------------------------------

    async def get_batch_item(
        self,
        *,
        batch_id: str,
        item_id: str,
        customer_id: str,
    ) -> BatchItemRow | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            select * from extract_batch_items
             where id = $1 and batch_id = $2 and customer_id = $3
            """,
            item_id,
            batch_id,
            customer_id,
        )
        return _batch_item_row(row) if row else None

    async def list_batch_items_after(
        self,
        *,
        batch_id: str,
        customer_id: str,
        limit: int = 100,
        cursor_updated_at: datetime | None = None,
        cursor_id: str | None = None,
    ) -> list[BatchItemRow]:
        """Polling cursor: items in (updated_at, id) ascending order."""
        pool = await self._get_pool()
        if cursor_updated_at is None:
            rows = await pool.fetch(
                """
                select * from extract_batch_items
                 where batch_id = $1 and customer_id = $2
                 order by updated_at asc, id asc
                 limit $3
                """,
                batch_id,
                customer_id,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                select * from extract_batch_items
                 where batch_id = $1 and customer_id = $2
                   and (updated_at, id) > ($3, $4)
                 order by updated_at asc, id asc
                 limit $5
                """,
                batch_id,
                customer_id,
                cursor_updated_at,
                cursor_id,
                limit,
            )
        return [_batch_item_row(r) for r in rows]

    async def cancel_batch(
        self,
        *,
        batch_id: str,
        customer_id: str,
    ) -> BatchRow | None:
        """Flip remaining `pending` items to `cancelled`. Running items finish
        on their own; the batch state machine settles when their workers update.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            batch = await conn.fetchrow(
                """
                select * from extract_batches
                 where id = $1 and customer_id = $2
                 for update
                """,
                batch_id,
                customer_id,
            )
            if batch is None:
                return None
            if batch["status"] in BatchStatus.TERMINAL:
                return _batch_row(batch)
            now = _utcnow()
            await conn.execute(
                """
                update extract_batch_items
                   set status = $3,
                       completed_at = coalesce(completed_at, $4),
                       updated_at = $4
                 where batch_id = $1 and customer_id = $2 and status = $5
                """,
                batch_id,
                customer_id,
                ItemStatus.CANCELLED,
                now,
                ItemStatus.PENDING,
            )
            counts = await _recompute_counts(conn, batch_id)
            new_status = derive_batch_status(counts, current=batch["status"])
            updated = await conn.fetchrow(
                """
                update extract_batches
                   set counts_json = $2::jsonb,
                       status = $3,
                       completed_at = case
                           when $3 in ($4,$5,$6,$7,$8) and completed_at is null
                             then $9 else completed_at end
                 where id = $1
                returning *
                """,
                batch_id,
                json.dumps(counts),
                new_status,
                BatchStatus.COMPLETED,
                BatchStatus.PARTIALLY_FAILED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
                BatchStatus.EXPIRED,
                now,
            )
            # Webhook enqueue on the null→set flip. The batch
            # row was selected FOR UPDATE above, so `batch["completed_at"]`
            # is the authoritative prior value. A cancel that leaves running
            # items enqueues nothing here — the last item settles through
            # _update_item_terminal, which carries its own enqueue.
            if batch["completed_at"] is None and updated["completed_at"] is not None:
                await enqueue_batch_event(conn, batch_row=updated)
            return _batch_row(updated)

    # --- Worker claim + state machine -----------------------------------

    async def count_claimable_items(self) -> int:
        """Backlog gauge for autoscaling: items a worker could claim right now.

        Deliberately mirrors the *eligibility* half of ``claim_next_item``'s CTE
        — ready pending items plus running items whose lease has expired — so the
        metric measures the same queue the workers actually pull from. It does
        NOT apply the per-customer concurrency cap: that cap is a fairness limit
        on one customer, and excluding capped work would under-report the backlog
        and suppress scale-out exactly when one customer floods the queue.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """
                select count(*) from extract_batch_items i
                 where (
                         (i.status = $1 and (i.run_after is null or i.run_after <= now()))
                      or (i.status = $2 and i.lease_expires_at is not null
                                       and i.lease_expires_at <= now())
                       )
                """,
                ItemStatus.PENDING,
                ItemStatus.RUNNING,
            )
        return int(value or 0)

    async def claim_next_item(
        self,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> ClaimedItem | None:
        """Claim one ready item (or revive a stale-leased running one).

        Single-tenant: there is no per-customer fairness cap — total
        concurrency is bounded by the worker's slot count. `run_after` is
        folded into the WHERE so we never lock-then-skip. Returns ``None``
        if nothing is ready.
        """
        lease_token = new_lease_token()
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            with eligible as (
                select i.id
                  from extract_batch_items i
                 where (
                         (i.status = $1 and (i.run_after is null or i.run_after <= now()))
                      or (i.status = $2 and i.lease_expires_at is not null
                                       and i.lease_expires_at <= now())
                       )
                 order by i.id
                 limit 1
                 for update of i skip locked
            )
            update extract_batch_items i
               set status = $2,
                   lease_token = $3,
                   lease_expires_at = now() + ($4 || ' seconds')::interval,
                   attempts = i.attempts + 1,
                   started_at = coalesce(i.started_at, now()),
                   updated_at = now()
              from eligible
             where i.id = eligible.id
            returning
                i.id            as item_id,
                i.batch_id      as batch_id,
                i.customer_id   as customer_id,
                i.file_id       as file_id,
                i.url           as url,
                i.attempts      as attempts,
                i.lease_token   as lease_token,
                i.lease_expires_at as lease_expires_at,
                (select phi_safe from extract_batches b where b.id = i.batch_id) as phi_safe,
                (select extract_text from extract_batches b where b.id = i.batch_id) as extract_text,
                (select extract_images from extract_batches b where b.id = i.batch_id) as extract_images,
                (select ocr from extract_batches b where b.id = i.batch_id) as ocr,
                (select table_output_format from extract_batches b where b.id = i.batch_id)
                    as table_output_format,
                (select chunking from extract_batches b where b.id = i.batch_id) as chunking,
                (select chunk_size from extract_batches b where b.id = i.batch_id) as chunk_size,
                (select coalesce(b.engine, 'baseline') from extract_batches b
                  where b.id = i.batch_id) as engine,
                (select s3_bucket from extract_files f where f.id = i.file_id) as file_s3_bucket,
                (select s3_key from extract_files f where f.id = i.file_id) as file_s3_key,
                (select filename from extract_files f where f.id = i.file_id) as file_filename,
                (select content_type from extract_files f where f.id = i.file_id) as file_content_type
            """,
            ItemStatus.PENDING,
            ItemStatus.RUNNING,
            lease_token,
            str(lease_seconds),
        )
        if row is None:
            return None
        # Bump batch.status -> running on first item claim. Cheap; idempotent.
        await pool.execute(
            """
            update extract_batches
               set status = $2,
                   started_at = coalesce(started_at, now())
             where id = $1 and status = $3
            """,
            row["batch_id"],
            BatchStatus.RUNNING,
            BatchStatus.PENDING,
        )
        return ClaimedItem(
            item_id=row["item_id"],
            batch_id=row["batch_id"],
            customer_id=row["customer_id"],
            phi_safe=bool(row["phi_safe"]),
            file_id=row["file_id"],
            url=row["url"],
            file_s3_bucket=row["file_s3_bucket"],
            file_s3_key=row["file_s3_key"],
            file_filename=row["file_filename"],
            file_content_type=row["file_content_type"],
            lease_token=row["lease_token"],
            lease_expires_at=row["lease_expires_at"],
            attempts=row["attempts"],
            extract_text=bool(row["extract_text"]),
            extract_images=bool(row["extract_images"]),
            ocr=row["ocr"],
            engine=row["engine"] or "baseline",
            table_output_format=row["table_output_format"] or "markdown",
            chunking=row["chunking"] or "none",
            chunk_size=row["chunk_size"] or 1000,
        )

    async def heartbeat_lease(
        self,
        *,
        item_id: str,
        lease_token: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        pool = await self._get_pool()
        result = await pool.execute(
            """
            update extract_batch_items
               set lease_expires_at = now() + ($3 || ' seconds')::interval,
                   updated_at = now()
             where id = $1 and lease_token = $2 and status = $4
            """,
            item_id,
            lease_token,
            str(lease_seconds),
            ItemStatus.RUNNING,
        )
        # asyncpg returns "UPDATE n"
        return result.endswith(" 1")

    async def update_item_succeeded(
        self,
        *,
        item_id: str,
        lease_token: str,
        page_count: int,
        result_s3_bucket: str,
        result_s3_key: str,
    ) -> BatchRow | None:
        return await self._update_item_terminal(
            item_id=item_id,
            lease_token=lease_token,
            new_status=ItemStatus.SUCCEEDED,
            page_count=page_count,
            error_code=None,
            error_message=None,
            result_s3_bucket=result_s3_bucket,
            result_s3_key=result_s3_key,
            run_after=None,
            release_lease=True,
        )

    async def update_item_failed(
        self,
        *,
        item_id: str,
        lease_token: str,
        error_code: str,
        error_message: str | None,
        page_count: int | None = None,
    ) -> BatchRow | None:
        return await self._update_item_terminal(
            item_id=item_id,
            lease_token=lease_token,
            new_status=ItemStatus.FAILED,
            page_count=page_count,
            error_code=error_code,
            error_message=error_message,
            result_s3_bucket=None,
            result_s3_key=None,
            run_after=None,
            release_lease=True,
        )

    async def reschedule_item(
        self,
        *,
        item_id: str,
        lease_token: str,
        backoff_seconds: int,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Return a still-running item to `pending` with `run_after` set.

        Used when the worker hits a transient failure (Baseten 5xx/429,
        S3 timeout, etc.) and wants to retry later. The `attempts`
        counter is incremented; callers cap it themselves before deciding
        whether to fail the item terminally instead of rescheduling.
        """
        pool = await self._get_pool()
        run_after = _utcnow() + timedelta(seconds=backoff_seconds)
        result = await pool.execute(
            """
            update extract_batch_items
               set status = $3,
                   run_after = $4,
                   error_code = $5,
                   error_message = $6,
                   lease_token = null,
                   lease_expires_at = null,
                   updated_at = now()
             where id = $1 and lease_token = $2 and status = $7
            """,
            item_id,
            lease_token,
            ItemStatus.PENDING,
            run_after,
            error_code,
            error_message,
            ItemStatus.RUNNING,
        )
        return result.endswith(" 1")

    async def _update_item_terminal(
        self,
        *,
        item_id: str,
        lease_token: str,
        new_status: str,
        page_count: int | None,
        error_code: str | None,
        error_message: str | None,
        result_s3_bucket: str | None,
        result_s3_key: str | None,
        run_after: datetime | None,
        release_lease: bool,
    ) -> BatchRow | None:
        """Apply a terminal item state and recompute the parent batch counts.

        Returns the updated parent batch row (so callers know whether the
        batch itself reached a terminal state).
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            updated = await conn.fetchrow(
                """
                update extract_batch_items
                   set status = $3,
                       page_count = coalesce($4, page_count),
                       error_code = $5,
                       error_message = $6,
                       result_s3_bucket = coalesce($7, result_s3_bucket),
                       result_s3_key = coalesce($8, result_s3_key),
                       run_after = $9,
                       lease_token = case when $10 then null else lease_token end,
                       lease_expires_at = case when $10 then null else lease_expires_at end,
                       completed_at = now(),
                       updated_at = now()
                 where id = $1 and lease_token = $2 and status = $11
                returning batch_id
                """,
                item_id,
                lease_token,
                new_status,
                page_count,
                error_code,
                error_message,
                result_s3_bucket,
                result_s3_key,
                run_after,
                release_lease,
                ItemStatus.RUNNING,
            )
            if updated is None:
                return None
            batch_id = updated["batch_id"]
            # Lock the parent BEFORE recomputing counts. Under READ COMMITTED,
            # a concurrent finisher that recomputed first would otherwise see a
            # stale non-terminal count; blocking on the lock first means our
            # post-lock recompute sees every committed sibling row, so the
            # last finisher always derives the true terminal status (and
            # `completed_at` is set exactly once).
            locked = await conn.fetchrow(
                "select status, completed_at from extract_batches where id = $1 for update",
                batch_id,
            )
            current = locked["status"]
            prior_completed_at = locked["completed_at"]
            counts = await _recompute_counts(conn, batch_id)
            new_batch_status = derive_batch_status(counts, current=current)
            now = _utcnow()
            batch_row = await conn.fetchrow(
                """
                update extract_batches
                   set counts_json = $2::jsonb,
                       status = $3,
                       completed_at = case
                           when $3 in ($4,$5,$6,$7,$8) and completed_at is null
                             then $9 else completed_at end
                 where id = $1
                returning *
                """,
                batch_id,
                json.dumps(counts),
                new_batch_status,
                BatchStatus.COMPLETED,
                BatchStatus.PARTIALLY_FAILED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
                BatchStatus.EXPIRED,
                now,
            )
            # Webhook enqueue: exactly on the null→set flip,
            # atomically with the terminal state change. We hold the batch
            # row lock, so the flip is observed by exactly one finisher.
            if prior_completed_at is None and batch_row["completed_at"] is not None:
                await enqueue_batch_event(conn, batch_row=batch_row)
            return _batch_row(batch_row)


# --- Helpers ----------------------------------------------------------------


async def _recompute_counts(conn, batch_id: str) -> dict[str, int]:
    rows = await conn.fetch(
        """
        select status, count(*) as n
          from extract_batch_items
         where batch_id = $1
         group by status
        """,
        batch_id,
    )
    counts = dict(ZERO_COUNTS)
    for row in rows:
        counts[row["status"]] = int(row["n"])
    return counts


# `derive_batch_status` lives in extract.core.batch (pure helper); imported above.


def _file_row(row) -> FileRow:
    return FileRow(
        id=row["id"],
        customer_id=row["customer_id"],
        phi_safe=bool(row["phi_safe"]),
        s3_bucket=row["s3_bucket"],
        s3_key=row["s3_key"],
        filename=row["filename"],
        content_type=row["content_type"],
        size_bytes=int(row["size_bytes"]),
        sha256=row["sha256"],
        status=row["status"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


def _batch_row(row) -> BatchRow:
    counts_raw = row["counts_json"]
    counts = json.loads(counts_raw) if isinstance(counts_raw, str) else dict(counts_raw or {})
    metadata_raw = row["metadata_json"]
    metadata: dict[str, Any] | None
    if metadata_raw is None:
        metadata = None
    elif isinstance(metadata_raw, str):
        metadata = json.loads(metadata_raw)
    else:
        metadata = dict(metadata_raw)
    return BatchRow(
        id=row["id"],
        customer_id=row["customer_id"],
        phi_safe=bool(row["phi_safe"]),
        status=row["status"],
        idempotency_key=row["idempotency_key"],
        total_items=int(row["total_items"]),
        counts={k: int(v) for k, v in {**ZERO_COUNTS, **counts}.items()},
        metadata=metadata,
        engine=row["engine"],
        extract_text=bool(row["extract_text"]),
        extract_images=bool(row["extract_images"]),
        ocr=row["ocr"],
        # .get(): asyncpg Record and plain dicts both support it;
        # missing column ⇒ default.
        table_output_format=row.get("table_output_format") or "markdown",
        chunking=row.get("chunking") or "none",
        chunk_size=row.get("chunk_size") or 1000,
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        expires_at=row["expires_at"],
    )


def _batch_item_row(row) -> BatchItemRow:
    return BatchItemRow(
        id=row["id"],
        batch_id=row["batch_id"],
        customer_id=row["customer_id"],
        file_id=row["file_id"],
        position=int(row["position"]),
        status=row["status"],
        page_count=row["page_count"] if row["page_count"] is None else int(row["page_count"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        result_s3_bucket=row["result_s3_bucket"],
        result_s3_key=row["result_s3_key"],
        attempts=int(row["attempts"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        updated_at=row["updated_at"],
        # .get(): tolerant of a rolling deploy racing the url-source migration
        # (same convention as table_output_format/chunking above).
        url=row.get("url"),
    )


# --- Factory ---------------------------------------------------------------


def from_settings() -> BatchRepo:
    return BatchRepo(dsn=settings.DATABASE_URL)


__all__ = [
    "BatchRepo",
    "BatchRow",
    "BatchItemRow",
    "BatchItemSpec",
    "BatchStatus",
    "ClaimedItem",
    "DEFAULT_FILE_TTL_SECONDS",
    "DEFAULT_BATCH_TTL_SECONDS",
    "DEFAULT_LEASE_SECONDS",
    "FileRow",
    "FileStatus",
    "ItemStatus",
    "from_settings",
    "new_batch_id",
    "new_file_id",
    "new_item_id",
    "new_lease_token",
]
