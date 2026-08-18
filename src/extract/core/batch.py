"""Pure helpers for the async batch lane — no DB, no S3, no FastAPI.

The DB-touching versions of these helpers live in
:mod:`extract.repos.batches`. Anything that can be pure lives here so it
can be unit-tested without infrastructure.

Three things are kept here:

- The batch / item state machines, exposed as enums + a transition
  function that derives the parent batch's status from per-status
  counts. Used by both the API (cancel) and the worker (item updates).
- Polling cursor encoding/decoding. We hand clients an opaque base64
  string of the form ``{updated_at_iso}|{item_id}``; this module
  encodes and decodes it so the route handler doesn't have to.
- A small set of error code constants the worker writes into
  ``extract_batch_items.error_code``. The API echoes them back to the
  client verbatim, so they're part of the public contract.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

# --- Status enums ----------------------------------------------------------
#
# These are the canonical status strings written to Postgres and returned to
# clients. The DB columns use plain `text` so adding a status is a code-only
# change; keep them stable across versions because customers will pattern-
# match on them.


class BatchStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    TERMINAL = frozenset({COMPLETED, PARTIALLY_FAILED, FAILED, CANCELLED, EXPIRED})


class ItemStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})


# Empty `counts_json` template — the API's polling response always
# contains all five keys, even when the per-status count is zero.
ZERO_COUNTS: dict[str, int] = {
    "pending": 0,
    "running": 0,
    "succeeded": 0,
    "failed": 0,
    "cancelled": 0,
}


# --- State transitions -----------------------------------------------------


_VALID_BATCH_TRANSITIONS: dict[str, frozenset[str]] = {
    BatchStatus.PENDING: frozenset(
        {
            BatchStatus.PENDING,
            BatchStatus.RUNNING,
            BatchStatus.COMPLETED,
            BatchStatus.PARTIALLY_FAILED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED,
        }
    ),
    BatchStatus.RUNNING: frozenset(
        {
            BatchStatus.RUNNING,
            BatchStatus.COMPLETED,
            BatchStatus.PARTIALLY_FAILED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED,
        }
    ),
    # Terminal states stay terminal.
    BatchStatus.COMPLETED: frozenset({BatchStatus.COMPLETED}),
    BatchStatus.PARTIALLY_FAILED: frozenset({BatchStatus.PARTIALLY_FAILED}),
    BatchStatus.FAILED: frozenset({BatchStatus.FAILED}),
    BatchStatus.CANCELLED: frozenset({BatchStatus.CANCELLED}),
    BatchStatus.EXPIRED: frozenset({BatchStatus.EXPIRED}),
}


def derive_batch_status(counts: dict[str, int], *, current: str | None) -> str:
    """Return the batch status implied by per-status item counts.

    Pure function. Mirrors the same logic in
    :func:`extract.repos.batches._derive_batch_status` (the SQL-side
    reuses this when it has access to it; the repo keeps a copy because
    it's invoked inside a transaction).
    """
    if current in BatchStatus.TERMINAL:
        return current  # idempotent; never undo a terminal state
    pending = counts.get("pending", 0)
    running = counts.get("running", 0)
    succeeded = counts.get("succeeded", 0)
    failed = counts.get("failed", 0)
    cancelled = counts.get("cancelled", 0)
    if pending == 0 and running == 0:
        if succeeded == 0 and failed == 0 and cancelled == 0:
            return BatchStatus.PENDING
        if failed == 0 and cancelled == 0:
            return BatchStatus.COMPLETED
        if succeeded == 0:
            return BatchStatus.FAILED if cancelled == 0 else BatchStatus.CANCELLED
        return BatchStatus.PARTIALLY_FAILED
    if running > 0:
        return BatchStatus.RUNNING
    return current or BatchStatus.PENDING


def normalize_counts(counts: dict[str, int] | None) -> dict[str, int]:
    """Fill in any missing per-status keys with 0."""
    return {**ZERO_COUNTS, **(counts or {})}


# --- Cursor encoding -------------------------------------------------------


def encode_item_cursor(*, updated_at: datetime, item_id: str) -> str:
    """Polling cursor for `(updated_at, item_id)` ascending pagination."""
    raw = f"{updated_at.isoformat()}|{item_id}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_item_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    if not cursor:
        return None
    pad = "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode((cursor + pad).encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid cursor: {cursor!r}") from e
    if "|" not in raw:
        raise ValueError(f"Invalid cursor: {cursor!r}")
    iso, item_id = raw.split("|", 1)
    try:
        when = datetime.fromisoformat(iso)
    except ValueError as e:
        raise ValueError(f"Invalid cursor timestamp: {iso!r}") from e
    return when, item_id


def encode_batch_cursor(*, created_at: datetime, batch_id: str) -> str:
    raw = f"{created_at.isoformat()}|{batch_id}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_batch_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    if not cursor:
        return None
    pad = "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode((cursor + pad).encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid cursor: {cursor!r}") from e
    if "|" not in raw:
        raise ValueError(f"Invalid cursor: {cursor!r}")
    iso, batch_id = raw.split("|", 1)
    try:
        when = datetime.fromisoformat(iso)
    except ValueError as e:
        raise ValueError(f"Invalid cursor timestamp: {iso!r}") from e
    return when, batch_id


# --- Error codes (public contract) -----------------------------------------
#
# These strings show up in `error.code` on item poll responses. Keep them
# stable; downstream customer integrations may match on them.

class ItemErrorCode:
    PAYMENT_REQUIRED = "payment_required"           # over quota / no credits
    BILLING_UNAVAILABLE = "billing_unavailable"      # transient billing-svc error
    EXTRACTION_FAILED = "extraction_failed"          # generic extraction error
    UNSUPPORTED_INPUT = "unsupported_input"
    DOCUMENT_TOO_LARGE = "document_too_large"
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    OCR_PROVIDER_ERROR = "ocr_provider_error"
    UPLOAD_MISSING = "upload_missing"                # file row exists but S3 has no object
    URL_FETCH_FAILED = "url_fetch_failed"            # url source: bad/forbidden/expired/blocked
    INTERNAL_ERROR = "internal_error"

    # Retryable codes (worker may schedule a retry rather than fail terminally).
    RETRYABLE = frozenset(
        {
            BILLING_UNAVAILABLE,
            OCR_PROVIDER_ERROR,
            INTERNAL_ERROR,
        }
    )


def is_retryable(error_code: str | None) -> bool:
    return error_code in ItemErrorCode.RETRYABLE


# --- Resource projection ---------------------------------------------------
#
# Flat dict-shaped projections used by the API to render JSON for clients.
# Kept here so the same renderer is used by the route handlers and the
# polling endpoint without rebuilding shapes in two places.


def project_batch(batch_row, *, items: Iterable | None = None) -> dict:
    return {
        "object": "batch",
        "id": batch_row.id,
        "status": batch_row.status,
        "counts": batch_row.counts,
        "total_items": batch_row.total_items,
        "options": {
            "extract_text": batch_row.extract_text,
            "extract_images": batch_row.extract_images,
            "table_output_format": batch_row.table_output_format,
            "chunking": getattr(batch_row, "chunking", "none"),
            "chunk_size": getattr(batch_row, "chunk_size", 1000),
        },
        "metadata": batch_row.metadata,
        "created_at": batch_row.created_at.isoformat(),
        "started_at": batch_row.started_at.isoformat() if batch_row.started_at else None,
        "completed_at": batch_row.completed_at.isoformat() if batch_row.completed_at else None,
        "expires_at": batch_row.expires_at.isoformat(),
    }


def redact_url(url: str | None) -> str | None:
    """Strip query + fragment before a url-sourced item is echoed back.

    A presigned URL's query string IS its bearer credential (S3/GCS/Azure
    signatures live there); the full url is stored in the DB for the worker
    to fetch/retry with, but anyone with read access to a batch's items
    should only ever see scheme+host+path, never the live signature."""
    if not url:
        return url
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def project_item(item_row) -> dict:
    payload: dict = {
        "object": "batch_item",
        "id": item_row.id,
        "batch_id": item_row.batch_id,
        "file_id": item_row.file_id,
        "url": redact_url(getattr(item_row, "url", None)),
        "position": item_row.position,
        "status": item_row.status,
        "page_count": item_row.page_count,
        "attempts": item_row.attempts,
        "started_at": item_row.started_at.isoformat() if item_row.started_at else None,
        "completed_at": item_row.completed_at.isoformat() if item_row.completed_at else None,
        "updated_at": item_row.updated_at.isoformat(),
    }
    if item_row.status == ItemStatus.SUCCEEDED:
        payload["result_url"] = (
            f"/v1/batches/{item_row.batch_id}/items/{item_row.id}/result"
        )
    if item_row.error_code:
        payload["error"] = {
            "code": item_row.error_code,
            "message": item_row.error_message,
        }
    return payload


__all__ = [
    "ItemErrorCode",
    "decode_batch_cursor",
    "decode_item_cursor",
    "derive_batch_status",
    "encode_batch_cursor",
    "encode_item_cursor",
    "is_retryable",
    "normalize_counts",
    "project_batch",
    "project_item",
    "redact_url",
]
