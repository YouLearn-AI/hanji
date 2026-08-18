"""POST /v1/batches and friends — async batch lifecycle endpoints.

Five routes:

- ``POST /v1/batches`` — create a batch from either previously uploaded
  ``file_id``s (``source.type: "files"``) or raw ``url``s (``source.type:
  "urls"``). URL items skip the upload step: we fetch each URL when that
  item starts processing; source bytes are never written to our storage,
  only the extraction result is. ``Idempotency-Key`` request header
  dedupes within the batch's 3-day window.
- ``GET /v1/batches/{id}`` — poll. Returns the batch with paginated items
  via ``?cursor=...&limit=...``. Cursor is opaque ``(updated_at, item_id)``.
- ``GET /v1/batches/{id}/items/{item_id}/result`` — 302 redirect to a
  presigned S3 GET URL pointing at the result blob.
- ``POST /v1/batches/{id}/cancel`` — flips remaining ``pending`` items
  to ``cancelled``. Running items finish on their own.
- ``GET /v1/batches`` — list the customer's recent batches with cursor
  pagination on ``(created_at, batch_id)``.

Like the sync routes, there is no auth here — front the server with your
own gateway if you expose it publicly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from extract.api.errors import FlatAPIError
from extract.api.routes._helpers import (
    RequestContext,
    api_signer_dep,
    batch_repo_dep,
    batches_prevalidate_dep,
)
from extract.clients.upload_signer import UploadSigner
from extract.config import settings
from extract.core.batch import (
    BatchStatus,
    ItemStatus,
    decode_batch_cursor,
    decode_item_cursor,
    encode_batch_cursor,
    encode_item_cursor,
    project_batch,
    project_item,
)
from extract.core.webhooks import (
    WebhookMode,
    normalize_mode,
    validate_metadata_size,
    validate_webhook_url,
)
from extract.repos.batches import (
    BatchItemSpec,
    BatchRepo,
    BatchRow,
    FileRow,
    FileStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/batches", tags=["v1", "async-batch"])

# Module-level dependency singletons — keeps B008 quiet and mirrors the
# pattern in `extract.api.routes.v1`.
_batches_prevalidate_dependency = Depends(batches_prevalidate_dep)
_batch_repo_dependency = Depends(batch_repo_dep)
_api_signer_dependency = Depends(api_signer_dep)
_idempotency_key_header = Header(default=None, alias="Idempotency-Key")
_after_cursor = Query(default=None)
_status_query = Query(default=None)
_get_batch_limit = Query(default=100, ge=1, le=500)
_list_batches_limit = Query(default=50, ge=1, le=200)

# --- Pydantic ---------------------------------------------------------------


class FilesSource(BaseModel):
    type: Literal["files"] = "files"
    file_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Uploaded file ids (from POST /v1/files) to process, 1-10,000 per batch.",
    )


class UrlSource(BaseModel):
    """Public or presigned URLs — no upload step. Source bytes are never
    written to our storage; only the extraction result is. Presigned URLs
    work the same as public ones — pass the full URL through unmodified,
    including its signature query string."""

    type: Literal["urls"] = "urls"
    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Public or presigned HTTP(S) URLs to fetch and process, 1-100 per "
            "batch. Each URL becomes one item; we fetch it when that item "
            "starts processing, not when you submit the batch — a slow or "
            "large download doesn't block this request."
        ),
    )

    @field_validator("urls")
    @classmethod
    def _validate_urls(cls, urls: list[str]) -> list[str]:
        for url in urls:
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ValueError(f"Not a valid http(s) URL: {url!r}")
        return urls


BatchSource = Annotated[FilesSource | UrlSource, Field(discriminator="type")]


class WebhookConfig(BaseModel):
    """Per-batch webhook opt-in. `registered` is an accepted alias of
    `svix`; only `svix` is documented."""

    mode: Literal["svix", "registered", "direct", "disabled"] = Field(
        ...,
        description=(
            '"svix": signed delivery to your registered endpoints. "direct": '
            'unsigned delivery to `url` (prototyping). "disabled": suppress '
            "for this batch (same as omitting `webhook`)."
        ),
    )
    url: str | None = Field(
        default=None,
        max_length=2048,
        description='Destination URL. Required (HTTPS, public hostname) iff mode is "direct".',
    )


class CreateBatchRequest(BaseModel):
    source: BatchSource = Field(
        ...,
        description=(
            'The documents to process: {"type": "files", "file_ids": [...]} for '
            'previously-uploaded files, or {"type": "urls", "urls": [...]} to '
            "have us fetch each URL directly (no upload step; source bytes "
            "never written to our storage)."
        ),
    )
    extract_text: bool = Field(
        default=True,
        description="Include text chunks in each item's result. Set false to skip text spans in every item.",
    )
    extract_images: bool = Field(
        default=True,
        description="Include figure (image) chunks in each item's result. Set false to skip figure extraction in every item.",
    )
    table_output_format: Literal["markdown", "html"] = Field(
        default="markdown",
        description=(
            "How table chunks are structured in each item's result: "
            "GitHub-Flavored-Markdown cells (default) or an HTML table."
        ),
    )
    # Opt-in RAG chunking, replayed on every item in the batch.
    chunking: Literal["none", "semantic"] = Field(
        default="none",
        description=(
            "'none' (default): item results unchanged. 'semantic': each item's "
            "result additionally carries `segments` — elements grouped toward "
            "chunk_size characters at semantic/structural boundaries."
        ),
    )
    chunk_size: int = Field(
        default=1000,
        description=(
            "Target segment size in characters. Segments land in a +/-25% band "
            "around this target. Only meaningful when chunking is enabled; "
            "validated (200-8000) only in that case and ignored otherwise."
        ),
    )

    @model_validator(mode="after")
    def _validate_chunking(self) -> CreateBatchRequest:
        # Mirrors ExtractRequest's enable-gated rule (back-compat).
        if self.chunking != "none":
            if not (200 <= self.chunk_size <= 8000):
                raise ValueError(
                    "chunk_size must be between 200 and 8000 characters when chunking is enabled"
                )
        elif not (200 <= self.chunk_size <= 8000):
            self.chunk_size = 1000
        return self

    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Arbitrary JSON stored with the batch and echoed back on every "
            "poll, in the batch list, and in webhook event bodies. Put your "
            "own run ids, tenant ids, or job references here. It never "
            "affects processing. Maximum 16 KB serialized."
        ),
    )

    webhook: WebhookConfig | None = Field(
        default=None,
        description=(
            "Optional completion webhook for this batch. Omitted means no "
            "webhook fires. `{\"mode\": \"svix\"}` delivers a signed "
            "`batch.update` event to your registered endpoints when the batch "
            "reaches a terminal status; `{\"mode\": \"direct\", \"url\": ...}` "
            "sends an unsigned event to the given HTTPS URL (prototyping)."
        ),
    )

    @model_validator(mode="after")
    def _validate_webhook(self) -> CreateBatchRequest:
        metadata_error = validate_metadata_size(self.metadata)
        if metadata_error:
            raise ValueError(metadata_error)
        if self.webhook is None:
            return self
        mode = normalize_mode(self.webhook.mode)
        if mode == WebhookMode.DIRECT:
            if not self.webhook.url:
                raise ValueError('webhook.url is required when webhook.mode is "direct"')
            # The self-host escape hatch lifts the public-HTTPS requirement
            # (the delivery transport honors the same flag).
            if not settings.EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS:
                url_error = validate_webhook_url(self.webhook.url)
                if url_error:
                    raise ValueError(f"webhook.url: {url_error}")
        elif self.webhook.url:
            raise ValueError('webhook.url is only accepted when webhook.mode is "direct"')
        return self


class BatchCounts(BaseModel):
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0


class BatchUpdateEvent(BaseModel):
    """The webhook body delivered when a batch reaches a terminal status.
    Documentation-only model — registered on ``app.webhooks`` so it renders
    in the API reference; nothing calls this as an endpoint. The live
    serializer is ``extract.core.webhooks.build_batch_update_body``."""

    type: Literal["batch.update"] = Field(
        "batch.update", description="Event type. Only `batch.update` exists today."
    )
    batch_id: str = Field(..., description="The batch that reached a terminal status.")
    status: Literal["completed", "partially_failed", "failed", "cancelled"] = Field(
        ...,
        description=(
            "Terminal batch status. Treat `completed` AND `partially_failed` as "
            "'results are ready' — `partially_failed` means some items succeeded "
            "and some failed (check `counts`), not a total failure."
        ),
    )
    counts: BatchCounts = Field(..., description="Per-item-status counts at completion.")
    total_items: int = Field(..., description="Number of items in the batch.")
    metadata: dict[str, Any] | None = Field(
        None, description="The `metadata` you passed on `POST /v1/batches`, echoed verbatim."
    )
    completed_at: str | None = Field(
        None, description="ISO-8601 UTC timestamp when the batch reached its terminal status."
    )
    results_expires_at: str | None = Field(
        None,
        description=(
            "ISO-8601 UTC timestamp after which this batch's results are deleted "
            "(3-day default retention). Fetch results before then; a resend after "
            "this point is refused."
        ),
    )


class BatchResource(BaseModel):
    object: str = "batch"
    id: str
    status: str
    counts: BatchCounts
    total_items: int
    options: dict[str, Any]
    metadata: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    expires_at: str


class BatchItemResource(BaseModel):
    object: str = "batch_item"
    id: str
    batch_id: str
    file_id: str | None = None
    url: str | None = Field(
        default=None,
        description=(
            "For URL-sourced items only (null for uploaded files). Query "
            "string stripped so a presigned signature never appears in the "
            "response. items[].error.message gets the same treatment for "
            "url_fetch_failed items."
        ),
    )
    position: int
    status: str
    page_count: int | None = None
    attempts: int
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str
    result_url: str | None = None
    error: dict[str, Any] | None = None


class BatchListResponse(BaseModel):
    object: str = "list"
    data: list[BatchResource]
    next_cursor: str | None = None


class BatchPollResponse(BaseModel):
    object: str = "batch"
    id: str
    status: str
    counts: BatchCounts
    total_items: int
    options: dict[str, Any]
    metadata: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    expires_at: str
    items: list[BatchItemResource]
    next_cursor: str | None = None


# --- Polling rate limit ------------------------------------------------------
#
# 200 req/s on the status-polling GETs, enforced in-process per API replica
# (abuse protection against runaway polling loops). Result-redirect fetches
# are deliberately exempt: fetching results isn't a polling loop.


class _PollRateLimiter:
    """Per-second fixed-window counter, one bucket per org."""

    def __init__(self) -> None:
        self._window: int = 0
        self._counts: dict[str, int] = {}

    def allow(self, customer_id: str, *, limit: int) -> bool:
        now = int(time.time())
        if now != self._window:
            self._window = now
            self._counts = {}
        count = self._counts.get(customer_id, 0) + 1
        self._counts[customer_id] = count
        return count <= limit


_poll_rate_limiter = _PollRateLimiter()

_POLL_RATE_LIMIT_DETAIL = (
    "Polling rate limit exceeded. Use webhooks for completion notification "
    "instead of polling: pass webhook: {\"mode\": \"svix\"} on POST /v1/batches "
    "and register an endpoint with `python -m extract.webhooks add`."
)


def _enforce_poll_rate_limit(customer_id: str) -> None:
    limit = settings.EXTRACT_BATCH_POLL_RATE_LIMIT_PER_SECOND
    if limit <= 0:
        return
    if not _poll_rate_limiter.allow(customer_id, limit=limit):
        raise HTTPException(
            status_code=429,
            detail=_POLL_RATE_LIMIT_DETAIL,
            headers={"Retry-After": "1"},
        )


# --- Helpers ----------------------------------------------------------------


def _resource_from_batch(row: BatchRow) -> BatchResource:
    payload = project_batch(row)
    return BatchResource(
        id=payload["id"],
        status=payload["status"],
        counts=BatchCounts(**payload["counts"]),
        total_items=payload["total_items"],
        options=payload["options"],
        metadata=payload["metadata"],
        created_at=payload["created_at"],
        started_at=payload["started_at"],
        completed_at=payload["completed_at"],
        expires_at=payload["expires_at"],
    )


async def _head_and_mark_uploaded(
    repo: BatchRepo,
    signer: UploadSigner,
    customer_id: str,
    row: FileRow,
) -> FileRow | None:
    """Reconcile one stale ``pending_upload`` row against S3.

    The presigned PUT bypasses our API, so a finished upload can leave the DB
    row reading ``pending_upload``. HEAD the object and, if present, flip the
    row to ``uploaded``. Returns the updated row, or ``None`` when the object
    isn't in S3 (``head_upload`` already swallows missing/expired/denied/
    transient as ``None``), leaving the caller to treat the file as not yet
    uploaded. Mirrors the best-effort reconciliation in ``GET /v1/files/{id}``.
    """
    head = await signer.head_upload(key=row.s3_key)
    if head is None:
        return None
    updated = await repo.mark_file_uploaded(
        file_id=row.id,
        customer_id=customer_id,
        sha256=head.get("ETag", "").strip('"') or None,
    )
    if updated is not None:
        return updated
    # A concurrent request flipped the row between our HEAD and UPDATE
    # (``mark_file_uploaded`` only matches ``pending_upload``). The object
    # exists, so re-read the now-``uploaded`` row instead of falsely 409ing.
    return await repo.get_file(file_id=row.id, customer_id=customer_id)


# --- Endpoints --------------------------------------------------------------


@router.post(
    "",
    response_model=BatchResource,
    operation_id="create_batch_v1",
    summary="Create a batch",
    description=(
        "Submit uploaded files as one batch job. Returns immediately with "
        '`status: "pending"`; poll `GET /v1/batches/{batch_id}` for per-item progress. '
        "Pass an `Idempotency-Key` header to make retries safe: the same key within "
        "3 days returns the same batch instead of creating a duplicate."
    ),
    responses={
        404: {
            "description": "One or more `file_id`s don't exist for this account (or their 3-day TTL lapsed).",
            "content": {
                "application/json": {
                    "example": {"error": "file_not_found", "file_ids": ["file_abc123"]}
                }
            },
        },
        409: {
            "description": (
                "One or more files have no bytes in storage: the upload never completed "
                "or expired. A *finished* upload does not 409."
            ),
            "content": {
                "application/json": {
                    "example": {"error": "file_not_uploaded", "file_ids": ["file_abc123"]}
                }
            },
        },
    },
)
async def create_batch_endpoint(
    body: CreateBatchRequest,
    ctx: RequestContext = _batches_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
    signer: UploadSigner = _api_signer_dependency,
    idempotency_key: str | None = _idempotency_key_header,
) -> BatchResource:
    if not repo.configured:
        raise HTTPException(
            status_code=503,
            detail="Async batch DB unavailable (DATABASE_URL unset).",
        )

    # Idempotency dedup. We check before any work so the caller can safely
    # retry POSTs without double-creating.
    if idempotency_key:
        existing = await repo.find_batch_by_idempotency(
            customer_id=ctx.customer_id, idempotency_key=idempotency_key
        )
        if existing is not None:
            return _resource_from_batch(existing)

    items: list[BatchItemSpec]
    if isinstance(body.source, UrlSource):
        # Fetched when the item starts processing (batch_worker.py) — no
        # upload/resolution step; source bytes are never written to storage.
        urls = list(dict.fromkeys(body.source.urls))  # dedupe, preserve order
        items = [BatchItemSpec(url=u) for u in urls]
    else:
        # Resolve all file_ids in one shot. Files must (a) belong to this
        # customer and (b) be `uploaded`. Anything else is rejected.
        file_ids = list(dict.fromkeys(body.source.file_ids))  # dedupe, preserve order
        rows: dict[str, FileRow] = {}
        missing: list[str] = []
        for fid in file_ids:
            row = await repo.get_file(file_id=fid, customer_id=ctx.customer_id)
            if row is None:
                missing.append(fid)
            else:
                rows[fid] = row

        # Self-heal stale `pending_upload` rows. The presigned PUT goes straight to
        # S3 and never touches our API, so a fully-uploaded file can still read
        # `pending_upload` here — the documented flow (POST /v1/files → PUT →
        # POST /v1/batches) hits this every time. HEAD each pending file and flip
        # the ones actually present in S3, mirroring GET /v1/files/{id}. Files that
        # are correctly `uploaded` already (e.g. the client polled GET first) cost
        # nothing here; only genuinely-absent files stay pending and 409 below.
        pending = [fid for fid, row in rows.items() if row.status != FileStatus.UPLOADED]
        if pending and signer.configured_for_uploads:
            healed = await asyncio.gather(
                *(
                    _head_and_mark_uploaded(repo, signer, ctx.customer_id, rows[fid])
                    for fid in pending
                )
            )
            for fid, updated in zip(pending, healed, strict=True):
                if updated is not None:
                    rows[fid] = updated

        not_uploaded = [fid for fid, row in rows.items() if row.status != FileStatus.UPLOADED]

        if missing:
            raise FlatAPIError(404, {"error": "file_not_found", "file_ids": missing})
        if not_uploaded:
            raise FlatAPIError(409, {"error": "file_not_uploaded", "file_ids": not_uploaded})

        files = [rows[fid] for fid in file_ids if fid in rows]
        if not files:
            raise HTTPException(status_code=400, detail="source.file_ids must be non-empty")
        items = [BatchItemSpec(file_id=f.id) for f in files]

    webhook_mode = normalize_mode(body.webhook.mode) if body.webhook else None
    webhook_url = body.webhook.url if body.webhook and webhook_mode == WebhookMode.DIRECT else None
    batch_row, _ = await repo.create_batch(
        customer_id=ctx.customer_id,
        phi_safe=ctx.phi_safe,
        idempotency_key=idempotency_key,
        engine=None,
        extract_text=body.extract_text,
        extract_images=body.extract_images,
        ocr="auto",
        table_output_format=body.table_output_format,
        chunking=body.chunking,
        chunk_size=body.chunk_size,
        metadata=body.metadata,
        items=items,
        webhook_mode=webhook_mode,
        webhook_url=webhook_url,
    )
    logger.info(
        "batch.created batch_id=%s total_items=%d webhook_mode=%s",
        batch_row.id,
        batch_row.total_items,
        webhook_mode,
    )
    return _resource_from_batch(batch_row)


@router.get(
    "/{batch_id}",
    response_model=BatchPollResponse,
    operation_id="get_batch_v1",
    summary="Poll a batch",
    description=(
        "Batch status plus paginated per-item statuses, ordered by last update. "
        "Pass back `next_cursor` as `cursor` to fetch only the items that changed "
        "since your previous poll."
    ),
)
async def get_batch_endpoint(
    batch_id: str,
    ctx: RequestContext = _batches_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
    cursor: str | None = _after_cursor,
    limit: int = _get_batch_limit,
) -> BatchPollResponse:
    _enforce_poll_rate_limit(ctx.customer_id)
    batch = await repo.get_batch(batch_id=batch_id, customer_id=ctx.customer_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    try:
        decoded = decode_item_cursor(cursor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cursor_updated_at = decoded[0] if decoded else None
    cursor_id = decoded[1] if decoded else None

    items = await repo.list_batch_items_after(
        batch_id=batch_id,
        customer_id=ctx.customer_id,
        limit=limit + 1,  # peek one extra to know if more exist
        cursor_updated_at=cursor_updated_at,
        cursor_id=cursor_id,
    )
    has_more = len(items) > limit
    items = items[:limit]
    next_cursor = (
        encode_item_cursor(updated_at=items[-1].updated_at, item_id=items[-1].id)
        if has_more and items
        else None
    )

    base = project_batch(batch)
    return BatchPollResponse(
        id=base["id"],
        status=base["status"],
        counts=BatchCounts(**base["counts"]),
        total_items=base["total_items"],
        options=base["options"],
        metadata=base["metadata"],
        created_at=base["created_at"],
        started_at=base["started_at"],
        completed_at=base["completed_at"],
        expires_at=base["expires_at"],
        items=[BatchItemResource(**project_item(i)) for i in items],
        next_cursor=next_cursor,
    )


@router.get(
    "/{batch_id}/items/{item_id}/result",
    operation_id="get_batch_item_result_v1",
    summary="Download an item result",
    description=(
        "Redirects (`302`) to a presigned S3 GET for the item's result JSON, so follow "
        "redirects. Results expire 3 days after item completion."
    ),
    responses={
        302: {"description": "Redirect (`Location`) to a presigned S3 GET for the result blob."},
        404: {"description": "Batch or item not found for this account."},
        409: {
            "description": "Item hasn't reached `succeeded` yet; keep polling `GET /v1/batches/{batch_id}`.",
            "content": {
                "application/json": {"example": {"error": "result_not_ready", "status": "running"}}
            },
        },
        410: {"description": "Result is gone; the 3-day retention window has passed."},
    },
)
async def get_batch_item_result_endpoint(
    batch_id: str,
    item_id: str,
    ctx: RequestContext = _batches_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
    signer: UploadSigner = _api_signer_dependency,
) -> Response:
    item = await repo.get_batch_item(
        batch_id=batch_id, item_id=item_id, customer_id=ctx.customer_id
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Batch item not found")
    if item.status != ItemStatus.SUCCEEDED:
        raise FlatAPIError(409, {"error": "result_not_ready", "status": item.status})
    if not item.result_s3_bucket or not item.result_s3_key:
        raise HTTPException(status_code=410, detail="Result no longer available")

    presigned = await signer.presign_download(
        bucket=item.result_s3_bucket,
        key=item.result_s3_key,
    )
    return Response(
        status_code=302,
        headers={"Location": presigned.url},
    )


@router.post(
    "/{batch_id}/cancel",
    response_model=BatchResource,
    operation_id="cancel_batch_v1",
    summary="Cancel a batch",
    description=(
        "Flip the batch's remaining `pending` items to `cancelled`. Items already "
        "`running` finish on their own. Cancelled items are not billed."
    ),
)
async def cancel_batch_endpoint(
    batch_id: str,
    ctx: RequestContext = _batches_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
) -> BatchResource:
    batch = await repo.cancel_batch(batch_id=batch_id, customer_id=ctx.customer_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return _resource_from_batch(batch)


@router.get(
    "",
    response_model=BatchListResponse,
    operation_id="list_batches_v1",
    summary="List your batches",
    description="Your account's batches, newest first, with cursor pagination and an optional `status` filter.",
)
async def list_batches_endpoint(
    ctx: RequestContext = _batches_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
    status: str | None = _status_query,
    cursor: str | None = _after_cursor,
    limit: int = _list_batches_limit,
) -> BatchListResponse:
    _enforce_poll_rate_limit(ctx.customer_id)
    if status is not None and status not in (
        BatchStatus.PENDING,
        BatchStatus.RUNNING,
        BatchStatus.COMPLETED,
        BatchStatus.PARTIALLY_FAILED,
        BatchStatus.FAILED,
        BatchStatus.CANCELLED,
        BatchStatus.EXPIRED,
    ):
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    try:
        decoded = decode_batch_cursor(cursor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    cursor_created_at = decoded[0] if decoded else None
    cursor_id = decoded[1] if decoded else None
    rows = await repo.list_batches(
        customer_id=ctx.customer_id,
        status=status,
        limit=limit + 1,
        cursor_created_at=cursor_created_at,
        cursor_id=cursor_id,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = (
        encode_batch_cursor(created_at=rows[-1].created_at, batch_id=rows[-1].id)
        if has_more and rows
        else None
    )
    return BatchListResponse(
        data=[_resource_from_batch(r) for r in rows],
        next_cursor=next_cursor,
    )
