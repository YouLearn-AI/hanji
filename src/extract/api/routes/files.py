"""POST /v1/files and GET /v1/files/{file_id}.

The upload pattern: customers POST metadata, get back a presigned PUT URL
plus a ``file_id``, then PUT the bytes directly to S3. Our API never sees
the upload payload — bandwidth stays between the customer and S3.

The customer can then reference any number of ``file_id``s in a
``POST /v1/batches`` request. Files have a 3-day TTL on the S3 side
(lifecycle rule), and a matching ``expires_at`` recorded in the DB row.

There is no ``DELETE /v1/files/{id}`` in v1 — the lifecycle rule is the
only cleanup mechanism. Re-add later if a customer requests immediate
purge semantics.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from extract.api.routes._helpers import (
    RequestContext,
    api_signer_dep,
    batch_repo_dep,
    files_prevalidate_dep,
)
from extract.clients.upload_signer import UploadSigner
from extract.config import settings
from extract.core.pdf import MAX_SIZE_BYTES
from extract.repos.batches import BatchRepo, FileRow, FileStatus, new_file_id

router = APIRouter(prefix="/v1/files", tags=["v1", "async-batch"])

# Module-level dependency singletons — keeps B008 quiet and mirrors the
# pattern used in `extract.api.routes.v1`.
_files_prevalidate_dependency = Depends(files_prevalidate_dep)
_batch_repo_dependency = Depends(batch_repo_dep)
_api_signer_dependency = Depends(api_signer_dep)

# Filenames are advisory but we cap their size to keep logs bounded.
_MAX_FILENAME_LEN = 512


def _max_upload_bytes() -> int:
    return settings.EXTRACT_UPLOAD_MAX_BYTES or MAX_SIZE_BYTES


class CreateFileRequest(BaseModel):
    filename: str | None = Field(default=None, max_length=_MAX_FILENAME_LEN)
    content_type: str | None = Field(default=None, max_length=128)
    size_bytes: int = Field(..., gt=0)


class UploadDescriptor(BaseModel):
    method: str
    url: str
    expires_at: str


class FileResource(BaseModel):
    object: str = "file"
    id: str
    customer_id: str
    status: str
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int
    sha256: str | None = None
    created_at: str
    expires_at: str


class CreateFileResponse(FileResource):
    upload: UploadDescriptor


def _file_resource(row: FileRow) -> FileResource:
    return FileResource(
        id=row.id,
        customer_id=row.customer_id,
        status=row.status,
        filename=row.filename,
        content_type=row.content_type,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        created_at=row.created_at.isoformat(),
        expires_at=row.expires_at.isoformat(),
    )


@router.post(
    "",
    response_model=CreateFileResponse,
    operation_id="create_file_v1",
    summary="Register a file for batch upload",
    description=(
        "Reserve a file slot and get back a `file_id` plus a presigned S3 PUT URL. "
        "PUT the file bytes to that URL (they go directly to S3, never through this API), "
        "then reference the `file_id` in `POST /v1/batches`. Uploads expire after 3 days."
    ),
    responses={
        413: {"description": "`size_bytes` exceeds the upload cap."},
        503: {"description": "Async batch uploads are not configured for this deployment."},
    },
)
async def create_file_endpoint(
    body: CreateFileRequest,
    ctx: RequestContext = _files_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
    signer: UploadSigner = _api_signer_dependency,
) -> Any:
    if not signer.configured_for_uploads:
        raise HTTPException(
            status_code=503,
            detail="Async batch uploads are not configured (EXTRACT_UPLOADS_BUCKET unset).",
        )
    cap = _max_upload_bytes()
    if body.size_bytes > cap:
        raise HTTPException(
            status_code=413,
            detail=f"size_bytes {body.size_bytes} exceeds maximum of {cap}.",
        )
    if not repo.configured:
        raise HTTPException(status_code=503, detail="Async batch DB unavailable")

    # Compute file_id + s3_key up front so the row insert and the presigned
    # PUT URL agree on exactly the same key. The 3-day lifecycle cleans up
    # orphaned rows + S3 objects if the customer abandons the PUT.
    file_id = new_file_id()
    s3_key = signer.upload_key_for(customer_id=ctx.customer_id, file_id=file_id)
    file_row = await repo.create_file(
        customer_id=ctx.customer_id,
        phi_safe=ctx.phi_safe,
        s3_bucket=signer.uploads_bucket or "",
        s3_key=s3_key,
        filename=body.filename,
        content_type=body.content_type,
        size_bytes=body.size_bytes,
        file_id=file_id,
    )
    upload = await signer.presign_upload(
        customer_id=ctx.customer_id,
        file_id=file_row.id,
        content_type=body.content_type,
        content_length=body.size_bytes,
    )

    return CreateFileResponse(
        **_file_resource(file_row).model_dump(),
        upload=UploadDescriptor(
            method=upload.method,
            url=upload.url,
            expires_at=upload.expires_at.isoformat(),
        ),
    )


@router.get(
    "/{file_id}",
    response_model=FileResource,
    operation_id="get_file_v1",
    summary="Check upload status",
    description=(
        "Fetch a file's metadata and upload `status`. Useful to confirm a presigned PUT "
        "landed before submitting a batch, though `POST /v1/batches` re-checks S3 on "
        "submit, so this call is optional."
    ),
)
async def get_file_endpoint(
    file_id: str,
    ctx: RequestContext = _files_prevalidate_dependency,
    repo: BatchRepo = _batch_repo_dependency,
    signer: UploadSigner = _api_signer_dependency,
) -> Any:
    row = await repo.get_file(file_id=file_id, customer_id=ctx.customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    # Best-effort: if status is still `pending_upload`, head the S3 object
    # to detect a finished but unannounced upload, and flip to `uploaded`.
    # This matters because POST /v1/batches verifies file status.
    if row.status == FileStatus.PENDING_UPLOAD and signer.configured_for_uploads:
        head = await signer.head_upload(key=row.s3_key)
        if head is not None:
            updated = await repo.mark_file_uploaded(
                file_id=row.id,
                customer_id=ctx.customer_id,
                sha256=head.get("ETag", "").strip('"') or None,
            )
            if updated is not None:
                row = updated
    return _file_resource(row)
