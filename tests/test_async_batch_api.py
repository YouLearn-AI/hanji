"""HTTP tests for /v1/files and /v1/batches.

In-process FastAPI app via httpx ASGITransport. The DB-backed
:class:`BatchRepo` is replaced with an in-memory fake; the boto3
:class:`UploadSigner` is replaced with a stub that just returns
predictable strings. Same pattern as :mod:`test_extract_file_endpoint`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from extract.api.errors import FlatAPIError, flat_api_error_handler
from extract.api.routes import _helpers as helpers
from extract.api.routes._helpers import RequestContext
from extract.api.routes.batches import router as batches_router
from extract.api.routes.files import router as files_router
from extract.core.batch import BatchStatus, ItemStatus, derive_batch_status
from extract.repos.batches import (
    BatchItemRow,
    BatchRow,
    FileRow,
    FileStatus,
    new_batch_id,
    new_file_id,
    new_item_id,
)

# --- Fakes ------------------------------------------------------------------


class _FakeBatchRepo:
    """In-memory BatchRepo. Mirrors the public method surface used by the
    routes; intentionally omits the worker-only ``claim_next_item`` etc."""

    def __init__(self) -> None:
        self.files: dict[str, FileRow] = {}
        self.batches: dict[str, BatchRow] = {}
        self.items: dict[str, list[BatchItemRow]] = {}  # batch_id -> items

    @property
    def configured(self) -> bool:
        return True

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
        ttl_seconds: int = 24 * 3600,
    ) -> FileRow:
        fid = file_id or new_file_id()
        now = datetime.now(tz=UTC)
        row = FileRow(
            id=fid,
            customer_id=customer_id,
            phi_safe=phi_safe,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=None,
            status=FileStatus.PENDING_UPLOAD,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self.files[fid] = row
        return row

    async def mark_file_uploaded(
        self, *, file_id: str, customer_id: str, sha256: str | None = None
    ) -> FileRow | None:
        row = self.files.get(file_id)
        if row is None or row.customer_id != customer_id:
            return None
        if row.status != FileStatus.PENDING_UPLOAD:
            return None
        new = FileRow(
            **{**row.__dict__, "status": FileStatus.UPLOADED, "sha256": sha256 or row.sha256}
        )
        self.files[file_id] = new
        return new

    async def get_file(self, *, file_id: str, customer_id: str) -> FileRow | None:
        row = self.files.get(file_id)
        if row is None or row.customer_id != customer_id:
            return None
        return row

    async def find_batch_by_idempotency(
        self, *, customer_id: str, idempotency_key: str
    ) -> BatchRow | None:
        for b in self.batches.values():
            if b.customer_id == customer_id and b.idempotency_key == idempotency_key:
                return b
        return None

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
        metadata: dict | None,
        items,
        table_output_format: str = "markdown",
        chunking: str = "none",
        chunk_size: int = 1000,
        webhook_mode: str | None = None,
        webhook_url: str | None = None,
        ttl_seconds: int = 24 * 3600,
    ):
        self.last_webhook = (webhook_mode, webhook_url)
        bid = new_batch_id()
        now = datetime.now(tz=UTC)
        counts = {"pending": len(items), "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0}
        batch = BatchRow(
            id=bid,
            customer_id=customer_id,
            phi_safe=phi_safe,
            status=BatchStatus.PENDING,
            idempotency_key=idempotency_key,
            total_items=len(items),
            counts=counts,
            metadata=metadata,
            engine=engine,
            extract_text=extract_text,
            extract_images=extract_images,
            ocr=ocr,
            table_output_format=table_output_format,
            chunking=chunking,
            chunk_size=chunk_size,
            created_at=now,
            started_at=None,
            completed_at=None,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self.batches[bid] = batch
        item_rows: list[BatchItemRow] = []
        for pos, spec in enumerate(items):
            row = BatchItemRow(
                id=new_item_id(),
                batch_id=bid,
                customer_id=customer_id,
                file_id=spec.file_id,
                url=spec.url,
                position=pos,
                status=ItemStatus.PENDING,
                page_count=None,
                error_code=None,
                error_message=None,
                result_s3_bucket=None,
                result_s3_key=None,
                attempts=0,
                started_at=None,
                completed_at=None,
                updated_at=now,
            )
            item_rows.append(row)
        self.items[bid] = item_rows
        return batch, item_rows

    async def get_batch(self, *, batch_id: str, customer_id: str) -> BatchRow | None:
        b = self.batches.get(batch_id)
        if b is None or b.customer_id != customer_id:
            return None
        return b

    async def list_batches(
        self,
        *,
        customer_id: str,
        status: str | None = None,
        limit: int = 50,
        cursor_created_at: datetime | None = None,
        cursor_id: str | None = None,
    ) -> list[BatchRow]:
        rows = sorted(
            (b for b in self.batches.values() if b.customer_id == customer_id),
            key=lambda r: (r.created_at, r.id),
            reverse=True,
        )
        if status:
            rows = [r for r in rows if r.status == status]
        if cursor_created_at is not None:
            rows = [r for r in rows if (r.created_at, r.id) < (cursor_created_at, cursor_id or "")]
        return rows[:limit]

    async def get_batch_item(
        self, *, batch_id: str, item_id: str, customer_id: str
    ) -> BatchItemRow | None:
        for r in self.items.get(batch_id, []):
            if r.id == item_id and r.customer_id == customer_id:
                return r
        return None

    async def list_batch_items_after(
        self,
        *,
        batch_id: str,
        customer_id: str,
        limit: int = 100,
        cursor_updated_at: datetime | None = None,
        cursor_id: str | None = None,
    ) -> list[BatchItemRow]:
        rows = sorted(
            (r for r in self.items.get(batch_id, []) if r.customer_id == customer_id),
            key=lambda r: (r.updated_at, r.id),
        )
        if cursor_updated_at is not None:
            rows = [
                r for r in rows if (r.updated_at, r.id) > (cursor_updated_at, cursor_id or "")
            ]
        return rows[:limit]

    async def cancel_batch(
        self, *, batch_id: str, customer_id: str
    ) -> BatchRow | None:
        batch = self.batches.get(batch_id)
        if batch is None or batch.customer_id != customer_id:
            return None
        if batch.status in BatchStatus.TERMINAL:
            return batch
        # Flip remaining pending items
        new_items: list[BatchItemRow] = []
        cancelled_n = 0
        for it in self.items.get(batch_id, []):
            if it.status == ItemStatus.PENDING:
                new_items.append(
                    BatchItemRow(
                        **{
                            **it.__dict__,
                            "status": ItemStatus.CANCELLED,
                            "completed_at": datetime.now(tz=UTC),
                            "updated_at": datetime.now(tz=UTC),
                        }
                    )
                )
                cancelled_n += 1
            else:
                new_items.append(it)
        self.items[batch_id] = new_items
        # Recompute counts
        counts = {
            "pending": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for it in new_items:
            counts[it.status] = counts.get(it.status, 0) + 1
        new_status = derive_batch_status(counts, current=batch.status)
        new_batch = BatchRow(**{**batch.__dict__, "status": new_status, "counts": counts})
        self.batches[batch_id] = new_batch
        return new_batch


class _FakeSigner:
    def __init__(self) -> None:
        self.uploads_bucket = "test-uploads"
        self.results_bucket = "test-results"
        self.signed: list[tuple[str, str]] = []

    @property
    def configured_for_uploads(self) -> bool:
        return True

    @property
    def configured_for_results(self) -> bool:
        return True

    def upload_key_for(self, *, customer_id: str, file_id: str) -> str:
        return f"{customer_id}/{file_id}"

    def result_key_for(self, *, batch_id: str, item_id: str) -> str:
        return f"{batch_id}/{item_id}.json"

    async def presign_upload(self, *, customer_id, file_id, content_type=None, content_length=None):
        from extract.clients.upload_signer import PresignedUpload

        key = self.upload_key_for(customer_id=customer_id, file_id=file_id)
        self.signed.append(("PUT", key))
        return PresignedUpload(
            method="PUT",
            url=f"https://test/{key}?sig=stub",
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=30),
            bucket=self.uploads_bucket,
            key=key,
        )

    async def presign_download(self, *, bucket: str, key: str, ttl_seconds=None):
        from extract.clients.upload_signer import PresignedDownload

        return PresignedDownload(
            url=f"https://test-results/{key}?sig=stub",
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=15),
        )

    async def head_upload(self, *, key: str):
        return None  # tests that need it set will override.


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_repo() -> _FakeBatchRepo:
    return _FakeBatchRepo()


@pytest.fixture
def fake_signer() -> _FakeSigner:
    return _FakeSigner()


def _ctx() -> RequestContext:
    return RequestContext(customer_id="cus_test")


def _build_app(
    *,
    repo: _FakeBatchRepo,
    signer: _FakeSigner,
    ctx_factory=_ctx,
) -> FastAPI:
    app = FastAPI(default_response_class=ORJSONResponse)
    app.add_exception_handler(FlatAPIError, flat_api_error_handler)
    app.include_router(files_router)
    app.include_router(batches_router)
    app.state.batch_repo = repo
    app.state.upload_signer = signer
    app.dependency_overrides[helpers.batch_repo_dep] = lambda: repo
    app.dependency_overrides[helpers.api_signer_dep] = lambda: signer
    app.dependency_overrides[helpers.files_prevalidate_dep] = ctx_factory
    app.dependency_overrides[helpers.batches_prevalidate_dep] = ctx_factory
    return app


@pytest.fixture
async def client(
    fake_repo: _FakeBatchRepo,
    fake_signer: _FakeSigner,
) -> AsyncIterator[httpx.AsyncClient]:
    app = _build_app(repo=fake_repo, signer=fake_signer)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Tests: /v1/files -------------------------------------------------------


async def test_create_file_returns_presigned_put(
    client: httpx.AsyncClient, fake_signer: _FakeSigner
):
    r = await client.post(
        "/v1/files",
        json={"filename": "doc.pdf", "size_bytes": 12345, "content_type": "application/pdf"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "file"
    assert body["status"] == FileStatus.PENDING_UPLOAD
    assert body["filename"] == "doc.pdf"
    assert body["upload"]["method"] == "PUT"
    assert body["upload"]["url"].startswith("https://test/")
    assert fake_signer.signed[0][0] == "PUT"


async def test_create_file_rejects_oversize(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/files", json={"filename": "huge.pdf", "size_bytes": 10**12}
    )
    assert r.status_code == 413


async def test_get_file_returns_404_when_missing(client: httpx.AsyncClient):
    r = await client.get("/v1/files/file_does_not_exist")
    assert r.status_code == 404


# --- Tests: /v1/batches -----------------------------------------------------


async def _upload_one(client: httpx.AsyncClient, repo: _FakeBatchRepo) -> str:
    r = await client.post(
        "/v1/files", json={"filename": "doc.pdf", "size_bytes": 1024}
    )
    assert r.status_code == 200, r.text
    fid = r.json()["id"]
    # Mark uploaded directly via the fake repo since we don't actually PUT to S3.
    await repo.mark_file_uploaded(file_id=fid, customer_id="cus_test")
    return fid


async def test_create_batch_happy_path(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    fids = [await _upload_one(client, fake_repo) for _ in range(3)]
    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": fids}, "ocr": "auto"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "batch"
    assert body["status"] == BatchStatus.PENDING
    assert body["total_items"] == 3


async def test_create_batch_with_urls_creates_items_without_upload(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    """UrlSource never touches /v1/files or the upload signer — the worker
    fetches each url directly (asserted at the worker level elsewhere)."""
    r = await client.post(
        "/v1/batches",
        json={
            "source": {
                "type": "urls",
                "urls": [
                    "https://example.com/a.pdf",
                    "https://bucket.s3.amazonaws.com/b.pdf?X-Amz-Signature=secret",
                ],
            }
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_items"] == 2
    items = fake_repo.items[body["id"]]
    assert {i.file_id for i in items} == {None}
    assert {i.url for i in items} == {
        "https://example.com/a.pdf",
        "https://bucket.s3.amazonaws.com/b.pdf?X-Amz-Signature=secret",
    }


async def test_create_batch_with_urls_dedupes_preserving_order(
    client: httpx.AsyncClient, fake_repo: _FakeBatchRepo
):
    r = await client.post(
        "/v1/batches",
        json={
            "source": {
                "type": "urls",
                "urls": ["https://example.com/a.pdf", "https://example.com/a.pdf"],
            }
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["total_items"] == 1


@pytest.mark.parametrize(
    "urls",
    [
        ["not-a-url"],
        ["ftp://example.com/a.pdf"],
        ["javascript:alert(1)"],
        [],
    ],
)
async def test_create_batch_urls_rejects_invalid_input(client: httpx.AsyncClient, urls):
    r = await client.post("/v1/batches", json={"source": {"type": "urls", "urls": urls}})
    assert r.status_code == 422, r.text


async def test_poll_batch_echoes_redacted_url_and_null_file_id(
    client: httpx.AsyncClient,
):
    signed_url = "https://bucket.s3.amazonaws.com/patients/report.pdf?X-Amz-Signature=secret"
    r = await client.post(
        "/v1/batches", json={"source": {"type": "urls", "urls": [signed_url]}}
    )
    assert r.status_code == 200, r.text
    batch_id = r.json()["id"]

    poll = await client.get(f"/v1/batches/{batch_id}")
    assert poll.status_code == 200, poll.text
    (item,) = poll.json()["items"]
    assert item["file_id"] is None
    assert item["url"] == "https://bucket.s3.amazonaws.com/patients/report.pdf"
    assert "X-Amz-Signature" not in item["url"]


async def test_create_batch_carries_chunking_options(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    # Plan 077: chunking options persist on the batch row (the worker replays
    # them per item) and echo on the resource.
    fids = [await _upload_one(client, fake_repo)]
    r = await client.post(
        "/v1/batches",
        json={
            "source": {"type": "files", "file_ids": fids},
            "chunking": "semantic",
            "chunk_size": 800,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["options"]["chunking"] == "semantic"
    assert body["options"]["chunk_size"] == 800
    row = fake_repo.batches[body["id"]]
    assert row.chunking == "semantic" and row.chunk_size == 800
    # poll echoes the same options
    poll = await client.get(f"/v1/batches/{body['id']}")
    assert poll.json()["options"]["chunking"] == "semantic"


async def test_create_batch_chunk_size_gated_validation(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    fids = [await _upload_one(client, fake_repo)]
    # enabled + out of range -> 422
    r = await client.post(
        "/v1/batches",
        json={
            "source": {"type": "files", "file_ids": fids},
            "chunking": "semantic",
            "chunk_size": 50,
        },
    )
    assert r.status_code == 422
    # disabled + stray out-of-range size -> accepted (back-compat), stored default
    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": fids}, "chunk_size": 50},
    )
    assert r.status_code == 200, r.text
    assert r.json()["options"]["chunking"] == "none"
    assert r.json()["options"]["chunk_size"] == 1000


async def test_create_batch_webhook_field(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    # Plan 083: opt-in webhook config on POST /v1/batches.
    fids = [await _upload_one(client, fake_repo)]
    # Omitted -> NULL mode (no webhook; Reducto-parity default).
    r = await client.post("/v1/batches", json={"source": {"type": "files", "file_ids": fids}})
    assert r.status_code == 200, r.text
    assert fake_repo.last_webhook == (None, None)
    # svix mode -> stored, no url.
    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": fids}, "webhook": {"mode": "svix"}},
    )
    assert r.status_code == 200, r.text
    assert fake_repo.last_webhook == ("svix", None)
    # `registered` alias normalizes to svix.
    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": fids}, "webhook": {"mode": "registered"}},
    )
    assert r.status_code == 200, r.text
    assert fake_repo.last_webhook == ("svix", None)
    # direct mode -> url stored.
    r = await client.post(
        "/v1/batches",
        json={
            "source": {"type": "files", "file_ids": fids},
            "webhook": {"mode": "direct", "url": "https://hooks.example.com/x"},
        },
    )
    assert r.status_code == 200, r.text
    assert fake_repo.last_webhook == ("direct", "https://hooks.example.com/x")


async def test_create_batch_webhook_validation(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    fids = [await _upload_one(client, fake_repo)]
    base = {"source": {"type": "files", "file_ids": fids}}
    # direct without url -> 422.
    r = await client.post("/v1/batches", json={**base, "webhook": {"mode": "direct"}})
    assert r.status_code == 422
    # direct with non-https url -> 422.
    r = await client.post(
        "/v1/batches",
        json={**base, "webhook": {"mode": "direct", "url": "http://hooks.example.com/x"}},
    )
    assert r.status_code == 422
    # direct with raw-IP url -> 422 (SSRF fail-fast; transport re-checks).
    r = await client.post(
        "/v1/batches",
        json={**base, "webhook": {"mode": "direct", "url": "https://169.254.170.2/x"}},
    )
    assert r.status_code == 422
    # url with a non-direct mode -> 422.
    r = await client.post(
        "/v1/batches",
        json={**base, "webhook": {"mode": "svix", "url": "https://hooks.example.com/x"}},
    )
    assert r.status_code == 422
    # metadata over the 16 KB cap -> 422.
    r = await client.post(
        "/v1/batches", json={**base, "metadata": {"k": "x" * (16 * 1024 + 1)}}
    )
    assert r.status_code == 422


async def test_batch_poll_rate_limit_429_steers_to_webhooks(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
    monkeypatch,
):
    from extract.api.routes import batches as batches_module
    from extract.config import settings as app_settings

    fids = [await _upload_one(client, fake_repo)]
    r = await client.post("/v1/batches", json={"source": {"type": "files", "file_ids": fids}})
    batch_id = r.json()["id"]
    monkeypatch.setattr(app_settings, "EXTRACT_BATCH_POLL_RATE_LIMIT_PER_SECOND", 3)
    # Fresh limiter window so the test is deterministic.
    batches_module._poll_rate_limiter._window = 0
    batches_module._poll_rate_limiter._counts = {}
    statuses = [(await client.get(f"/v1/batches/{batch_id}")).status_code for _ in range(5)]
    assert statuses.count(429) >= 1
    limited = await client.get(f"/v1/batches/{batch_id}")
    if limited.status_code == 429:
        assert "webhook" in limited.json()["detail"].lower()
        assert limited.headers.get("retry-after") == "1"
    # Result-redirect fetches are exempt by design — no limiter call there.


async def test_create_batch_rejects_unuploaded_file(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
):
    # Create file but don't mark uploaded.
    r = await client.post("/v1/files", json={"filename": "x.pdf", "size_bytes": 100})
    assert r.status_code == 200
    fid = r.json()["id"]
    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": [fid]}},
    )
    assert r.status_code == 409
    assert r.json()["error"] == "file_not_uploaded"


async def test_create_batch_self_heals_stale_pending_upload(
    client: httpx.AsyncClient,
    fake_repo: _FakeBatchRepo,
    fake_signer: _FakeSigner,
):
    # The presigned PUT bypasses our API, so a file whose bytes are already in
    # S3 can still read `pending_upload`. The documented flow (POST /v1/files →
    # PUT → POST /v1/batches, no GET in between) hits this every time. Batch
    # creation should HEAD S3, find the object, flip the row, and succeed —
    # not 409.
    r = await client.post("/v1/files", json={"filename": "doc.pdf", "size_bytes": 100})
    assert r.status_code == 200
    fid = r.json()["id"]
    assert fake_repo.files[fid].status == FileStatus.PENDING_UPLOAD

    async def _head(*, key: str):
        return {"ETag": '"deadbeef"'}

    fake_signer.head_upload = _head  # the object really is in S3

    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": [fid]}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["total_items"] == 1
    # Reconciled to `uploaded` as a side effect, with the S3 ETag as sha256.
    assert fake_repo.files[fid].status == FileStatus.UPLOADED
    assert fake_repo.files[fid].sha256 == "deadbeef"


async def test_create_batch_rejects_unknown_file(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/batches",
        json={"source": {"type": "files", "file_ids": ["file_does_not_exist"]}},
    )
    assert r.status_code == 404
    assert r.json()["error"] == "file_not_found"


async def test_create_batch_idempotency_key_dedupes(
    client: httpx.AsyncClient, fake_repo: _FakeBatchRepo
):
    fid = await _upload_one(client, fake_repo)
    payload = {"source": {"type": "files", "file_ids": [fid]}}
    r1 = await client.post(
        "/v1/batches", json=payload, headers={"Idempotency-Key": "key-1"}
    )
    assert r1.status_code == 200
    bid_first = r1.json()["id"]
    r2 = await client.post(
        "/v1/batches", json=payload, headers={"Idempotency-Key": "key-1"}
    )
    assert r2.status_code == 200
    assert r2.json()["id"] == bid_first


async def test_get_batch_paginates_items(
    client: httpx.AsyncClient, fake_repo: _FakeBatchRepo
):
    fids = [await _upload_one(client, fake_repo) for _ in range(5)]
    r = await client.post(
        "/v1/batches", json={"source": {"type": "files", "file_ids": fids}}
    )
    bid = r.json()["id"]
    r = await client.get(f"/v1/batches/{bid}", params={"limit": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total_items"] == 5
    assert len(body["items"]) == 2
    assert body["next_cursor"] is not None
    # Follow the cursor
    r2 = await client.get(
        f"/v1/batches/{bid}", params={"limit": 5, "cursor": body["next_cursor"]}
    )
    body2 = r2.json()
    assert len(body2["items"]) == 3


async def test_cancel_flips_pending_to_cancelled(
    client: httpx.AsyncClient, fake_repo: _FakeBatchRepo
):
    fids = [await _upload_one(client, fake_repo) for _ in range(2)]
    r = await client.post(
        "/v1/batches", json={"source": {"type": "files", "file_ids": fids}}
    )
    bid = r.json()["id"]
    r = await client.post(f"/v1/batches/{bid}/cancel")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == BatchStatus.CANCELLED
    assert body["counts"]["cancelled"] == 2


async def test_get_item_result_redirects_when_succeeded(
    client: httpx.AsyncClient, fake_repo: _FakeBatchRepo
):
    fid = await _upload_one(client, fake_repo)
    r = await client.post(
        "/v1/batches", json={"source": {"type": "files", "file_ids": [fid]}}
    )
    bid = r.json()["id"]
    item = fake_repo.items[bid][0]
    # Simulate worker completion in the fake.
    fake_repo.items[bid][0] = BatchItemRow(
        **{
            **item.__dict__,
            "status": ItemStatus.SUCCEEDED,
            "page_count": 4,
            "result_s3_bucket": "test-results",
            "result_s3_key": f"{bid}/{item.id}.json",
            "completed_at": datetime.now(tz=UTC),
            "updated_at": datetime.now(tz=UTC),
        }
    )
    r = await client.get(f"/v1/batches/{bid}/items/{item.id}/result")
    # FastAPI test client follows redirects by default; ASGITransport doesn't.
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://test-results/")


async def test_get_item_result_409_when_not_ready(
    client: httpx.AsyncClient, fake_repo: _FakeBatchRepo
):
    fid = await _upload_one(client, fake_repo)
    r = await client.post(
        "/v1/batches", json={"source": {"type": "files", "file_ids": [fid]}}
    )
    bid = r.json()["id"]
    item = fake_repo.items[bid][0]
    r = await client.get(f"/v1/batches/{bid}/items/{item.id}/result")
    assert r.status_code == 409
    assert r.json()["error"] == "result_not_ready"


