"""Worker-level tests for url-sourced batch items (source.type = "urls").

Mirrors ``test_batch_worker_images.py``: ``_process_item`` runs with a REAL
``Extractor`` against fakes for the repo/result-store/billing, but the
download step hits an ``httpx.MockTransport`` instead of the network. DNS
resolution (the SSRF guard's ``_assert_public_url``) is monkeypatched to a
public IP, same pattern as ``test_webhooks_transport.py``.
"""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from extract.core import Extractor
from extract.core.batch import ItemErrorCode
from extract.repos.batches import ClaimedItem
from extract.storage.inline import InlineStorage
from extract.workers.batch_worker import WorkerState, _process_item

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_FIXTURE = Path(__file__).parent / "data" / "invoice_synth.pdf"


def _fake_public_getaddrinfo(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]


class _Repo:
    def __init__(self) -> None:
        self.succeeded: list[dict] = []
        self.failed: list[dict] = []
        self.rescheduled: list[dict] = []

    async def heartbeat_lease(self, **kwargs) -> None:
        pass

    async def update_item_succeeded(self, **kwargs) -> None:
        self.succeeded.append(kwargs)

    async def update_item_failed(self, **kwargs) -> None:
        self.failed.append(kwargs)

    async def reschedule_item(self, **kwargs) -> None:
        self.rescheduled.append(kwargs)


class _ResultStore:
    """``fetch_upload`` raises if called — url-sourced items must never touch
    it; the source bytes come from the direct url fetch instead."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    async def fetch_upload(self, *, bucket: str, key: str) -> bytes:
        raise AssertionError("url-sourced item must not call fetch_upload (no S3 round-trip)")

    async def write_result(self, *, batch_id: str, item_id: str, body: bytes):
        self.written.append(body)
        return SimpleNamespace(
            bucket="results", key=f"{batch_id}/{item_id}.json", bytes_written=len(body)
        )


def _claimed_url(url: str) -> ClaimedItem:
    return ClaimedItem(
        item_id="item_url",
        batch_id="batch_url",
        customer_id="cus_url",
        phi_safe=False,
        file_id=None,
        file_s3_bucket=None,
        file_s3_key=None,
        file_filename=None,
        file_content_type=None,
        lease_token="lease",
        lease_expires_at=datetime.now(tz=UTC),
        attempts=1,
        extract_text=True,
        extract_images=False,
        ocr="auto",
        engine="baseline",
        url=url,
    )


def _state(http_client: httpx.AsyncClient) -> WorkerState:
    state = WorkerState()
    state.repo = _Repo()  # type: ignore[assignment]
    state.result_store = _ResultStore()  # type: ignore[assignment]
    state.http = http_client
    state.extractor = Extractor(storage=InlineStorage())
    state.stop_event = asyncio.Event()
    return state


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_url_item_fetches_directly_and_succeeds(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    pdf_bytes = PDF_FIXTURE.read_bytes()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=pdf_bytes))
    )
    state = _state(client)
    await _process_item(state, _claimed_url("https://example.com/report.pdf"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.failed == []
    assert len(repo.succeeded) == 1

    result = json.loads(state.result_store.written[0])  # type: ignore[union-attr]
    assert result["file_id"] is None
    assert result["url"] == "https://example.com/report.pdf"


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_url_item_echoes_url_with_query_string_stripped(monkeypatch):
    """The full url (incl. a presigned signature) is what's actually fetched,
    but the persisted/echoed result never carries the query string."""
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    pdf_bytes = PDF_FIXTURE.read_bytes()
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, content=pdf_bytes)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = _state(client)
    signed_url = "https://bucket.s3.amazonaws.com/patients/doc.pdf?X-Amz-Signature=supersecret"
    await _process_item(state, _claimed_url(signed_url))

    # The fetch connects to the VETTED IP (SSRF pinning, 2026-07-30) rather than
    # re-resolving the name — but everything a presigned URL is signed over must
    # survive that rewrite, or S3 would reject us. So assert the properties that
    # actually matter instead of the literal URL string:
    sent = seen_requests[0]
    assert sent.url.host == "93.184.216.34", "should connect to the pinned address"
    assert sent.url.path == "/patients/doc.pdf"
    assert str(sent.url.query, "utf-8") == "X-Amz-Signature=supersecret"
    # ...and the Host header still carries the real bucket, which is what SigV4
    # signs and what S3 matches on.
    assert sent.headers["host"] == "bucket.s3.amazonaws.com"
    # The persisted result never does.
    result = json.loads(state.result_store.written[0])  # type: ignore[union-attr]
    assert result["url"] == "https://bucket.s3.amazonaws.com/patients/doc.pdf"
    assert "X-Amz-Signature" not in result["url"]


async def test_url_item_404_fails_terminal_not_retried(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    state = _state(client)
    await _process_item(state, _claimed_url("https://example.com/missing.pdf"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.rescheduled == []
    assert len(repo.failed) == 1
    assert repo.failed[0]["error_code"] == ItemErrorCode.URL_FETCH_FAILED


async def test_url_item_403_expired_presigned_fails_terminal_not_retried(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    state = _state(client)
    await _process_item(state, _claimed_url("https://bucket.s3.amazonaws.com/x?sig=expired"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.rescheduled == []
    assert len(repo.failed) == 1
    assert repo.failed[0]["error_code"] == ItemErrorCode.URL_FETCH_FAILED


async def test_url_item_500_reschedules_as_transient(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    state = _state(client)
    await _process_item(state, _claimed_url("https://example.com/flaky.pdf"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.failed == []
    assert len(repo.rescheduled) == 1
    assert repo.rescheduled[0]["error_code"] == ItemErrorCode.INTERNAL_ERROR


async def test_url_item_failed_error_message_strips_presigned_signature(monkeypatch):
    """A failed fetch's error_message embeds the url verbatim (our own prefix
    plus httpx's own 'for url ...' phrasing) — both occurrences must have the
    query string (the live signature) stripped before it's persisted/returned,
    matching the redaction already applied to the item's `url` field."""
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    state = _state(client)
    signed_url = "https://bucket.s3.amazonaws.com/patients/doc.pdf?X-Amz-Signature=supersecret"
    await _process_item(state, _claimed_url(signed_url))

    repo: _Repo = state.repo  # type: ignore[assignment]
    message = repo.failed[0]["error_message"]
    assert "X-Amz-Signature" not in message
    assert "supersecret" not in message
    assert "https://bucket.s3.amazonaws.com/patients/doc.pdf" in message


async def test_url_item_rescheduled_error_message_strips_presigned_signature(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    state = _state(client)
    signed_url = "https://bucket.s3.amazonaws.com/patients/doc.pdf?X-Amz-Signature=supersecret"
    await _process_item(state, _claimed_url(signed_url))

    repo: _Repo = state.repo  # type: ignore[assignment]
    message = repo.rescheduled[0]["error_message"]
    assert "X-Amz-Signature" not in message
    assert "supersecret" not in message


async def test_url_item_never_touches_result_store_fetch_upload(monkeypatch):
    """``_ResultStore.fetch_upload`` raises if called; a passing run here
    proves url-sourced items skip our S3 uploads bucket entirely."""
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    state = _state(client)
    await _process_item(state, _claimed_url("https://example.com/x.pdf"))
    # No AssertionError raised => fetch_upload was never called.
    assert state.result_store.written == []  # type: ignore[union-attr]


async def test_url_item_blocked_private_ip_fails_terminal(monkeypatch):
    def fake_private(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 80))]

    monkeypatch.setattr("socket.getaddrinfo", fake_private)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    state = _state(client)
    await _process_item(state, _claimed_url("https://internal.example/x.pdf"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert len(repo.failed) == 1
    assert repo.failed[0]["error_code"] == ItemErrorCode.URL_FETCH_FAILED
    assert repo.rescheduled == []
