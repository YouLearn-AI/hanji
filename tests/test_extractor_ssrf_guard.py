"""Integration tests for `block_private` threading through `Extractor.aextract`.

`core/io.py`'s `_download_guarded`/`_assert_public_url` guard is already unit
tested directly (see the batch-url-source feature's test suite). These tests
instead prove the `block_private` flag actually reaches that guard from the
top of the `Extractor` API — through both download call sites `aextract` can
hit: its own magic-sniff download (no-extension URLs) and `extract_pdf`'s
download (`.pdf`-extension URLs, the common case). DNS resolution is
monkeypatched so this stays fully offline, same pattern as
`test_webhooks_transport.py` / `test_io_remote_fetch.py`.
"""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

from extract.core import Extractor, ExtractRequest, RemoteFetchError

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_FIXTURE = REPO_ROOT / "web" / "public" / "demo" / "attention.pdf"


def _fake_private_getaddrinfo(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 443))]


def _fake_public_getaddrinfo(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]


def _mock_client(body: bytes = b"unused") -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_pdf_extension_url_blocks_private_ip_when_guarded(monkeypatch):
    """.pdf-extension URLs skip Extractor's own download and go straight to
    extract_pdf's — confirms THAT call site got the guard too."""
    monkeypatch.setattr("socket.getaddrinfo", _fake_private_getaddrinfo)
    request = ExtractRequest(url="https://internal.example/doc.pdf")
    async with _mock_client() as client:
        with pytest.raises(RemoteFetchError) as exc_info:
            await Extractor(download_client=client).aextract(request, block_private=True)
    assert exc_info.value.retryable is False


async def test_no_extension_url_blocks_private_ip_when_guarded(monkeypatch):
    """No-extension URLs (arxiv-style) download inside Extractor.aextract
    itself for magic-sniffing — confirms THIS call site got the guard."""
    monkeypatch.setattr("socket.getaddrinfo", _fake_private_getaddrinfo)
    request = ExtractRequest(url="https://internal.example/report")
    async with _mock_client() as client:
        with pytest.raises(RemoteFetchError):
            await Extractor(download_client=client).aextract(request, block_private=True)


async def test_guard_is_on_by_default(monkeypatch):
    """DEFAULT-DENY: a caller that says nothing is GUARDED.

    This test previously asserted the opposite — that the default took the
    unguarded path — which is precisely how three routes shipped without the
    guard. On 2026-07-30 an attacker used the anonymous playground (one of those
    routes) to probe 169.254.169.254 and friends; only Fargate's lack of an EC2
    metadata endpoint stopped it. The default is the security property here, so
    it is the thing worth pinning.
    """
    async with _mock_client() as client:
        request = ExtractRequest(url="https://internal.example/report")
        with pytest.raises(RemoteFetchError):
            await Extractor(download_client=client).aextract(request)


async def test_guard_can_still_be_opted_out_explicitly(monkeypatch):
    """The escape hatch survives for a caller that genuinely needs it — but it
    now has to be spelled out, so it shows up in review."""

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the opted-out path must not resolve DNS itself")

    monkeypatch.setattr("socket.getaddrinfo", fail_if_called)
    async with _mock_client(b"not a real document") as client:
        from extract.core import UnsupportedInput

        request = ExtractRequest(url="https://internal.example/report")
        with pytest.raises(UnsupportedInput):
            await Extractor(download_client=client).aextract(request, block_private=False)


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_block_private_true_still_succeeds_for_public_address(monkeypatch):
    """The guard only blocks non-public addresses — a real public-resolving
    url still extracts normally with block_private=True."""
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    pdf_bytes = PDF_FIXTURE.read_bytes()
    request = ExtractRequest(url="https://example.com/doc.pdf", ocr="never")
    async with _mock_client(pdf_bytes) as client:
        res = await Extractor(download_client=client).aextract(request, block_private=True)
    assert res.page_count > 0


@pytest.mark.parametrize(
    "host",
    [
        "169.254.169.254",  # EC2 instance metadata
        "169.254.170.2",  # ECS/Fargate task-credentials endpoint
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
    ],
)
async def test_guard_refuses_every_internal_target(host, monkeypatch):
    """The live 2026-07-30 probe set, pinned as a regression.

    The attacker walked the standard encodings of the cloud metadata address;
    the guard resolves the host and judges the RESOLVED ip, which is what makes
    decimal/octal/nip.io/IPv6-mapped spellings collapse to the same answer.
    169.254.170.2 matters most here: on Fargate that is the endpoint that hands
    out this task's role credentials, and the role can read, write and DELETE
    every object in the production document bucket.
    """
    import socket

    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 80))],
    )
    async with _mock_client() as client:
        request = ExtractRequest(url="https://looks-harmless.example/doc.pdf")
        with pytest.raises(RemoteFetchError):
            await Extractor(download_client=client).aextract(request)
