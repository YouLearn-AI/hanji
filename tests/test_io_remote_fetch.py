"""Tests for the retryable/status_code classification on RemoteFetchError.

The async batch worker needs to know, for a url-sourced item's fetch
failure, whether retrying is worth it (transient) or not (permanent — bad
url, forbidden, not found, expired presigned signature, SSRF-blocked). These
tests exercise ``_download_guarded`` (the path url-sourced batch items use)
fully offline: DNS resolution is monkeypatched (same pattern as
``test_webhooks_transport.py``) and the HTTP layer is an ``httpx.MockTransport``.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from extract.core.errors import RemoteFetchError
from extract.core.io import load_bytes

PUBLIC_IP = "93.184.216.34"  # example.com


def _fake_public_getaddrinfo(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 443))]


def _fake_private_getaddrinfo(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port or 443))]


def _client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_guarded_download_success_returns_bytes(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(200, content=b"hello"))
    data = await load_bytes(
        url="https://example.com/doc.pdf", client=client, block_private=True
    )
    assert data == b"hello"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
async def test_guarded_download_4xx_is_not_retryable(monkeypatch, status):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(status))
    with pytest.raises(RemoteFetchError) as exc_info:
        await load_bytes(url="https://example.com/doc.pdf", client=client, block_private=True)
    err = exc_info.value
    assert err.retryable is False
    assert err.status_code == status


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503])
async def test_guarded_download_transient_status_is_retryable(monkeypatch, status):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(status))
    with pytest.raises(RemoteFetchError) as exc_info:
        await load_bytes(url="https://example.com/doc.pdf", client=client, block_private=True)
    err = exc_info.value
    assert err.retryable is True
    assert err.status_code == status


async def test_guarded_download_network_error_is_retryable(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    client = _client_for(handler)
    with pytest.raises(RemoteFetchError) as exc_info:
        await load_bytes(url="https://example.com/doc.pdf", client=client, block_private=True)
    err = exc_info.value
    assert err.retryable is True
    assert err.status_code is None


async def test_guarded_download_blocks_private_ip(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_private_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(200, content=b"should not be reached"))
    with pytest.raises(RemoteFetchError) as exc_info:
        await load_bytes(
            url="https://internal.example/doc.pdf", client=client, block_private=True
        )
    assert exc_info.value.retryable is False


async def test_guarded_download_rejects_non_http_scheme(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(200, content=b"unreachable"))
    with pytest.raises(RemoteFetchError) as exc_info:
        await load_bytes(url="ftp://example.com/doc.pdf", client=client, block_private=True)
    assert exc_info.value.retryable is False


async def test_remote_fetch_error_is_still_an_extraction_failed(monkeypatch):
    """Existing callers (the sync url route) catch ``ExtractionFailed`` — a
    ``RemoteFetchError`` must still satisfy that isinstance check."""
    from extract.core.errors import ExtractionFailed

    monkeypatch.setattr("socket.getaddrinfo", _fake_public_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(404))
    with pytest.raises(ExtractionFailed):
        await load_bytes(url="https://example.com/doc.pdf", client=client, block_private=True)


async def test_default_is_guarded(monkeypatch):
    """``block_private`` defaults TRUE — a caller that says nothing is guarded.

    This asserted the opposite until 2026-07-30, which is how three routes
    shipped unguarded and an attacker reached the metadata address through the
    anonymous playground. The default IS the security property, so it is what
    gets pinned; the legacy path now has to be requested explicitly.
    """
    resolved: list[str] = []

    def recording_getaddrinfo(host, *_args, **_kwargs):
        resolved.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("socket.getaddrinfo", recording_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(200, content=b"guarded ok"))
    data = await load_bytes(url="https://example.com/doc.pdf", client=client)
    assert data == b"guarded ok"
    assert resolved == ["example.com"], "the default path must resolve + vet the host"


async def test_legacy_path_still_available_when_asked_for_explicitly(monkeypatch):
    """The opt-out survives, but now has to be spelled out so review sees it."""

    def fake_getaddrinfo(*_args, **_kwargs):
        raise AssertionError("the opted-out path must not resolve DNS itself")

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    client = _client_for(lambda r: httpx.Response(200, content=b"legacy ok"))
    data = await load_bytes(url="https://example.com/doc.pdf", client=client, block_private=False)
    assert data == b"legacy ok"
