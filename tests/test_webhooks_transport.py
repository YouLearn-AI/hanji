"""SSRF-safe transport tests (plan 083 §3.7/§5). LAUNCH-BLOCKING: direct mode
is ungated (§7-Q8), so this guard is load-bearing from day one for both modes.
"""

from __future__ import annotations

import httpx
import pytest

from extract.clients import webhook_transport
from extract.clients.webhook_transport import WebhookDeliveryBlocked, deliver


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.example.com/x",  # not https
        "https://127.0.0.1/x",  # raw loopback IP
        "https://10.0.0.5/x",  # raw RFC1918 IP
        "https://169.254.169.254/latest/meta-data",  # EC2 IMDS
        "https://169.254.170.2/v2/credentials",  # ECS credential endpoint
        "https://[::1]/x",  # IPv6 loopback literal
    ],
)
async def test_blocked_destinations(url: str):
    result = await deliver(url=url, body=b"{}", headers={})
    assert not result.ok
    assert result.error_code == "ssrf_blocked"


async def test_hostname_resolving_to_metadata_ip_blocked(monkeypatch):
    """DNS-rebind shape: a public-looking hostname whose A record is the ECS
    credential endpoint."""

    def fake_getaddrinfo(host, port, **kwargs):
        return [(2, 1, 6, "", ("169.254.170.2", 443))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    result = await deliver(url="https://evil.example.com/hook", body=b"{}", headers={})
    assert not result.ok
    assert result.error_code == "ssrf_blocked"


async def test_localhost_hostname_blocked():
    result = await deliver(url="https://localhost/hook", body=b"{}", headers={})
    assert not result.ok
    assert result.error_code == "ssrf_blocked"


async def test_partially_private_resolution_blocked(monkeypatch):
    """If ANY resolved address is non-public the whole destination is refused
    (an attacker controls which record a racing resolver sees)."""

    def fake_getaddrinfo(host, port, **kwargs):
        return [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("10.1.2.3", 443)),
        ]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    with pytest.raises(WebhookDeliveryBlocked):
        await webhook_transport._resolve_public_ip("half-evil.example.com", 443)


async def test_resolve_prefers_ipv4_when_mixed(monkeypatch):
    """Dual-stack hosts must pin an A record, not a random AAAA.

    Worker egress often cannot complete IPv6 connects; random AAAA selection
    caused intermittent 'All connection attempts failed' deliveries.
    """

    def fake_getaddrinfo(host, port, **kwargs):
        return [
            (10, 1, 6, "", ("2600:1f16:d83:1200::1", 443, 0, 0)),
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (10, 1, 6, "", ("2600:1f16:d83:1201::2", 443, 0, 0)),
            (2, 1, 6, "", ("93.184.216.35", 443)),
        ]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    chosen = {await webhook_transport._resolve_public_ip("dual.example.com", 443) for _ in range(40)}
    assert chosen <= {"93.184.216.34", "93.184.216.35"}
    assert chosen  # at least one pick


async def test_resolve_uses_ipv6_when_aaaa_only(monkeypatch):
    def fake_getaddrinfo(host, port, **kwargs):
        return [
            (10, 1, 6, "", ("2606:2800:220:1::248", 443, 0, 0)),
        ]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    ip = await webhook_transport._resolve_public_ip("v6only.example.com", 443)
    assert ip == "2606:2800:220:1::248"


# --- Pinning + delivery semantics (mock transport, no real network) -----------


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


async def _deliver_via(monkeypatch, handler, url="https://hooks.example.com/x"):
    async def fake_resolve(host, port):
        return "93.184.216.34"

    monkeypatch.setattr(webhook_transport, "_resolve_public_ip", fake_resolve)
    async with _mock_client(handler) as client:
        return await deliver(url=url, body=b'{"a":1}', headers={"x-h": "v"}, client=client)


async def test_connection_pinned_to_vetted_ip_with_host_and_sni(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url_host"] = request.url.host
        seen["host_header"] = request.headers["host"]
        seen["sni"] = request.extensions.get("sni_hostname")
        seen["body"] = request.content
        return httpx.Response(200)

    result = await _deliver_via(monkeypatch, handler)
    assert result.ok and result.status_code == 200
    assert seen["url_host"] == "93.184.216.34"  # TCP target: the pinned IP
    assert seen["host_header"] == "hooks.example.com"  # HTTP host: the real name
    assert seen["sni"] == "hooks.example.com"  # TLS SNI + cert verification name
    assert seen["body"] == b'{"a":1}'


async def test_redirect_is_failure_not_followed(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(302, headers={"location": "https://internal/creds"})

    result = await _deliver_via(monkeypatch, handler)
    assert not result.ok
    assert result.error_code == "non_2xx"
    assert calls["n"] == 1  # never followed


async def test_non_2xx_is_failure(monkeypatch):
    result = await _deliver_via(monkeypatch, lambda r: httpx.Response(500))
    assert not result.ok and result.status_code == 500
    assert result.error_code == "non_2xx"


async def test_timeout_maps_to_timeout_code(monkeypatch):
    import asyncio

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    async def fake_resolve(host, port):
        return "93.184.216.34"

    monkeypatch.setattr(webhook_transport, "_resolve_public_ip", fake_resolve)
    async with _mock_client(slow_handler) as client:
        result = await deliver(
            url="https://hooks.example.com/x",
            body=b"{}",
            headers={},
            client=client,
            timeout_seconds=0.2,
        )
    assert not result.ok
    assert result.error_code == "timeout"


async def test_response_body_never_read(monkeypatch):
    """Blind SSRF property: only the status line is consumed. A body that
    would explode on read must not affect the result."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 10_000_000)

    result = await _deliver_via(monkeypatch, handler)
    assert result.ok
