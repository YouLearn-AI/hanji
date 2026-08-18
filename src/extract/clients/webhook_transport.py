"""SSRF-safe delivery transport for webhooks.

Every attempt (not just registration time):

1. Resolves the destination host and rejects any non-public A/AAAA record
   (reuses ``extract.core.io._is_blocked_ip`` — RFC1918, loopback, the entire
   169.254.0.0/16 link-local range incl. cloud metadata endpoints, multicast,
   reserved; one blocklist, no drift).
2. **Pins the vetted IP for the connection while preserving TLS SNI +
   certificate hostname verification + the Host header** — closes the
   DNS-rebind TOCTOU gap that a resolve-then-connect check leaves open.
3. Never follows redirects (any 3xx is a failed attempt).
4. Enforces a hard total deadline (15s).
5. Reads only the status line; the response body is closed unread.

``EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS=true`` (self-host escape hatch) skips
the https requirement, the resolve-time blocklist, and the IP pinning so a
deployment can deliver to services on its own private network. Leave it off
whenever endpoint URLs come from untrusted users.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging as _logging
import random
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

from extract.config import settings
from extract.core.io import _is_blocked_ip  # single shared blocklist
from extract.core.webhooks import DELIVERY_TIMEOUT_SECONDS

# httpx logs "HTTP Request: POST <full url>" at INFO — destination URLs
# should not leak into logs (they can embed tokens). Defense in depth;
# logger levels are process-global.
for _leaky in ("httpx", "httpcore"):
    _logging.getLogger(_leaky).setLevel(_logging.WARNING)


class WebhookDeliveryBlocked(Exception):
    """The destination failed SSRF validation — never retried as transient."""


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    status_code: int | None
    error_code: str | None  # finite vocabulary for logs/metrics (PHI-safe)
    error_detail: str | None  # stored on the delivery row only, never logged


def _is_ipv6(ip: str) -> bool:
    return ":" in ip


async def _resolve_public_ip(host: str, port: int) -> str:
    """Resolve and return one vetted public IP, or raise WebhookDeliveryBlocked.

    Prefers IPv4 when any A record is present: worker egress commonly lacks
    working IPv6 paths, and picking a random AAAA (Happy-Eyeballs-unaware
    pin) has produced connect failures that never reached the consumer's
    process while IPv4 succeeded. IPv6 is still used when the host is
    AAAA-only.
    """
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise WebhookDeliveryBlocked(f"could not resolve host: {e}") from e
    ips = [info[4][0] for info in infos]
    if not ips:
        raise WebhookDeliveryBlocked("host resolved to no addresses")
    for ip in ips:
        if _is_blocked_ip(ip):
            raise WebhookDeliveryBlocked(f"host resolves to a non-public address ({ip})")
    # Prefer IPv4 among already-vetted public addresses; fall back to IPv6
    # only when the name has no A records.
    v4 = [ip for ip in ips if not _is_ipv6(ip)]
    pool = v4 or ips
    return random.choice(pool)


def _pin_url(url: str, ip: str) -> str:
    parts = urlsplit(url)
    host = ip
    if ":" in ip:  # IPv6 literal needs brackets
        host = f"[{ip}]"
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{host}{port}", parts.path, parts.query, ""))


async def deliver(
    *,
    url: str,
    body: bytes,
    headers: dict[str, str],
    client: httpx.AsyncClient | None = None,
    timeout_seconds: float = DELIVERY_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """POST ``body`` to ``url`` through the pinned-IP transport.

    Success is any 2xx within the deadline. Every other outcome maps to a
    finite ``error_code`` (``ssrf_blocked`` | ``timeout`` | ``connect_error``
    | ``tls_error`` | ``non_2xx``) so logs and metrics stay PHI-clean.
    """
    parts = urlsplit(url)
    allow_private = settings.EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS
    if parts.scheme != "https" and not allow_private:
        return DeliveryResult(False, None, "ssrf_blocked", "non-https URL")
    if parts.scheme not in ("http", "https"):
        return DeliveryResult(False, None, "ssrf_blocked", "non-http(s) URL")
    host = parts.hostname
    if not host:
        return DeliveryResult(False, None, "ssrf_blocked", "URL has no host")
    if not allow_private:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return DeliveryResult(False, None, "ssrf_blocked", "raw-IP destination")

    if allow_private:
        pinned_url = url
        send_headers = dict(headers)
    else:
        try:
            ip = await _resolve_public_ip(host, parts.port or 443)
        except WebhookDeliveryBlocked as e:
            return DeliveryResult(False, None, "ssrf_blocked", str(e))
        pinned_url = _pin_url(url, ip)
        send_headers = dict(headers)
        send_headers["host"] = parts.netloc.rsplit("@", 1)[-1]

    own_client = client is None
    if client is None:
        client = _new_client()
    try:
        async with asyncio.timeout(timeout_seconds):
            # sni_hostname: httpcore uses it as ssl server_hostname, so both
            # SNI and certificate verification run against the REAL hostname
            # even though the TCP connection goes to the pinned IP.
            request = client.build_request(
                "POST",
                pinned_url,
                content=body,
                headers=send_headers,
                extensions={"sni_hostname": host},
            )
            response = await client.send(request, stream=True)
            try:
                status = response.status_code
            finally:
                await response.aclose()  # discard the body unread, capped at zero
        if 200 <= status < 300:
            return DeliveryResult(True, status, None, None)
        if 300 <= status < 400:
            return DeliveryResult(False, status, "non_2xx", "redirect refused")
        return DeliveryResult(False, status, "non_2xx", f"HTTP {status}")
    except TimeoutError:
        return DeliveryResult(False, None, "timeout", f"no 2xx within {timeout_seconds}s")
    except httpx.ConnectError as e:
        detail = str(e)
        code = "tls_error" if "SSL" in detail or "certificate" in detail.lower() else "connect_error"
        return DeliveryResult(False, None, code, detail[:500])
    except httpx.HTTPError as e:
        return DeliveryResult(False, None, "connect_error", str(e)[:500])
    finally:
        if own_client:
            await client.aclose()


def _new_client() -> httpx.AsyncClient:
    """A dedicated client: no redirects, per-phase timeouts under the hard
    deadline, no shared state with the extraction client (`state.http`)."""
    client = httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(DELIVERY_TIMEOUT_SECONDS, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
    )
    return client


__all__ = ["DeliveryResult", "WebhookDeliveryBlocked", "deliver", "_new_client"]
