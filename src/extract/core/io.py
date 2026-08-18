"""Input loading + PDF repair.

Sources: local path, HTTP(S) URL, in-memory bytes. All routes converge on
bytes that the downstream extractor opens with PyMuPDF.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from extract.core.errors import DocumentTooLarge, ExtractionFailed, RemoteFetchError
from extract.logger import get_logger

logger = get_logger()


# Exclude Brotli from Accept-Encoding — some servers return corrupt gzip
# payloads when Brotli is negotiated, which breaks httpx decompression.
_HTTP_HEADERS = {"Accept-Encoding": "gzip, deflate"}

# SSRF guard: cap manual redirect hops (we follow redirects ourselves so each
# hop's resolved address is re-validated — httpx's auto-follow would skip that).
_MAX_REDIRECTS = 5
# See _is_blocked_ip: reachable in-VPC but never a legitimate document host.
_EXTRA_BLOCKED_NETS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("2002::/16"),
)


async def load_bytes(
    *,
    url: str | None = None,
    path: str | None = None,
    data: bytes | None = None,
    max_size: int | None = None,
    client: httpx.AsyncClient | None = None,
    block_private: bool = True,
) -> bytes:
    """Return raw file bytes from whichever input was provided.

    Exactly one of ``url``, ``path``, or ``data`` must be set. ``block_private``
    defaults ``False`` — existing callers keep the original download behavior
    byte-for-byte. The anonymous demo passes ``True`` to opt into the SSRF guard
    (resolve + block non-public addresses + per-redirect revalidation), since it
    fetches attacker-controlled URLs without auth.
    """
    if data is not None:
        _enforce_size(len(data), max_size)
        return data

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ExtractionFailed(f"File not found: {path}")
        size = p.stat().st_size
        _enforce_size(size, max_size)
        return await asyncio.to_thread(p.read_bytes)

    if url is not None:
        return await _download(
            url, max_size=max_size, client=client, block_private=block_private
        )

    raise ExtractionFailed("No input source supplied (url/path/data all None).")


def _safe_content_length(value: str | None) -> int:
    """Parse a Content-Length header, returning 0 for missing/non-numeric values
    (a malformed header from an untrusted server must not crash the download)."""
    if not value:
        return 0
    try:
        return max(0, int(value.strip()))
    except (ValueError, AttributeError):
        return 0


def _is_blocked_ip(ip: str) -> bool:
    """Block loopback / private / link-local (incl. 169.254.169.254 cloud
    metadata) / multicast / reserved / unspecified — IPv4 and IPv6."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → refuse
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return True
    # Ranges Python does not call "private" but which are never a customer's
    # document host, and ARE reachable inside a cloud VPC:
    #   100.64.0.0/10  carrier-grade NAT — AWS uses it for internal routing
    #                  (EKS pod addressing, NAT gateways), so it is in-network.
    #   2002::/16      6to4, which tunnels to an embedded IPv4 address and can
    #                  therefore smuggle a private v4 destination past a v6 check.
    return any(addr in net for net in _EXTRA_BLOCKED_NETS)


async def _assert_public_url(url: str) -> None:
    """Validate scheme + resolve the host, refusing if any resolved address is
    non-public. Note: this is a resolve-time check; DNS-rebinding between this
    check and the connect is a residual gap (would need IP pinning to close).

    Every failure here is permanent — the same URL will fail identically on a
    retry — so each is raised as non-retryable for callers with a retry loop.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise RemoteFetchError(f"Unsupported URL scheme: {parts.scheme!r}", retryable=False)
    host = parts.hostname
    if not host:
        raise RemoteFetchError("URL has no host", retryable=False)
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, parts.port or None)
    except OSError as e:
        raise RemoteFetchError(f"Could not resolve host: {host}", retryable=False) from e
    ips = {info[4][0] for info in infos}
    if not ips:
        raise RemoteFetchError(f"Could not resolve host: {host}", retryable=False)
    for ip in ips:
        if _is_blocked_ip(ip):
            raise RemoteFetchError(
                f"Refusing to fetch a non-public address ({host} -> {ip}).", retryable=False
            )


async def _download(
    url: str,
    *,
    max_size: int | None,
    client: httpx.AsyncClient | None = None,
    block_private: bool = True,
) -> bytes:
    # DEFAULT-DENY (2026-07-30). This used to default to False, so the guard was
    # opt-in — and only 2 of the 5 URL-fetch routes remembered to opt in. The
    # three that did not were the INTERNAL playground routes, i.e. the anonymous
    # surface any visitor can reach, which is exactly backwards. On that day an
    # attacker probed us with 169.254.169.254, its nip.io/decimal/octal/IPv6
    # encodings, localhost and 10.0.0.1 — the public lane refused every one, the
    # playground lane attempted the connection and was saved only by Fargate not
    # exposing EC2's metadata endpoint. A guard you must remember to switch on is
    # a guard that will be forgotten, so it is now on unless a caller explicitly
    # opts out.
    if not block_private:
        return await _download_legacy(url, max_size=max_size, client=client)
    return await _download_guarded(url, max_size=max_size, client=client)


async def _download_legacy(
    url: str,
    *,
    max_size: int | None,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Original download path — preserved verbatim so existing URL extraction
    (sync API, etc.) behaves exactly as before this change."""
    owned_client = client is None
    if client is None:
        timeout = httpx.Timeout(None, connect=30.0)
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        async with client.stream("GET", url, headers=_HTTP_HEADERS) as resp:
            resp.raise_for_status()

            declared = int(resp.headers.get("content-length", 0) or 0)
            if declared:
                _enforce_size(declared, max_size)

            content = bytearray()
            async for chunk in resp.aiter_bytes():
                content.extend(chunk)
                _enforce_size(len(content), max_size)
            return bytes(content)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        # 408/409/429 and 5xx may succeed on a later attempt; other 4xx (bad
        # URL, forbidden, not found, expired signature) will not.
        retryable = status in (408, 409, 429) or status >= 500
        raise RemoteFetchError(
            f"Failed to download {url}: {e}", retryable=retryable, status_code=status
        ) from e
    except httpx.HTTPError as e:
        # Network-level failure (timeout, connection reset, DNS blip) with no
        # response at all — may succeed on a later attempt.
        raise RemoteFetchError(f"Failed to download {url}: {e}", retryable=True) from e
    finally:
        if owned_client:
            await client.aclose()


def _assert_public_url_parts(url: str):
    """Validate scheme + presence of a host, returning the parsed URL.

    Split out from ``_assert_public_url`` so the guarded download can do the
    cheap syntactic checks, then resolve ONCE and connect to that exact address
    — resolving a second time is the rebinding window.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise RemoteFetchError(f"Unsupported URL scheme: {parts.scheme!r}", retryable=False)
    if not parts.hostname:
        raise RemoteFetchError("URL has no host", retryable=False)
    return parts


async def _resolve_and_vet(host: str, port: int) -> str | None:
    """Resolve ``host``, reject if ANY address is non-public, return one to pin.

    Returns ``None`` when ``host`` is already a literal IP: there is no name to
    rebind, so there is nothing to pin — the address itself is vetted instead.

    Rejecting on ANY blocked address (rather than picking a public one) is
    deliberate: a host that answers with both a public and a private record is a
    rebinding attempt, not a multi-homed server we want to reach.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if _is_blocked_ip(host):
            raise RemoteFetchError(
                f"Refusing to fetch a non-public address ({host}).", retryable=False
            )
        return None

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise RemoteFetchError(f"Could not resolve host: {host}", retryable=False) from e
    ips = [info[4][0] for info in infos]
    if not ips:
        raise RemoteFetchError(f"Could not resolve host: {host}", retryable=False)
    for ip in ips:
        if _is_blocked_ip(ip):
            raise RemoteFetchError(
                f"Refusing to fetch a non-public address ({host} -> {ip}).", retryable=False
            )
    return ips[0]


def _pin_url(url: str, ip: str) -> str:
    """Rewrite ``url`` to connect to ``ip`` verbatim, preserving port and path."""
    parts = urlsplit(url)
    host = f"[{ip}]" if ":" in ip else ip
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{host}{port}", parts.path, parts.query, ""))


async def _download_guarded(
    url: str,
    *,
    max_size: int | None,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """SSRF-guarded download for untrusted (anonymous-demo) URLs: resolve +
    block non-public addresses, follow redirects manually so each hop is
    re-validated before connecting."""
    owned_client = client is None
    if client is None:
        timeout = httpx.Timeout(None, connect=30.0)
        client = httpx.AsyncClient(timeout=timeout)

    # Pinning rewrites the request URL to an IP, and httpx embeds that URL in its
    # own exception text. Two things then break: the caller sees an opaque IP
    # instead of the host it asked for, and the batch worker's presigned-URL
    # redaction — which matches on the ORIGINAL url — stops matching, leaking the
    # X-Amz-Signature it exists to strip. So every pinned form is mapped back
    # before any message escapes.
    # Map the pinned IP back to the host it stands for. Deliberately keyed on the
    # IP and not the whole URL: httpx normalizes URLs (spaces, unicode, escaping)
    # before embedding them in its exceptions, so a full-string replace silently
    # misses — and what survived was a pinned URL still carrying the caller's
    # X-Amz-Signature, which the batch worker's redaction then failed to match
    # because it looks for the ORIGINAL url. An IP literal has no such normalized
    # forms, and swapping it back reconstitutes exactly the string that
    # redaction does match.
    pinned_to_original: dict[str, str] = {}

    def _unpin(text: str) -> str:
        for pinned_ip, original_host in pinned_to_original.items():
            text = text.replace(pinned_ip, original_host)
        return text

    try:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            # Validate scheme/host, then CONNECT TO THE VETTED IP rather than
            # re-resolving the name. Resolve-then-connect leaves a DNS-rebinding
            # window: a TTL-0 nameserver answers public for the check and
            # 169.254.170.2 (the Fargate endpoint that vends this task's role) for
            # the connection. Pinning removes the second lookup, so there is no
            # window. Same technique the webhook transport already uses.
            parts = _assert_public_url_parts(current)
            host = parts.hostname or ""
            default_port = 443 if parts.scheme == "https" else 80
            pinned_ip = await _resolve_and_vet(host, parts.port or default_port)
            target = _pin_url(current, pinned_ip) if pinned_ip else current
            if pinned_ip:
                # host, not URL — see _unpin.
                pinned_to_original[pinned_ip] = host
            headers = dict(_HTTP_HEADERS)
            if pinned_ip:
                # Host header + TLS SNI + certificate verification all stay on the
                # REAL hostname; only the TCP destination is the pinned address.
                headers["Host"] = parts.netloc.rsplit("@", 1)[-1]
            extensions = {"sni_hostname": host} if pinned_ip and parts.scheme == "https" else {}
            async with client.stream(
                "GET",
                target,
                headers=headers,
                follow_redirects=False,
                extensions=extensions,
            ) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise RemoteFetchError(
                            f"Redirect without Location: {current}", retryable=False
                        )
                    current = str(httpx.URL(current).join(location))
                    continue
                resp.raise_for_status()

                declared = _safe_content_length(resp.headers.get("content-length"))
                if declared:
                    _enforce_size(declared, max_size)

                content = bytearray()
                async for chunk in resp.aiter_bytes():
                    content.extend(chunk)
                    _enforce_size(len(content), max_size)
                return bytes(content)
        raise RemoteFetchError(f"Too many redirects fetching {url}", retryable=False)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        retryable = status in (408, 409, 429) or status >= 500
        raise RemoteFetchError(
            f"Failed to download {url}: {_unpin(str(e))}",
            retryable=retryable,
            status_code=status,
        ) from e
    except httpx.HTTPError as e:
        raise RemoteFetchError(
            f"Failed to download {url}: {_unpin(str(e))}", retryable=True
        ) from e
    finally:
        if owned_client:
            await client.aclose()


def _enforce_size(actual: int, max_size: int | None) -> None:
    if max_size and actual > max_size:
        raise DocumentTooLarge(f"File is {actual} bytes, exceeds max_size={max_size}.")


def repair_pdf_bytes(pdf_bytes: bytes) -> bytes | None:
    """Attempt to repair a malformed PDF via pikepdf/QPDF.

    Returns repaired bytes on success, or None if the file can't be repaired.
    Safe to call on healthy PDFs — it just linearizes them.
    """
    try:
        import pikepdf  # lazy

        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            buf = io.BytesIO()
            pdf.save(buf, linearize=True)
            return buf.getvalue()
    except Exception as e:
        logger.warning("PDF repair failed: %s", e)
        return None
