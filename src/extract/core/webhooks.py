"""Pure webhook primitives for the async-batch completion webhooks.

Everything here is deterministic and dependency-free (stdlib only):
signing is **Svix-wire-compatible** so the stock ``svix`` library verifies our
deliveries unchanged; the event body is serialized exactly once at enqueue and
those bytes are what every attempt signs and sends (never re-serialize — key
ordering / whitespace drift would break verification).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets as _secrets
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

# --- Constants ---------------------------------------------------------------


class WebhookMode:
    SVIX = "svix"  # delivery to registered endpoints, Svix-wire-compatible
    REGISTERED = "registered"  # accepted alias for SVIX (docs mention only svix)
    DIRECT = "direct"  # unsigned, inline per-request URL
    DISABLED = "disabled"

    ALL = frozenset({SVIX, REGISTERED, DIRECT, DISABLED})


EVENT_BATCH_UPDATE = "batch.update"
EVENT_PING = "webhook.ping"

# Svix retry ladder: initial attempt, then these delays. 8 attempts total,
# the last ~27h35m after the first. Applies to both modes.
RETRY_LADDER_SECONDS: tuple[int, ...] = (5, 300, 1800, 7200, 18000, 36000, 36000)
MAX_ATTEMPTS = len(RETRY_LADDER_SECONDS) + 1

# Success requires a 2xx within this budget; redirects and every other status
# are failures.
DELIVERY_TIMEOUT_SECONDS = 15.0

# Caller-authored metadata is echoed into every webhook body; cap at create
# time.
METADATA_MAX_BYTES = 16 * 1024

SECRET_PREFIX = "whsec_"

DELIVERY_STATUS_PENDING = "pending"
DELIVERY_STATUS_DELIVERING = "delivering"
DELIVERY_STATUS_SUCCEEDED = "succeeded"
DELIVERY_STATUS_FAILED = "failed"
DELIVERY_STATUS_CANCELLED = "cancelled"


# --- Ids ---------------------------------------------------------------------


def new_message_id() -> str:
    """The svix-id: stable across every retry of one event (consumer dedup key)."""
    return f"msg_{uuid.uuid4().hex}"


def new_delivery_id() -> str:
    return f"whd_{uuid.uuid4().hex}"


def new_endpoint_id() -> str:
    return f"whe_{uuid.uuid4().hex}"


# --- Secrets -----------------------------------------------------------------


def generate_secret() -> str:
    """``whsec_`` + base64(24 random bytes) — the exact Svix secret format."""
    return SECRET_PREFIX + base64.b64encode(_secrets.token_bytes(24)).decode("ascii")


def secret_bytes(secret: str) -> bytes:
    """The HMAC key: base64-decode of the secret after the ``whsec_`` prefix."""
    raw = secret[len(SECRET_PREFIX) :] if secret.startswith(SECRET_PREFIX) else secret
    return base64.b64decode(raw)


# --- Signing (Svix scheme) ---------------------------------------------------


def sign(secret: str, *, msg_id: str, timestamp: int, body: bytes) -> str:
    """One ``v1,<base64sig>`` component: HMAC-SHA256 over ``{id}.{timestamp}.{body}``."""
    to_sign = f"{msg_id}.{timestamp}.".encode() + body
    digest = hmac.new(secret_bytes(secret), to_sign, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode("ascii")


def signature_headers(
    secrets: list[str],
    *,
    msg_id: str,
    body: bytes,
    timestamp: int | None = None,
) -> dict[str, str]:
    """Full header set for one attempt.

    ``timestamp`` is fresh per attempt (re-sign each retry over the SAME
    stored body). ``secrets`` carries [current] or [current, old] during the
    24h rotation overlap — the signature header is space-delimited. Also emits
    the Standard-Webhooks aliases; svix libs accept either.
    """
    ts = timestamp if timestamp is not None else int(datetime.now(tz=UTC).timestamp())
    sigs = " ".join(sign(s, msg_id=msg_id, timestamp=ts, body=body) for s in secrets)
    return {
        "svix-id": msg_id,
        "svix-timestamp": str(ts),
        "svix-signature": sigs,
        "webhook-id": msg_id,
        "webhook-timestamp": str(ts),
        "webhook-signature": sigs,
        "content-type": "application/json",
    }


# --- Event bodies (serialized ONCE; these bytes are immutable) ----------------


def _iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    aware = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_batch_update_body(
    *,
    batch_id: str,
    status: str,
    counts: dict[str, int],
    total_items: int,
    metadata: dict[str, Any] | None,
    completed_at: datetime | None,
    results_expires_at: datetime | None,
) -> bytes:
    """The one v1 event body. Thin by design: ids + status + counts +
    echoed metadata — never results or result URLs. ``results_expires_at`` is
    the batch's result-retention deadline (same value the resend path
    honors): after it, ``GET /v1/batches/{id}`` results are gone, so consumers
    know how long they have to fetch."""
    payload = {
        "type": EVENT_BATCH_UPDATE,
        "batch_id": batch_id,
        "status": status,
        "counts": counts,
        "total_items": total_items,
        "metadata": metadata,
        "completed_at": _iso_z(completed_at),
        "results_expires_at": _iso_z(results_expires_at),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_ping_body(ping_id: str) -> bytes:
    """Sentinel test event — docs tell handlers to short-circuit on the type,
    and the batch_id is unmistakably fake so nobody tries to GET it."""
    payload = {"type": EVENT_PING, "batch_id": f"ping_{ping_id}"}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# --- Request-time validation ---------------------------------------------------


def normalize_mode(mode: str | None) -> str | None:
    """Public alias handling: ``registered`` -> ``svix``. None stays None
    (omitted == disabled — webhooks are opt-in per batch)."""
    if mode is None:
        return None
    if mode == WebhookMode.REGISTERED:
        return WebhookMode.SVIX
    return mode


def validate_webhook_url(url: str) -> str | None:
    """Syntactic checks shared by endpoint registration and direct mode:
    HTTPS-only, a hostname (not a raw IP literal), not obviously
    non-public. Resolve-time + connect-time enforcement lives in the
    SSRF-safe transport; this is the fail-fast, user-facing layer.

    Returns an error string, or None when acceptable.
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        return "webhook URLs must use https"
    host = parts.hostname
    if not host:
        return "webhook URL has no host"
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass  # a hostname, good
    else:
        return "webhook URLs must use a hostname, not a raw IP address"
    lowered = host.lower()
    if lowered == "localhost" or lowered.endswith(".localhost") or "." not in lowered:
        return "webhook URLs must use a public hostname"
    if parts.username or parts.password:
        return "webhook URLs must not embed credentials"
    return None


def validate_metadata_size(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    size = len(json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if size > METADATA_MAX_BYTES:
        return (
            f"metadata is {size} bytes serialized; the maximum is {METADATA_MAX_BYTES} "
            "(it is echoed into every webhook delivery)"
        )
    return None


def next_attempt_delay_seconds(attempts_done: int) -> int | None:
    """Delay before the next attempt given ``attempts_done`` so far, or None
    when the ladder is exhausted (mark the delivery failed)."""
    idx = attempts_done - 1
    if idx < 0:
        return 0
    if idx >= len(RETRY_LADDER_SECONDS):
        return None
    return RETRY_LADDER_SECONDS[idx]
