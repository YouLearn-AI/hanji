"""Golden signature vectors for the webhook signing scheme (plan 083 §5).

The drop-in guarantee, as a unit test: the STOCK ``svix`` library's
``Webhook.verify`` must accept our emitted headers + immutable body. If these
fail, a ported Reducto handler breaks — this is the contract that matters.
"""

from __future__ import annotations

import json
import time

import pytest
from svix.webhooks import Webhook, WebhookVerificationError

from extract.core.webhooks import (
    METADATA_MAX_BYTES,
    build_batch_update_body,
    build_ping_body,
    generate_secret,
    new_message_id,
    next_attempt_delay_seconds,
    normalize_mode,
    signature_headers,
    validate_metadata_size,
    validate_webhook_url,
)

BODY = build_batch_update_body(
    batch_id="batch_0123abc",
    status="completed",
    counts={"pending": 0, "running": 0, "succeeded": 4, "failed": 1, "cancelled": 0},
    total_items=5,
    metadata={"prior_auth_id": "PA-1234"},
    completed_at=None,
    results_expires_at=None,
)


def test_svix_lib_verifies_our_headers():
    secret = generate_secret()
    msg_id = new_message_id()
    headers = signature_headers([secret], msg_id=msg_id, body=BODY)
    verified = Webhook(secret).verify(BODY, headers)
    assert verified["batch_id"] == "batch_0123abc"
    assert verified["status"] == "completed"
    assert verified["counts"]["succeeded"] == 4


def test_svix_lib_rejects_wrong_secret():
    headers = signature_headers([generate_secret()], msg_id=new_message_id(), body=BODY)
    with pytest.raises(WebhookVerificationError):
        Webhook(generate_secret()).verify(BODY, headers)


def test_svix_lib_rejects_tampered_body():
    secret = generate_secret()
    headers = signature_headers([secret], msg_id=new_message_id(), body=BODY)
    tampered = BODY.replace(b'"completed"', b'"failed___"')
    with pytest.raises(WebhookVerificationError):
        Webhook(secret).verify(tampered, headers)


def test_rotation_overlap_dual_signature_verifies_with_either_secret():
    """During the 24h rotation overlap the header carries both signatures —
    a consumer still on the OLD secret keeps verifying."""
    new, old = generate_secret(), generate_secret()
    headers = signature_headers([new, old], msg_id=new_message_id(), body=BODY)
    assert Webhook(new).verify(BODY, headers) is not None
    assert Webhook(old).verify(BODY, headers) is not None


def test_retry_resigns_fresh_timestamp_over_same_body():
    """Retries re-sign a fresh timestamp over the SAME stored bytes; the
    message id (svix-id) is stable across attempts — the dedup key."""
    secret = generate_secret()
    msg_id = new_message_id()
    first = signature_headers([secret], msg_id=msg_id, body=BODY, timestamp=int(time.time()) - 30)
    second = signature_headers([secret], msg_id=msg_id, body=BODY)
    assert first["svix-id"] == second["svix-id"]
    assert first["svix-timestamp"] != second["svix-timestamp"]
    assert first["svix-signature"] != second["svix-signature"]
    assert Webhook(secret).verify(BODY, second) is not None


def test_stale_timestamp_rejected_replay_protection():
    secret = generate_secret()
    headers = signature_headers(
        [secret], msg_id=new_message_id(), body=BODY, timestamp=int(time.time()) - 3600
    )
    with pytest.raises(WebhookVerificationError):
        Webhook(secret).verify(BODY, headers)


def test_standard_webhooks_aliases_present():
    headers = signature_headers([generate_secret()], msg_id=new_message_id(), body=BODY)
    assert headers["webhook-id"] == headers["svix-id"]
    assert headers["webhook-timestamp"] == headers["svix-timestamp"]
    assert headers["webhook-signature"] == headers["svix-signature"]


def test_body_shape_matches_spec():
    payload = json.loads(BODY)
    assert payload == {
        "type": "batch.update",
        "batch_id": "batch_0123abc",
        "status": "completed",
        "counts": {"pending": 0, "running": 0, "succeeded": 4, "failed": 1, "cancelled": 0},
        "total_items": 5,
        "metadata": {"prior_auth_id": "PA-1234"},
        "completed_at": None,
        "results_expires_at": None,
    }


def test_results_expires_at_serialized_iso_z():
    from datetime import UTC, datetime

    body = json.loads(
        build_batch_update_body(
            batch_id="batch_x",
            status="completed",
            counts={"pending": 0, "running": 0, "succeeded": 1, "failed": 0, "cancelled": 0},
            total_items=1,
            metadata=None,
            completed_at=datetime(2026, 8, 5, 12, 34, 56, tzinfo=UTC),
            results_expires_at=datetime(2026, 8, 8, 12, 34, 56, tzinfo=UTC),
        )
    )
    assert body["completed_at"] == "2026-08-05T12:34:56Z"
    assert body["results_expires_at"] == "2026-08-08T12:34:56Z"


def test_ping_body_is_sentinel():
    payload = json.loads(build_ping_body("abc123"))
    assert payload == {"type": "webhook.ping", "batch_id": "ping_abc123"}


# --- ladder + validation helpers ---------------------------------------------


def test_retry_ladder_is_svix_parity():
    delays = [next_attempt_delay_seconds(n) for n in range(1, 9)]
    assert delays == [5, 300, 1800, 7200, 18000, 36000, 36000, None]
    # 8 attempts total; the last lands ~27h35m after the first.
    assert sum(d for d in delays if d) == 27 * 3600 + 35 * 60 + 5


def test_normalize_mode_registered_alias():
    assert normalize_mode("registered") == "svix"
    assert normalize_mode("svix") == "svix"
    assert normalize_mode(None) is None
    assert normalize_mode("disabled") == "disabled"


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://hooks.example.com/extract", True),
        ("http://hooks.example.com/extract", False),  # https only
        ("https://203.0.113.7/hook", False),  # raw IP
        ("https://localhost/hook", False),
        ("https://internal/hook", False),  # no dot
        ("https://user:pw@hooks.example.com/x", False),  # embedded creds
    ],
)
def test_validate_webhook_url(url: str, ok: bool):
    assert (validate_webhook_url(url) is None) == ok


def test_metadata_cap():
    assert validate_metadata_size({"k": "v"}) is None
    big = {"k": "x" * (METADATA_MAX_BYTES + 1)}
    assert validate_metadata_size(big) is not None
