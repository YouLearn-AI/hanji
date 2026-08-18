"""Pure unit tests for the async-batch state machine + cursor helpers.

No DB, no FastAPI. Mirrors how Sidekiq / Oban / Graphile-Worker test the
deterministic core: feed in counts, assert the derived status; round-trip
cursors end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from extract.core.batch import (
    BatchStatus,
    ItemErrorCode,
    ItemStatus,
    decode_batch_cursor,
    decode_item_cursor,
    derive_batch_status,
    encode_batch_cursor,
    encode_item_cursor,
    is_retryable,
    normalize_counts,
    project_item,
    redact_url,
)


def _counts(**kwargs: int) -> dict[str, int]:
    base = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0}
    base.update(kwargs)
    return base


# --- derive_batch_status ----------------------------------------------------


def test_pending_when_only_pending_items():
    assert derive_batch_status(_counts(pending=5), current=None) == BatchStatus.PENDING


def test_running_when_any_running_item():
    assert derive_batch_status(_counts(pending=4, running=1), current=None) == BatchStatus.RUNNING


def test_completed_when_all_succeeded():
    assert derive_batch_status(_counts(succeeded=10), current=None) == BatchStatus.COMPLETED


def test_failed_when_no_succeeded_and_some_failed():
    assert derive_batch_status(_counts(failed=5), current=None) == BatchStatus.FAILED


def test_cancelled_when_only_cancelled():
    assert derive_batch_status(_counts(cancelled=3), current=None) == BatchStatus.CANCELLED


def test_partially_failed_when_mixed_terminal():
    assert (
        derive_batch_status(_counts(succeeded=4, failed=1), current=None)
        == BatchStatus.PARTIALLY_FAILED
    )


def test_terminal_states_are_sticky():
    # We never undo a terminal state — even if items magically reset.
    for terminal in BatchStatus.TERMINAL:
        assert derive_batch_status(_counts(pending=5), current=terminal) == terminal


def test_running_overrides_pending_when_one_running():
    assert (
        derive_batch_status(_counts(pending=99, running=1), current=BatchStatus.PENDING)
        == BatchStatus.RUNNING
    )


def test_normalize_counts_fills_missing_keys():
    assert normalize_counts({"succeeded": 2}) == _counts(succeeded=2)
    assert normalize_counts(None) == _counts()


# --- Cursors ----------------------------------------------------------------


def test_item_cursor_roundtrip():
    when = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)
    cursor = encode_item_cursor(updated_at=when, item_id="item_xyz")
    out = decode_item_cursor(cursor)
    assert out is not None
    decoded_when, decoded_id = out
    assert decoded_when == when
    assert decoded_id == "item_xyz"


def test_item_cursor_none_passthrough():
    assert decode_item_cursor(None) is None
    assert decode_item_cursor("") is None


def test_item_cursor_invalid_raises():
    with pytest.raises(ValueError):
        decode_item_cursor("not!base64!")


def test_batch_cursor_roundtrip():
    when = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)
    cursor = encode_batch_cursor(created_at=when, batch_id="batch_abc")
    decoded = decode_batch_cursor(cursor)
    assert decoded == (when, "batch_abc")


# --- Error codes ------------------------------------------------------------


def test_payment_required_is_terminal_not_retryable():
    assert not is_retryable(ItemErrorCode.PAYMENT_REQUIRED)


def test_billing_unavailable_is_retryable():
    assert is_retryable(ItemErrorCode.BILLING_UNAVAILABLE)


def test_unknown_error_codes_default_to_terminal():
    assert not is_retryable(None)
    assert not is_retryable("totally_made_up")


def test_url_fetch_failed_is_terminal_not_retryable():
    assert not is_retryable(ItemErrorCode.URL_FETCH_FAILED)


# --- redact_url ---------------------------------------------------------


def test_redact_url_strips_query_string_with_signature():
    url = "https://bucket.s3.amazonaws.com/patients/doc.pdf?X-Amz-Signature=secret&X-Amz-Expires=60"
    assert redact_url(url) == "https://bucket.s3.amazonaws.com/patients/doc.pdf"


def test_redact_url_strips_fragment():
    assert redact_url("https://example.com/a.pdf#page=2") == "https://example.com/a.pdf"


def test_redact_url_passthrough_when_no_query_or_fragment():
    assert redact_url("https://example.com/a.pdf") == "https://example.com/a.pdf"


def test_redact_url_none_and_empty_passthrough():
    assert redact_url(None) is None
    assert redact_url("") == ""


class _FakeItem:
    def __init__(
        self,
        *,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ):
        self.id = "item_1"
        self.batch_id = "batch_1"
        self.file_id = "file_1"
        self.position = 0
        self.status = status
        self.page_count = 3 if status == ItemStatus.SUCCEEDED else None
        self.attempts = 1
        self.started_at = datetime.now(tz=UTC)
        self.completed_at = datetime.now(tz=UTC) if status in ItemStatus.TERMINAL else None
        self.updated_at = datetime.now(tz=UTC)
        self.error_code = error_code
        self.error_message = error_message


def test_project_item_succeeded_includes_result_url():
    item = _FakeItem(status=ItemStatus.SUCCEEDED)
    payload = project_item(item)
    assert payload["status"] == "succeeded"
    assert payload["result_url"] == "/v1/batches/batch_1/items/item_1/result"
    assert "error" not in payload


def test_project_item_failed_includes_error():
    item = _FakeItem(
        status=ItemStatus.FAILED,
        error_code=ItemErrorCode.PAYMENT_REQUIRED,
        error_message="Quota exceeded",
    )
    payload = project_item(item)
    assert payload["status"] == "failed"
    assert "result_url" not in payload
    assert payload["error"] == {
        "code": "payment_required",
        "message": "Quota exceeded",
    }


def test_project_item_pending_has_neither():
    item = _FakeItem(status=ItemStatus.PENDING)
    payload = project_item(item)
    assert payload["status"] == "pending"
    assert "result_url" not in payload
    assert "error" not in payload
