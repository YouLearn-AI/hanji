"""Total-deadline + timeout-retry bounds on the Gemini schema-extract client.

Regression cover for 2026-07-30: the retry loop had a per-attempt timeout but no
overall budget, so 4 attempts x 180s plus backoff let a single request run 738s
before failing — a paying customer took that twice and left. These tests pin the
three properties that prevent a recurrence:

  1. the call cannot outlive GEMINI_EXTRACT_TOTAL_DEADLINE_SECONDS,
  2. repeated stalls stop early instead of burning the whole budget,
  3. the failure is a named UpstreamTimeout (-> ledger `upstream_timeout`,
     HTTP 504), not a bare TimeoutError falling through to `internal_error`.

Fast-failing errors (429/503) keep the full retry budget — a throttle is not a
stall, and the point of the change is to stop hangs, not to weaken recovery.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from extract.gemini_extract import GeminiSchemaExtractor
from extract.core.errors import UpstreamTimeout

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def _model(monkeypatch, *, deadline: float, per_attempt: float, max_timeouts: int = 2):
    from extract.config import settings

    monkeypatch.setattr(settings, "GOOGLE_VERTEX_PROJECT", "test-project", raising=False)
    monkeypatch.setattr(settings, "GEMINI_EXTRACT_TOTAL_DEADLINE_SECONDS", deadline, raising=False)
    monkeypatch.setattr(settings, "GEMINI_EXTRACT_TIMEOUT_SECONDS", per_attempt, raising=False)
    monkeypatch.setattr(
        settings, "GEMINI_EXTRACT_MAX_TIMEOUT_ATTEMPTS", max_timeouts, raising=False
    )
    return GeminiSchemaExtractor()


@pytest.mark.asyncio
async def test_total_deadline_bounds_wall_time(monkeypatch):
    """A permanently hanging provider must not outlive the budget.

    The old loop would run 4 x per_attempt here; the deadline must cut it short.
    """
    m = _model(monkeypatch, deadline=0.6, per_attempt=0.25)

    async def _hang(*a, **kw):
        await asyncio.sleep(10)  # never returns within any attempt window

    monkeypatch.setattr(m, "_call_once", _hang)

    started = time.monotonic()
    with pytest.raises(UpstreamTimeout):
        await m.generate_json("p", SCHEMA)
    elapsed = time.monotonic() - started
    # Generous ceiling: the point is "bounded", not a precise stopwatch.
    assert elapsed < 2.0, f"ran {elapsed:.2f}s — deadline not enforced"


@pytest.mark.asyncio
async def test_repeated_stalls_stop_at_max_timeout_attempts(monkeypatch):
    """Two stalls end it; the loop does not spend all four attempts hanging."""
    m = _model(monkeypatch, deadline=30.0, per_attempt=0.1, max_timeouts=2)
    calls = 0

    async def _hang(*a, **kw):
        nonlocal calls
        calls += 1
        raise TimeoutError("vertex stalled")

    monkeypatch.setattr(m, "_call_once", _hang)

    with pytest.raises(UpstreamTimeout):
        await m.generate_json("p", SCHEMA)
    assert calls == 2, f"expected 2 timeout attempts, got {calls}"



@pytest.mark.asyncio
async def test_throttles_keep_full_retry_budget(monkeypatch):
    """A 429 fails fast and usually clears — it must still get all 4 attempts,
    and must succeed if a later attempt does."""
    m = _model(monkeypatch, deadline=30.0, per_attempt=5.0)
    calls = 0

    async def _throttle_then_ok(*a, **kw):
        nonlocal calls
        calls += 1
        if calls < 4:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return {"a": "ok"}

    monkeypatch.setattr(m, "_call_once", _throttle_then_ok)
    monkeypatch.setattr("extract.gemini_extract._BACKOFF_BASE_S", 0.001)

    out = await m.generate_json("p", SCHEMA)
    assert out == {"a": "ok"}
    assert calls == 4


@pytest.mark.asyncio
async def test_non_retryable_raises_immediately(monkeypatch):
    """A 400 is the caller's schema — retrying it is pure latency."""
    m = _model(monkeypatch, deadline=30.0, per_attempt=5.0)
    calls = 0

    async def _bad_request(*a, **kw):
        nonlocal calls
        calls += 1
        raise RuntimeError("400 INVALID_ARGUMENT: bad schema")

    monkeypatch.setattr(m, "_call_once", _bad_request)

    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        await m.generate_json("p", SCHEMA)
    assert calls == 1


@pytest.mark.asyncio
async def test_success_on_first_attempt_is_unchanged(monkeypatch):
    """The happy path must not acquire new behaviour."""
    m = _model(monkeypatch, deadline=30.0, per_attempt=5.0)

    async def _ok(*a, **kw):
        return {"a": "value"}

    monkeypatch.setattr(m, "_call_once", _ok)
    assert await m.generate_json("p", SCHEMA) == {"a": "value"}


def test_upstream_timeout_maps_to_504():
    """Not 400 — the caller's document was fine and the request is retryable."""
    from fastapi import HTTPException

    from extract.api.routes.v1 import _map_extraction_error

    exc = _map_extraction_error(UpstreamTimeout("stalled", elapsed_s=180.0, attempts=2))
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 504
