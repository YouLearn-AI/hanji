"""Per-request stage timer.

A ``StageTimer`` is created once per HTTP request, threaded through the
extraction pipeline, and emitted as a structured log line at the end. The
``span(name)`` context manager records elapsed milliseconds; ``meta`` carries
non-timing fields (``doc_size_bytes``, ``pages_ocr_d``, etc.) so the route
handler doesn't need to plumb them through return values.

Designed to be cheap when unused: the optional ``timer`` argument on pipeline
functions defaults to ``None`` and a no-op span avoids ``time.perf_counter``
calls when no timer was passed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from contextlib import contextmanager as _cm


@_cm
def trace_span(name: str):
    """No-op span. The hosted platform hung OpenTelemetry here; the open-source
    build keeps the call sites and drops the exporter."""
    yield


@dataclass
class StageTimer:
    timings_ms: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    # Out-of-band, request-scoped payloads that are deliberately NOT serialized
    # into the structured log / EMF event (``build_extraction_event`` reads only
    # ``timings_ms`` and ``meta``). Use this for material that is too large or too
    # sensitive for logs — e.g. the per-page Gemini-fallback diagnostics (rendered
    # page bytes + raw model text, which is PHI) that the review-artifact writer
    # consumes after the request completes. Never read this in any logging path.
    sidecar: dict[str, object] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            with trace_span(name):
                yield
        finally:
            self.timings_ms[name] = (time.perf_counter() - t0) * 1000.0

    def record_ms(self, name: str, value_ms: float) -> None:
        self.timings_ms[name] = value_ms

    def add_ms(self, name: str, value_ms: float) -> None:
        self.timings_ms[name] = self.timings_ms.get(name, 0.0) + value_ms

    @contextmanager
    def accumulate(self, name: str) -> Iterator[None]:
        """Span variant that adds elapsed time to an existing key.

        Use this when the same logical phase runs many times in a loop and you
        want one summed timing rather than the last iteration's time. For
        single-shot phases prefer ``span``.
        """
        t0 = time.perf_counter()
        try:
            with trace_span(name):
                yield
        finally:
            self.add_ms(name, (time.perf_counter() - t0) * 1000.0)


@contextmanager
def maybe_span(timer: StageTimer | None, name: str) -> Iterator[None]:
    """Span helper that no-ops cleanly when ``timer`` is ``None``.

    Lets pipeline code be unconditional (``with maybe_span(timer, "parse"):``)
    without sprinkling ``if timer is not None`` everywhere.
    """
    if timer is None:
        yield
        return
    with timer.span(name):
        yield


@contextmanager
def maybe_accumulate(timer: StageTimer | None, name: str) -> Iterator[None]:
    """Like ``maybe_span`` but accumulates across repeated calls."""
    if timer is None:
        yield
        return
    with timer.accumulate(name):
        yield
