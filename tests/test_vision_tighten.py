"""Unit tests for Cloud Vision citation tightening (no network — pure span logic
+ a stub Vision client). Verifies the span match, shrink-only + fail-open contract,
value flattening, and in-place evidence update."""
from __future__ import annotations

from PIL import Image

from extract.core.models import FieldEvidence
from extract.core.vision_tighten import (
    CloudVisionTightener,
    _find_span,
    _flatten_values,
    tighten_evidence,
)


def _w(text, l, t, w, h):
    return (text, l, t, w, h)


def test_find_span_exact_consecutive():
    # "531 VINE PLACE" spread across 3 words -> box spans all three
    words = [_w("Street:", 0, 0, 40, 10), _w("531", 50, 0, 20, 10), _w("VINE", 75, 0, 30, 10), _w("PLACE", 110, 0, 40, 10)]
    box = _find_span("531 VINE PLACE", words)
    assert box == (50, 0, 150, 10)  # from "531" left to "PLACE" right


def test_find_span_alnum_insensitive():
    # punctuation/case differences don't matter (alnum concat)
    words = [_w("OSC7643675401", 10, 5, 100, 12)]
    assert _find_span("osc-7643675401", words) == (10, 5, 110, 17)


def test_find_span_date_fallback():
    words = [_w("06-03-1957", 0, 0, 80, 10)]
    assert _find_span("1957-06-03", words) is not None  # date-key match


def test_find_span_no_match_returns_none():
    words = [_w("NOTHING", 0, 0, 50, 10)]
    assert _find_span("531 VINE PLACE", words) is None


def test_flatten_values_nested():
    v = {"a": {"b": 1}, "c": [{"d": 2}]}
    assert _flatten_values(v) == {"a.b": 1, "c[0].d": 2}


class _StubTightener(CloudVisionTightener):
    """Returns a fixed sub-box for any crop whose value is 'HIT'."""

    def _words(self, crop_png):  # noqa: D401 — no network
        return [_w("HIT", 4, 4, 10, 6)]


def test_tighten_box_shrinks_only():
    t = _StubTightener()
    img = Image.new("RGB", (1000, 1000), "white")
    # coarse box is the whole page; the stub finds "HIT" tightly inside the crop
    new = t.tighten_box(img, "HIT", [0, 0, 1000, 1000])
    assert new is not None and (new[2] - new[0]) < 1000  # shrank


def test_tighten_box_failopen_on_empty_value():
    t = _StubTightener()
    img = Image.new("RGB", (1000, 1000), "white")
    assert t.tighten_box(img, "", [0, 0, 100, 100]) is None


def test_tighten_evidence_no_doc_bytes_returns_zero():
    ev = {"x": [FieldEvidence(page=1, bbox=[0, 0, 100, 100], text="v")]}
    assert tighten_evidence(ev, {"x": "v"}, b"") == 0


def test_tighten_evidence_none_evidence_returns_zero():
    assert tighten_evidence(None, {}, b"%PDF-1.4") == 0
