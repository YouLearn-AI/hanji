"""Tests for the Gemini OCR fallback provider.

Pins the provider's prompt to the shared parse-GT labeler and covers the
native-YXYX → OCRPageResult translation (coord transpose + scale, element kinds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from extract.config import settings
from extract.core.ocr import get_provider
from extract.core.ocr.gemini import (
    GEMINI_NATIVE_PROMPT,
    GeminiOcrProvider,
    parse_gemini_response,
)

REPO = Path(__file__).resolve().parent.parent


def test_registered():
    assert isinstance(get_provider("gemini"), GeminiOcrProvider)



def test_default_ocr_thinking_is_disabled():
    assert settings.GEMINI_OCR_THINKING_BUDGET == 0
    assert GeminiOcrProvider()._thinking_budget == 0


def test_parse_transposes_yxyx_and_maps_element_kinds():
    raw = (
        '[{"bbox_2d":[100,50,140,800],"type":"text","text_content":"Hello"},'
        '{"bbox_2d":[200,50,500,500],"type":"image","text_content":"<image>"},'
        '{"bbox_2d":[520,40,700,960],"type":"table",'
        '"text_content":"| a | b |\\n|---|---|\\n| 1 | 2 |"}]'
    )
    res = parse_gemini_response(raw, page_width=1000.0, page_height=1000.0)

    assert len(res.blocks) == 1
    assert res.blocks[0].text == "Hello"
    # YXYX [y=100, x=50, y=140, x=800] -> XYXY [50,100,800,140]; 0-1000 over a
    # 1000x1000 page is an identity scale.
    assert res.blocks[0].bbox == [50.0, 100.0, 800.0, 140.0]
    assert len(res.figures) == 1
    assert len(res.tables) == 1
    assert res.tables[0].n_cols == 2


def test_parse_checkbox_record_stays_text():
    # Checkbox marks are plain glyphs in text ([x]/[ ] convention) — there is
    # no separate selection element.
    raw = '[{"bbox_2d":[100,50,140,800],"type":"text","text_content":"[ ] No"}]'
    res = parse_gemini_response(raw, page_width=1000.0, page_height=1000.0)

    assert len(res.blocks) == 1
    assert res.blocks[0].text == "[ ] No"
    assert res.blocks[0].bbox == [50.0, 100.0, 800.0, 140.0]


def test_parse_routes_kv_records_to_key_values():
    # Gemini tags every record; a "kv" type routes to a KV region (parity with Qwen),
    # winning over content inference even when the region text contains pipes.
    raw = (
        '[{"bbox_2d":[100,50,300,800],"type":"kv",'
        '"text_content":"Next of Kin\\nName: Karen\\nWork Phone: <empty>\\n[ ] POA"},'
        '{"bbox_2d":[400,50,440,800],"type":"text","text_content":"Footer"}]'
    )
    res = parse_gemini_response(raw, page_width=1000.0, page_height=1000.0)

    assert len(res.key_values) == 1
    assert res.key_values[0].text.startswith("Next of Kin")
    assert "Work Phone: <empty>" in res.key_values[0].text
    assert res.tables == []
    assert len(res.blocks) == 1 and res.blocks[0].text == "Footer"


def test_parse_ignores_unparseable_response():
    assert parse_gemini_response("not json", page_width=100.0, page_height=100.0).blocks == []


def test_parse_preserves_emission_order_and_stamps_seq():
    # 2026-07-30 reading-order fix (parity with the Qwen path, 7ccdaa202):
    # Gemini reads in reading order, so the parser must NOT re-sort by bbox
    # center. A 2-column page emitted left-column-first (incl. a mid-column
    # table) is the case the old sort interleaved — and with seq unset the page
    # ALSO fell into the legacy type-bucketed assembly.
    raw = (
        '[{"bbox_2d":[100,50,140,400],"type":"text","text_content":"L1"},'
        '{"bbox_2d":[200,50,240,400],"type":"text","text_content":"L2"},'
        '{"bbox_2d":[300,50,400,400],"type":"table",'
        '"text_content":"| a | b |\\n|---|---|\\n| 1 | 2 |"},'
        '{"bbox_2d":[100,550,140,900],"type":"text","text_content":"R1"}]'
    )
    res = parse_gemini_response(raw, page_width=1000.0, page_height=1000.0)

    assert [b.text for b in res.blocks] == ["L1", "L2", "R1"]
    # seq is the RECORD index, shared across types, so assembly can interleave.
    assert [b.seq for b in res.blocks] == [0, 1, 3]
    assert [t.seq for t in res.tables] == [2]


def test_parse_stamps_seq_on_kv_and_figures():
    raw = (
        '[{"bbox_2d":[100,50,140,400],"type":"kv","text_content":"Name: Jane"},'
        '{"bbox_2d":[200,50,400,400],"type":"image","text_content":"<image>"}]'
    )
    res = parse_gemini_response(raw, page_width=1000.0, page_height=1000.0)
    assert [kv.seq for kv in res.key_values] == [0]
    assert [f.seq for f in res.figures] == [1]


def test_gemini_page_assembles_in_emission_order():
    # The seq the parser stamps must survive page assembly: a Gemini-fallback
    # page now ships the model's order instead of the legacy type-bucketed
    # (text → kv → tables → figures) grouping.
    import pymupdf

    from extract.core.pdf import _chunks_from_ocr_page_result

    raw = (
        '[{"bbox_2d":[100,50,140,400],"type":"text","text_content":"L1"},'
        '{"bbox_2d":[200,50,240,400],"type":"text","text_content":"L2"},'
        '{"bbox_2d":[300,50,400,400],"type":"table",'
        '"text_content":"| a | b |\\n|---|---|\\n| 1 | 2 |"},'
        '{"bbox_2d":[100,550,140,900],"type":"text","text_content":"R1"}]'
    )
    page_result = parse_gemini_response(raw, page_width=1000.0, page_height=1000.0)
    doc = pymupdf.open()
    doc.new_page(width=1000, height=1000)
    chunks = _chunks_from_ocr_page_result(
        doc, 0, page_result, image_bytes=b"", skip_text=False, include_figures=False
    )
    kinds = [c.page_content.split("\n")[0][:2] for c in chunks]
    assert kinds == ["L1", "L2", "| ", "R1"]


async def test_ocr_page_requires_vertex_project(monkeypatch):
    # No Developer-API fallback exists: an unconfigured provider must fail
    # loudly rather than silently route PHI pages to a non-BAA endpoint.
    monkeypatch.setattr(settings, "GOOGLE_VERTEX_PROJECT", None)
    provider = GeminiOcrProvider()
    with pytest.raises(RuntimeError, match="GOOGLE_VERTEX_PROJECT"):
        await provider.ocr_page(b"png", page_width=100, page_height=200)


class _StubGenai:
    """Records the kwargs `_make_client` constructs the client with."""

    def __init__(self):
        self.kwargs = None

    def Client(self, **kwargs):  # noqa: N802 - mirrors google.genai.Client
        self.kwargs = kwargs
        return object()


def test_make_client_is_vertex_only(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_VERTEX_PROJECT", "proj-1")
    monkeypatch.setattr(settings, "GOOGLE_VERTEX_LOCATION", "global")
    monkeypatch.setattr(settings, "GOOGLE_VERTEX_SA_JSON", None)
    stub = _StubGenai()
    GeminiOcrProvider()._make_client(stub)
    assert stub.kwargs == {
        "vertexai": True,
        "project": "proj-1",
        "location": "global",
        "credentials": None,
    }
