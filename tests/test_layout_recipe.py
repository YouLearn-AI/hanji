"""Unit tests for the per-org LAYOUT extraction recipe (core/layout_recipe.py).

Pure functions only — no model calls. Covers the chunk annotation, the per-org gate,
and that the layout prompt is actually wired (not the old stub).
"""

from __future__ import annotations

from extract.config import settings
from extract.core.layout_recipe import (
    LAYOUT_INSTRUCTIONS,
    layout_annotate,
    should_use_layout_recipe,
)
from extract.core.schema_extract import _IndexedChunk


def test_layout_annotate_adds_section_and_coords():
    # bbox is 0–1000 page-relative → center/10 = percent. Section heading on line 1.
    c = _IndexedChunk(text="Patient Information", page=1, bbox=[100, 200, 300, 400])
    out = layout_annotate(0, c)
    assert out == "[SECTION] (x20,y30) Patient Information"
    assert c.bbox == [100, 200, 300, 400]  # geometry untouched; only text rewritten


def test_layout_annotate_puts_prefix_on_own_line_for_multiline_chunks():
    """Since the P0-3 fix, chunk text keeps its newlines so a table's header row
    reaches the model intact — the annotation must not be glued onto it."""
    c = _IndexedChunk(text="| Payer | ID |\n|---|---|\n| MEDICARE | 1EG4 |", page=1,
                      bbox=[100, 200, 300, 400])
    out = layout_annotate(0, c)
    assert out == "(x20,y30)\n| Payer | ID |\n|---|---|\n| MEDICARE | 1EG4 |"


def test_layout_annotate_no_bbox_omits_coords():
    # The pre-parsed chunks route sends page_sizes=[] → bbox None → no coord prefix.
    c = _IndexedChunk(text="Home Phone: 555-1212", page=1, bbox=None)
    assert layout_annotate(0, c) == "Home Phone: 555-1212"


def test_layout_annotate_coords_without_section():
    c = _IndexedChunk(text="just a value", page=1, bbox=[0, 0, 100, 100])
    out = layout_annotate(0, c)
    assert out.startswith("(x5,y5) ")
    assert "[SECTION]" not in out


def test_should_use_layout_recipe_toggle(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACT_LAYOUT_RECIPE_ENABLED", True)
    assert should_use_layout_recipe() is True
    monkeypatch.setattr(settings, "EXTRACT_LAYOUT_RECIPE_ENABLED", False)
    assert should_use_layout_recipe() is False


def test_layout_instructions_wired_not_stub():
    assert LAYOUT_INSTRUCTIONS is not None
    assert "FAMILY, GIVEN" in LAYOUT_INSTRUCTIONS
