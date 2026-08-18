"""Document-level assembly (plan 032 §4 A7): merge gates, emission contract,
header dedup, cascades, furniture tolerance, and the no-op property."""
from __future__ import annotations

import pytest

from extract.core.assemble import (
    BoundaryDecision,
    assemble_chunks,
    decision_counters,
    leading_header_overlap,
    merge_table_chunks,
    render_table_markdown,
    table_view,
)
from extract.core.models import Chunk, ChunkType, TableCell

H = 792.0  # letter page height (points)
PAGES = [H, H, H]


def _cells(rows: list[list[str]]) -> list[TableCell]:
    return [TableCell(text=t, row=r, col=c)
            for r, row in enumerate(rows) for c, t in enumerate(row)]


def _table(page: int, bbox: list[float], rows: list[list[str]],
           confidence: float | None = None) -> Chunk:
    return Chunk(page_content=render_table_markdown(_cells(rows), len(rows), len(rows[0])),
                 page_no=page, bbox=bbox, chunk_type=ChunkType.TABLE,
                 cells=_cells(rows), n_rows=len(rows), n_cols=len(rows[0]),
                 confidence=confidence)


def _text(page: int, bbox: list[float], text: str) -> Chunk:
    return Chunk(page_content=text, page_no=page, bbox=bbox, chunk_type=ChunkType.TEXT)


# fragment A runs to the bottom of page 1; fragment B starts at the top of page 2
A = _table(1, [50, 400, 550, 780], [["Drug", "Dose"], ["Aspirin", "81 mg"]], confidence=90.0)
B = _table(2, [52, 30, 548, 200], [["Drug", "Dose"], ["Metformin", "500 mg"]], confidence=70.0)


def test_merge_happy_path_with_header_dedup():
    out, decisions = assemble_chunks([A, B], PAGES)
    assert len(out) == 1
    m = out[0]
    assert m.chunk_type == ChunkType.TABLE
    assert m.page_no == 1 and m.bbox == A.bbox            # first fragment wins
    assert m.merged_from_pages == [1, 2]
    assert m.n_rows == 3 and m.n_cols == 2                # B's repeated header dropped
    assert [d.merged for d in decisions] == [True]
    assert decisions[0].signals["header_rows_deduped"] == 1
    # GFM re-render is canonical and contains the re-indexed B row exactly once
    assert m.page_content == render_table_markdown(m.cells, 3, 2)
    assert m.page_content.count("Metformin") == 1 and m.page_content.count("Drug") == 1


def test_merge_provenance_cell_page_no_and_confidence_min():
    out, _ = assemble_chunks([A, B], PAGES)
    m = out[0]
    assert {c.page_no for c in m.cells if c.text in ("Drug", "Dose", "Aspirin")} == {1}
    assert all(c.page_no == 2 for c in m.cells if c.text == "Metformin")
    metformin = next(c for c in m.cells if c.text == "Metformin")
    assert metformin.row == 2                              # re-indexed past A's rows
    assert m.confidence == 70.0                            # min of parents (035 rule)


def test_no_header_repeat_appends_all_rows():
    b2 = _table(2, [52, 30, 548, 200], [["Lisinopril", "10 mg"], ["Atorva", "20 mg"]])
    out, decisions = assemble_chunks([A, b2], PAGES)
    assert len(out) == 1 and out[0].n_rows == 4
    assert decisions[0].signals["header_rows_deduped"] == 0


def test_geometry_gate_rejects_mid_page_tables():
    # two rows: header-only tables route to the continuation path instead
    far_a = _table(1, [50, 200, 550, 500], [["Drug", "Dose"], ["Aspirin", "81 mg"]])
    out, decisions = assemble_chunks([far_a, B], PAGES)
    assert len(out) == 2
    assert decisions[0].rejection == "geometry" and not decisions[0].geometry_candidate


def test_intervening_heading_rejects():
    # A ends at y=700 (inside the 12% bottom band, 697+); a section heading sits
    # below it at 705-720 — outside the 4% furniture band (760+) → real content
    # between the table and the page edge → reject.
    a2 = _table(1, [50, 300, 550, 700], [["Drug", "Dose"], ["Aspirin", "81 mg"]])
    heading = _text(1, [50, 705, 550, 720], "4. Discussion")
    out, decisions = assemble_chunks([a2, heading, B], [H, H])
    assert len(out) == 3
    assert decisions[0].rejection == "intervening_text_a"
    assert decisions[0].geometry_candidate  # rejected AFTER the geometry pre-gate


def test_footer_page_number_and_continued_are_tolerated():
    footer = _text(1, [250, 783, 350, 790], "12")                     # bottom band
    cont = _text(2, [60, 8, 250, 18], "Table 1 (continued)")          # top band
    out, decisions = assemble_chunks([A, footer, cont, B], PAGES)
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 1
    assert decisions[0].merged and decisions[0].continued_marker


def test_width_ratio_gate_rejects():
    narrow_b = _table(2, [52, 30, 280, 200], [["Drug", "Dose"], ["X", "1"]])
    out, decisions = assemble_chunks([A, narrow_b], PAGES)
    assert len(out) == 2 and decisions[0].rejection == "width_ratio"


def test_column_mismatch_gate_rejects():
    b3 = _table(2, [52, 30, 548, 200], [["a", "b", "c"], ["1", "2", "3"]])
    out, decisions = assemble_chunks([A, b3], PAGES)
    assert len(out) == 2 and decisions[0].rejection == "column_mismatch"


def test_three_page_cascade_merges_to_one():
    c = _table(3, [51, 25, 549, 180], [["Drug", "Dose"], ["Lipitor", "20 mg"]])
    b_tall = _table(2, [52, 30, 548, 780], [["Drug", "Dose"], ["Metformin", "500 mg"]])
    out, decisions = assemble_chunks([A, b_tall, c], PAGES)
    assert len(out) == 1
    m = out[0]
    assert m.merged_from_pages == [1, 2, 3]
    assert m.n_rows == 4                       # 2 + 1 + 1 (two headers deduped)
    assert [d.merged for d in decisions] == [True, True]
    assert {c_.page_no for c_ in m.cells} == {1, 2, 3}


def test_noop_single_page_and_no_tables():
    text_only = [_text(1, [10, 10, 100, 30], "hello"), _text(2, [10, 10, 100, 30], "world")]
    out, decisions = assemble_chunks(text_only, [H, H])
    assert out == text_only and decisions == []
    single = [_table(1, [50, 400, 550, 780], [["a"], ["b"]])]
    out, decisions = assemble_chunks(single, [H])
    assert out == single and decisions == []


def test_noop_preserves_object_identity_and_order():
    far_a = _table(1, [50, 100, 550, 300], [["x"]])
    far_b = _table(2, [50, 500, 550, 700], [["y"]])
    mid = _text(1, [10, 400, 100, 420], "para")
    inputs = [far_a, mid, far_b]
    out, decisions = assemble_chunks(inputs, [H, H])
    assert out == inputs and all(a is b for a, b in zip(out, inputs, strict=True))
    assert inputs[0].cells == far_a.cells      # input never mutated


def test_input_list_not_mutated_on_merge():
    inputs = [A, B]
    out, _ = assemble_chunks(inputs, PAGES)
    assert inputs == [A, B]                    # caller's list untouched
    assert A.merged_from_pages is None and B.merged_from_pages is None
    assert len(out) == 1


def test_decision_counters_shape():
    far_a = _table(1, [50, 200, 550, 500], [["x", "y"]])
    decisions = [
        BoundaryDecision(1, 2, merged=True, geometry_candidate=True),
        BoundaryDecision(2, 3, merged=False, geometry_candidate=True, rejection="width_ratio"),
        BoundaryDecision(3, 4, merged=False, geometry_candidate=False, rejection="geometry"),
    ]
    assert decision_counters(decisions) == {
        "pages_merge_candidate": 2, "tables_merged": 1, "merge_gate_rejections": 1}
    out, ds = assemble_chunks([far_a, B], PAGES)
    assert decision_counters(ds)["pages_merge_candidate"] == 0


def test_leading_header_overlap_multirow():
    a = _table(1, [0, 0, 100, 100], [["H1", "H2"], ["u1", "u2"], ["d", "e"]])
    b = _table(2, [0, 0, 100, 100], [["H1", "H2"], ["u1", "u2"], ["f", "g"]])
    assert leading_header_overlap(table_view(a), table_view(b)) == 2


def test_merge_table_chunks_direct_no_dedup():
    m = merge_table_chunks(A, B, header_rows=0)
    assert m.n_rows == 4 and m.page_content.count("Drug") == 2


def test_merge_carries_the_weaker_table_output_format():
    """A merged table holds BOTH fragments' cells, so if either side fell back to
    markdown-derived cells the merged table contains coarse table-level boxes and
    must say so. Dropping this let the table-format post-pass stamp the merged chunk
    "html", advertising repeated coarse boxes as true per-cell coordinates.
    """
    degraded = A.model_copy(update={"table_output_format": "markdown"})
    localized = B.model_copy(update={"table_output_format": "html"})
    assert merge_table_chunks(degraded, localized, header_rows=0).table_output_format == "markdown"
    assert merge_table_chunks(localized, degraded, header_rows=0).table_output_format == "markdown"
    # Both localized → the merged table really is cell_grid.
    assert merge_table_chunks(localized, localized, header_rows=0).table_output_format == "html"
    # Unset stays unset, so the table-format post-pass can still fill it in.
    assert merge_table_chunks(A, B, header_rows=0).table_output_format is None


@pytest.mark.parametrize(
    ("fmt_a", "fmt_b", "expected"),
    [
        (None, None, None),
        (None, "html", "html"),
        ("html", None, "html"),
        (None, "markdown", "markdown"),
        ("markdown", None, "markdown"),
        ("markdown", "markdown", "markdown"),
        ("markdown", "html", "markdown"),
        ("html", "markdown", "markdown"),
        ("html", "html", "html"),
    ],
)
def test_merged_format_every_pair(fmt_a, fmt_b, expected):
    """All nine combinations. markdown is absorbing (the merged table really does
    contain a coarse floor); None only survives when neither side was set."""
    a = A.model_copy(update={"table_output_format": fmt_a})
    b = B.model_copy(update={"table_output_format": fmt_b})
    assert merge_table_chunks(a, b, header_rows=0).table_output_format == expected


def test_merged_format_survives_a_transitive_chain():
    """Assembly folds pairwise, so a merged chunk can be merged again. One
    degraded fragment anywhere in a 3-page chain must still mark the result."""
    clean = B.model_copy(update={"table_output_format": "html"})
    degraded = A.model_copy(update={"table_output_format": "markdown"})
    # degraded fragment appears last in the fold
    first = merge_table_chunks(clean, clean, header_rows=0)
    assert merge_table_chunks(first, degraded, header_rows=0).table_output_format == "markdown"
    # ...and first in the fold
    first = merge_table_chunks(degraded, clean, header_rows=0)
    assert merge_table_chunks(first, clean, header_rows=0).table_output_format == "markdown"


# --------------------------------------------------------------------------- #
# GFM-text table fragments — the dominant live-path shape: the strict typed
# router rejects tables with multi-line cells, so they ship as TEXT chunks
# whose page_content IS the GFM (measured on prod parses 2026-06-11).
# --------------------------------------------------------------------------- #
GFM_A = ("| Database | Search strategy |\n"
         "|---|---|\n"
         "| Ovid MEDLINE | -(e-health or ehealth).tw\n"
         "-exp telemedicine/ |\n"
         "| Embase | -telerehabilitation.mp |")
GFM_B = ("| Database | Search strategy |\n"
         "|---|---|\n"
         "| CINAHL | -(mhealth or m-health).tw\n"
         "continued cell line |")


def _gfm_text(page: int, bbox: list[float], content: str) -> Chunk:
    return Chunk(page_content=content, page_no=page, bbox=bbox, chunk_type=ChunkType.TEXT)


TA = _gfm_text(1, [50, 400, 550, 780], GFM_A)
TB = _gfm_text(2, [52, 30, 548, 200], GFM_B)


def test_gfm_text_view_parses_multiline_cells():
    v = table_view(TA)
    assert v is not None and v.kind == "gfm_text" and v.n_cols == 2
    assert len(v.row_sigs) == 3                    # header + 2 data rows
    # the continuation line folded into the Ovid cell
    assert "exp telemedicine" in v.row_sigs[1][1][0]


def test_plain_prose_text_has_no_view():
    assert table_view(_text(1, [0, 0, 10, 10], "just a paragraph")) is None
    assert table_view(_text(1, [0, 0, 10, 10], "a | b\nno delimiter row")) is None


def test_gfm_text_merge_happy_path_with_header_dedup():
    out, decisions = assemble_chunks([TA, TB], PAGES)
    assert len(out) == 1
    m = out[0]
    assert m.chunk_type == ChunkType.TEXT
    assert m.merged_from_pages == [1, 2] and m.page_no == 1 and m.bbox == TA.bbox
    assert decisions[0].merged and decisions[0].signals["header_rows_deduped"] == 1
    # B's data spliced verbatim (multi-line cell intact), header + delimiter dropped
    assert m.page_content.count("| Database | Search strategy |") == 1
    assert m.page_content.count("|---|---|") == 1
    assert "CINAHL" in m.page_content and "continued cell line |" in m.page_content
    assert m.page_content.startswith(GFM_A)


def test_mixed_kind_boundary_merges_at_text_level():
    # typed A + gfm-text B: kind is parser noise (the strict router types small
    # clean fragments, leaves multi-line ones as text) → merge as TEXT
    out, decisions = assemble_chunks([A, TB], PAGES)
    assert len(out) == 1
    m = out[0]
    assert m.chunk_type == ChunkType.TEXT and m.merged_from_pages == [1, 2]
    assert decisions[0].merged
    assert decisions[0].signals["kind_a"] == "cells"
    assert decisions[0].signals["kind_b"] == "gfm_text"
    # A's canonical render heads the content; B's repeated header deduped
    assert m.page_content.startswith(A.page_content)
    assert m.page_content.count("| Drug | Dose |") == 1
    assert "CINAHL" in m.page_content


def test_mixed_kind_gfm_then_typed_merges():
    b_typed = _table(2, [52, 30, 548, 200],
                     [["Database", "Search strategy"], ["Embase", "telereh.mp"]])
    out, decisions = assemble_chunks([TA, b_typed], PAGES)
    assert len(out) == 1
    m = out[0]
    assert m.chunk_type == ChunkType.TEXT
    assert decisions[0].merged and decisions[0].signals["header_rows_deduped"] == 1
    assert m.page_content.startswith(GFM_A)
    assert m.page_content.rstrip().endswith("| Embase | telereh.mp |")


def test_paragraph_containing_continued_still_blocks():
    # a real paragraph (long, multi-word) containing "continued" is NOT furniture —
    # placed between the top furniture band (792*0.04=31.7) and the fragment top.
    b_low = _table(2, [52, 90, 548, 260], [["Drug", "Dose"], ["Metformin", "500 mg"]])
    para = _text(2, [52, 40, 548, 60],
                 "Patients continued their medication for six weeks after discharge "
                 "and were monitored for adverse events during the follow-up period.")
    out, decisions = assemble_chunks([A, para, b_low], PAGES)
    assert len(out) == 3
    assert decisions[0].rejection == "intervening_text_b"
    # whereas a SHORT "(continued)" caption in the same spot is tolerated
    caption = _text(2, [52, 40, 200, 60], "Table 1 (continued)")
    out, decisions = assemble_chunks([A, caption, b_low], PAGES)
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 1
    assert decisions[0].merged and decisions[0].continued_marker


def test_gfm_merge_no_repeated_header_appends_all_data_rows():
    b2 = _gfm_text(2, [52, 30, 548, 200],
                   "| PsycINFO | -burnout.tw |\n|---|---|\n| Scopus | -rtw.tw |")
    out, decisions = assemble_chunks([TA, b2], PAGES)
    assert len(out) == 1
    assert decisions[0].signals["header_rows_deduped"] == 0
    m = out[0]
    assert "PsycINFO" in m.page_content and "Scopus" in m.page_content
    assert m.page_content.count("|---|---|") == 1   # B's delimiter dropped


# --------------------------------------------------------------------------- #
# Header-only continuation path — a header-only table is the parser's tell
# that it failed to structure the rows (they ship as loose text below it) and
# the next page's table is a headerless continuation whose first row is DATA.
# Shape measured on Clyde referral packets (problem list spanning pages 3-4),
# 2026-07-31.
# --------------------------------------------------------------------------- #
# A: header sliver mid-page; its "rows" are loose text spans confined to its
# x-range, running into the bottom 12% band (>= 697).
HO_A = _table(1, [54, 470, 552, 485],
              [["Problem Description", "Onset Date", "Chronic", "Status", "Notes"]])
HO_ROWS = [
    _text(1, [54, 487, 214, 510], "Encounter for screening 09/11/2023"),
    _text(1, [231, 487, 239, 497], "N"),
    _text(1, [54, 622, 214, 632], "Hypercholesteremia 07/18/2019"),
    _text(1, [54, 700, 214, 738], "Vitamin D deficiency 01/31/2017"),
]
HO_FOOTER = _text(1, [54, 747, 376, 758], "Vargas 06/10/1962 Page: 2/3")   # furniture
# B: headerless continuation — narrower (empty trailing columns dropped),
# left-aligned, first row is data.
HL_B = _table(2, [53, 24, 238, 501],
              [["Acute pain of left knee", "03/06/2018", "N"],
               ["Atypical migraine", "05/14/2019", "N"]])


def test_headerless_continuation_merges_and_keeps_first_row():
    out, decisions = assemble_chunks([HO_A, *HO_ROWS, HO_FOOTER, HL_B], [H, H])
    tables = [c for c in out if c.chunk_type == ChunkType.TABLE]
    assert len(tables) == 1
    m = tables[0]
    d = decisions[0]
    assert d.merged and d.geometry_candidate
    assert d.signals["variant"] == "headerless_continuation"
    assert d.signals["header_rows_deduped"] == 0
    # B's first row is data, not a header — it must survive the merge
    assert "Acute pain of left knee" in m.page_content
    assert "Atypical migraine" in m.page_content
    assert m.n_rows == 3 and m.n_cols == 5          # 1 header + 2 data, A's columns
    assert m.merged_from_pages == [1, 2]
    # the loose row spans stay in place (folding them is a separate concern)
    assert sum(1 for c in out if c.chunk_type == ChunkType.TEXT) == len(HO_ROWS) + 1


def test_headerless_continuation_rejects_non_row_text_below():
    # a full-width paragraph below the sliver is NOT the orphaned-row shape
    para = _text(1, [54, 500, 550, 730],
                 "The patient was advised to follow up in three months and to "
                 "continue the current course of treatment as tolerated.")
    out, decisions = assemble_chunks([HO_A, para, HL_B], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_rows_below"


def test_headerless_continuation_rejects_when_rows_stop_mid_page():
    # row spans end at y=600 — nothing reaches the bottom band, so there is no
    # evidence the table runs off the page
    short_rows = [_text(1, [54, 487, 214, 510], "Encounter for screening 09/11/2023"),
                  _text(1, [54, 580, 214, 600], "Hypercholesteremia 07/18/2019")]
    out, decisions = assemble_chunks([HO_A, *short_rows, HL_B], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_geometry"


def test_headerless_continuation_rejects_misaligned_b():
    shifted = _table(2, [150, 24, 335, 501],
                     [["Acute pain of left knee", "03/06/2018", "N"]])
    out, decisions = assemble_chunks([HO_A, *HO_ROWS, shifted], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_alignment"


def test_headerless_continuation_rejects_more_columns_than_header():
    wide = _table(2, [53, 24, 550, 501],
                  [["a", "b", "c", "d", "e", "f"]] * 2)
    out, decisions = assemble_chunks([HO_A, *HO_ROWS, wide], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_columns"


def test_headerless_continuation_rejects_caption_above_b():
    # a caption above fragment 2 (below the furniture band) announces a NEW
    # table — the strongest evidence against a continuation (review finding 1)
    b_low = _table(2, [53, 80, 238, 501],
                   [["Acute pain of left knee", "03/06/2018", "N"]])
    caption = _text(2, [53, 60, 200, 75], "Table 3: Current Medications")
    out, decisions = assemble_chunks([HO_A, *HO_ROWS, HO_FOOTER, caption, b_low], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "intervening_text_b"


def test_headerless_continuation_vacuous_needs_corroboration():
    # A one-row table already in the bottom band with NOTHING below it carries
    # zero orphaned-row evidence; an unrelated half-width 2-col table opening
    # the next page must not fuse with it (review finding 2)
    a_bottom = _table(1, [50, 700, 550, 780], [["A", "B", "C", "D", "E"]])
    b_other = _table(2, [52, 30, 302, 200], [["x", "y"], ["1", "2"]])
    out, decisions = assemble_chunks([a_bottom, b_other], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_uncorroborated"


def test_headerless_continuation_rejects_table_width_paragraph_spans():
    # short spans that each fill the sliver's full width are paragraph lines,
    # not table rows — orphaned rows are narrow fragments (review finding 3)
    para_lines = [_text(1, [54, 500 + i * 50, 550, 530 + i * 50],
                        f"prose line {i} of a discussion section")
                  for i in range(5)]                    # reaches y=730 -> band
    out, decisions = assemble_chunks([HO_A, *para_lines, HL_B], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_rows_below"
    assert decisions[0].geometry_candidate          # near-miss stays visible


def test_headerless_continuation_gfm_text_b_merges_at_text_level():
    # the dominant live-path shape: B ships as a TEXT chunk whose content IS
    # the GFM; its first line is data and must survive, its delimiter dropped
    gfm_b = _gfm_text(2, [53, 24, 238, 501],
                      "| Acute pain of left knee | 03/06/2018 | N |\n"
                      "|---|---|---|\n"
                      "| Atypical migraine | 05/14/2019 | N |")
    out, decisions = assemble_chunks([HO_A, *HO_ROWS, HO_FOOTER, gfm_b], [H, H])
    assert decisions[0].merged
    m = next(c for c in out if c.merged_from_pages)
    assert m.chunk_type == ChunkType.TEXT               # mixed kind -> text level
    assert "Acute pain of left knee" in m.page_content
    assert "Atypical migraine" in m.page_content
    assert m.page_content.count("|---|---|---|") == 0   # B's delimiter dropped


def test_headerless_continuation_gfm_header_only_a():
    gfm_a = _gfm_text(1, [54, 470, 552, 485],
                      "| Problem Description | Onset Date | Chronic | Status | Notes |\n"
                      "| --- | --- | --- | --- | --- |")
    out, decisions = assemble_chunks([gfm_a, *HO_ROWS, HO_FOOTER, HL_B], [H, H])
    assert decisions[0].merged
    assert decisions[0].signals["variant"] == "headerless_continuation"
    m = next(c for c in out if c.merged_from_pages)
    assert "Acute pain of left knee" in m.page_content


def test_headerless_continuation_bboxless_b_conservatively_rejects():
    b_no_bbox = Chunk(page_content=HL_B.page_content, page_no=2, bbox=None,
                      chunk_type=ChunkType.TABLE, cells=HL_B.cells,
                      n_rows=HL_B.n_rows, n_cols=HL_B.n_cols)
    out, decisions = assemble_chunks([HO_A, *HO_ROWS, b_no_bbox], [H, H])
    assert sum(1 for c in out if c.chunk_type == ChunkType.TABLE) == 2
    assert decisions[0].rejection == "continuation_alignment"


def test_header_only_near_bottom_with_repeated_header_dedups():
    # A header-only table already inside the bottom band: the rows-below check
    # is vacuous, and a B that repeats the header dedups it as usual.
    a_bottom = _table(1, [50, 700, 550, 780], [["Drug", "Dose"]])
    b_rep = _table(2, [52, 30, 548, 200], [["Drug", "Dose"], ["Metformin", "500 mg"]])
    out, decisions = assemble_chunks([a_bottom, b_rep], [H, H])
    tables = [c for c in out if c.chunk_type == ChunkType.TABLE]
    assert len(tables) == 1
    assert decisions[0].merged
    assert decisions[0].signals["variant"] == "headerless_continuation"
    assert decisions[0].signals["header_rows_deduped"] == 1
    assert tables[0].page_content.count("Drug") == 1
