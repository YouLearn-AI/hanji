"""Plan 077 — semantic RAG chunking: grouper, table split, serialization.

Pure-function tests over synthetic chunk layouts (geometry in PDF points on a
612x792 page) plus the serializer byte-compat contract. The real-document
integration pass lives in tests/test_chunking_integration.py.
"""

from __future__ import annotations

import copy

import orjson
import pydantic
import pytest

from extract.core.assemble import render_table_markdown
from extract.core.chunking import (
    BAND_HIGH,
    BAND_LOW,
    build_segments,
    page_dimensions,
    render_document_content,
    telemetry_counters,
)
from extract.core.models import (
    Chunk,
    ChunkType,
    ExtractRequest,
    ExtractResponse,
    TableCell,
)

# --------------------------------------------------------------------------- #
# layout builders
# --------------------------------------------------------------------------- #


def text(content: str, page: int, y: float, *, x: float = 72, h: float = 12, w: float = 430) -> Chunk:
    return Chunk(
        page_content=content,
        page_no=page,
        bbox=[x, y, x + w, y + h],
        chunk_type=ChunkType.TEXT,
    )


def paragraph(page: int, y: float, *, sentences: int = 10, h: float = 48) -> Chunk:
    return text("The quick brown fox jumps over the lazy dog. " * sentences, page, y, h=h)


def make_table(page: int, y: float, *, body_rows: int, row_text: str = "value") -> Chunk:
    cells = [TableCell(text="Code", row=0, col=0), TableCell(text="Amount", row=0, col=1)]
    for r in range(1, body_rows + 1):
        cells.append(TableCell(text=f"J{r:04d}", row=r, col=0))
        cells.append(TableCell(text=f"{row_text} line item number {r}", row=r, col=1))
    n_rows = body_rows + 1
    md = render_table_markdown(cells, n_rows, 2)
    return Chunk(
        page_content=md,
        page_no=page,
        bbox=[36, y, 570, y + 12 * n_rows],
        chunk_type=ChunkType.TABLE,
        cells=cells,
        n_rows=n_rows,
        n_cols=2,
    )


def kv(page: int, y: float) -> Chunk:
    return Chunk(
        page_content="Name: Jane\nDOB: 1980-01-01\n[x] Option A\n[ ] Option B",
        page_no=page,
        bbox=[72, y, 300, y + 60],
        chunk_type=ChunkType.KEY_VALUE,
    )


def figure(page: int, y: float, *, url: str | None = None) -> Chunk:
    return Chunk(
        page_content="",
        page_no=page,
        bbox=[72, y, 300, y + 180],
        chunk_type=ChunkType.IMAGE,
        image_b64=None if url else "aGk=",
        image_url=url,
    )


def assert_invariants(chunks: list[Chunk], segments, *, chunk_size: int) -> None:
    """The plan §3 partition invariants, checked on every layout."""
    seen: list[int] = []
    for s in segments:
        assert s.char_count == len(s.content)
        assert s.pages == sorted({m.page_no for m in s.chunks})
        for m in s.chunks:
            seen.append(m.source_index)
    # every flat index appears in exactly one segment (split parts repeat the
    # table's index once per part — dedupe those before comparing)
    deduped = sorted(set(seen))
    assert deduped == list(range(len(chunks)))
    non_part = [
        m.source_index for s in segments for m in s.chunks if m.table_part is None
    ]
    assert len(non_part) == len(set(non_part)), "an unsplit element appeared twice"


# --------------------------------------------------------------------------- #
# grouper behavior
# --------------------------------------------------------------------------- #


def test_band_adherence_on_plain_prose():
    chunks = [paragraph(1, 60 + i * 60, sentences=4, h=36) for i in range(12)]
    segments = build_segments(chunks, chunk_size=1000)
    assert_invariants(chunks, segments, chunk_size=1000)
    lo, hi = round(1000 * BAND_LOW), round(1000 * BAND_HIGH)
    # everything but a trailing remainder must be inside the band
    for s in segments[:-1]:
        assert lo <= s.char_count <= hi, s.char_count
    assert segments[-1].char_count <= hi


def test_heading_starts_new_segment_and_glues_forward():
    chunks = [
        paragraph(1, 60, sentences=6),
        text("SECTION TWO", 1, 130, h=16),
        paragraph(1, 160, sentences=6),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    assert_invariants(chunks, segments, chunk_size=1000)
    starts = [s.chunks[0].source_index for s in segments]
    assert 1 in starts, "heading did not start a segment"
    for s in segments:
        members = [m.source_index for m in s.chunks]
        if 1 in members:
            assert members[-1] != 1, "heading left dangling at segment end"
            assert 2 in members, "heading separated from its content"


def test_figure_caption_glue():
    chunks = [
        paragraph(1, 60, sentences=6),
        figure(1, 130),
        text("Figure 1: revenue by quarter", 1, 316, h=12, w=200),
        paragraph(1, 350, sentences=6),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    assert_invariants(chunks, segments, chunk_size=1000)
    for s in segments:
        members = [m.source_index for m in s.chunks]
        if 1 in members:
            assert 2 in members, "caption separated from figure"
            # figures contribute nothing to content (no placeholder, no bytes);
            # the member record carries the grounding
            assert "figure](" not in s.content and "[figure]" not in s.content
            assert "aGk=" not in s.content
            assert s.content.startswith("Figure 1:") or "Figure 1:" in s.content


def test_list_run_stays_together():
    items = [text(f"- item number {i} with some detail", 1, 100 + i * 16, h=12) for i in range(6)]
    chunks = [paragraph(1, 40, sentences=4), *items]
    segments = build_segments(chunks, chunk_size=1000)
    assert_invariants(chunks, segments, chunk_size=1000)
    list_segments = {
        s.chunks[0].source_index
        for s in segments
        if any(m.source_index in range(1, 7) for m in s.chunks)
    }
    assert len(list_segments) == 1, "list run split across segments"


def test_kv_region_is_atomic_even_when_oversized():
    big_kv = Chunk(
        page_content="\n".join(f"Field {i}: value {i}" for i in range(200)),
        page_no=1,
        bbox=[72, 60, 540, 700],
        chunk_type=ChunkType.KEY_VALUE,
    )
    segments = build_segments([big_kv], chunk_size=1000)
    assert len(segments) == 1
    assert segments[0].char_count > round(1000 * BAND_HIGH)  # oversized, intact
    assert segments[0].chunks[0].table_part is None


def test_no_merge_across_page_boundary():
    chunks = [text("short tail", 1, 700, h=12), text("next page start", 2, 60, h=12)]
    segments = build_segments(chunks, chunk_size=1000)
    assert len(segments) == 2, "merged across a page boundary"


def test_bidirectional_rescue_prefers_valid_neighbor():
    # [~800, ~300, ~1000] with band 750-1250: the 300 must merge backward into
    # the 800 (forward merge with 1000 would exceed max).
    a = text("a" * 798, 1, 60, h=40)
    b = text("b" * 300, 1, 110, h=20)
    c = text("c" * 1000, 1, 140, h=52)
    segments = build_segments([a, b, c], chunk_size=1000)
    sizes = [s.char_count for s in segments]
    assert len(segments) == 2, sizes
    assert {m.source_index for m in segments[0].chunks} == {0, 1}


def test_segments_preserve_served_chunk_order():
    # 2026-07-30 reading-order fix, extending 7ccdaa202 to the RAG surface:
    # chunks[] already ships the model's emission order (column-aware, tables in
    # place), so segments/content must ship the SAME order. A 2-column page
    # emitted left-column-first with a mid-page table is exactly what a
    # bbox-center (y, x) re-sort destroys — it interleaves the columns and
    # displaces the table. Replayed over 1,392 saved champion decodes, the old
    # sort reordered 57.0% of pages (mean NED 0.0637 vs the served order).
    chunks = [
        text("L1 first left line", 1, 100, x=50, w=240),
        text("L2 second left line", 1, 150, x=50, w=240),
        make_table(1, 200, body_rows=2),
        text("L3 third left line", 1, 340, x=50, w=240),
        text("R1 first right line", 1, 100, x=320, w=240),
        text("R2 second right line", 1, 150, x=320, w=240),
        text("R3 third right line", 1, 200, x=320, w=240),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    members = [m.source_index for s in segments for m in s.chunks]
    assert members == list(range(len(chunks))), "segments re-ordered the served chunks"
    # ``content`` is the same surface and must agree with chunks[] too.
    content = render_document_content(chunks)
    at = [content.index(c.page_content) for c in chunks]
    assert at == sorted(at), "include_content re-ordered the served chunks"


def test_legacy_geometric_resort_is_gone():
    # Historical: until 2026-07-30 phase 0 re-derived order from bbox centers,
    # so a table listed AFTER the prose below it was pulled back between the two
    # prose chunks ("reading order rebuilt from type-bucketed input" — the flat
    # list used to arrive type-bucketed, prose → KV → tables → figures). Page
    # assembly now interleaves by ``seq``, so list order IS reading order and a
    # caller-supplied order is honoured verbatim.
    prose_above = text("above the table", 1, 60, h=12)
    prose_below = text("below the table", 1, 400, h=12)
    table = make_table(1, 100, body_rows=3)
    segments = build_segments([prose_above, prose_below, table], chunk_size=1000)
    members = [m.source_index for s in segments for m in s.chunks]
    assert members == [0, 1, 2], "geometric re-sort still applied"


def test_boxless_chunk_keeps_its_emitted_position():
    # Boxless chunks used to sink to the page end (bbox → +inf in the sort key),
    # a second order divergence between chunks[] and the RAG surface. With the
    # sort gone they stay where the model emitted them.
    chunks = [
        Chunk(page_content="BOXLESS-CRITICAL-VALUE", page_no=1, chunk_type=ChunkType.TEXT),
        text("second", 1, 100),
        text("third", 1, 200),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    members = [m.source_index for s in segments for m in s.chunks]
    assert members == [0, 1, 2]


def test_determinism():
    chunks = [
        paragraph(1, 60),
        text("HEADING", 1, 130, h=16),
        make_table(1, 160, body_rows=4),
        figure(2, 60),
        text("Figure 2: things", 2, 246, h=12, w=180),
    ]
    a = build_segments(chunks, chunk_size=1000)
    b = build_segments(chunks, chunk_size=1000)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_empty_and_single_element():
    assert build_segments([], chunk_size=1000) == []
    only = [paragraph(1, 60, sentences=3)]
    segments = build_segments(only, chunk_size=1000)
    assert len(segments) == 1
    assert segments[0].chunks[0].source_index == 0


# --------------------------------------------------------------------------- #
# table row-split (§4.1)
# --------------------------------------------------------------------------- #


def split_members(segments):
    return [m for s in segments for m in s.chunks if m.table_part is not None]


def test_oversized_table_splits_into_valid_parts():
    table = make_table(1, 60, body_rows=40, row_text="Payment adjustment detail")
    original_md = table.page_content
    segments = build_segments([table], chunk_size=1000)
    parts = split_members(segments)
    assert len(parts) >= 2
    header, delim = original_md.split("\n")[:2]
    reassembled: list[str] = []
    for m in sorted(parts, key=lambda m: m.table_part.index):
        lines = m.page_content.split("\n")
        assert lines[0] == header and lines[1] == delim, "part is not a valid table"
        assert m.n_rows == len(lines) - 1
        assert m.n_cols == 2
        assert len(m.page_content) <= round(1000 * BAND_HIGH)
        assert m.source_index == 0
        reassembled.extend(lines[2:])
    assert reassembled == original_md.split("\n")[2:], "body rows not conserved"
    counts = {m.table_part.count for m in parts}
    assert counts == {len(parts)}
    # row ranges tile the body exactly
    spans = sorted((m.table_part.row_start, m.table_part.row_end) for m in parts)
    assert spans[0][0] == 0
    for (_s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert s2 == e1 + 1


def test_small_table_never_splits():
    table = make_table(1, 60, body_rows=4)
    segments = build_segments([table], chunk_size=1000)
    assert split_members(segments) == []


def test_table_fitting_in_band_starts_new_segment_instead_of_splitting():
    prose = paragraph(1, 60, sentences=14)  # ~640 chars
    table = make_table(1, 130, body_rows=18)  # ~700 chars rendered, fits alone
    segments = build_segments([prose, table], chunk_size=1000)
    assert split_members(segments) == [], "in-band table was split"
    table_seg = next(s for s in segments if any(m.chunk_type == ChunkType.TABLE for m in s.chunks))
    assert len(table.page_content) <= round(1000 * BAND_HIGH)
    assert table_seg.char_count <= round(1000 * BAND_HIGH)


def test_rowspan_group_moves_as_unit():
    cells = [TableCell(text="H1", row=0, col=0), TableCell(text="H2", row=0, col=1)]
    r = 1
    while r < 30:
        # rows r and r+1 joined by a rowspan cell in col 0
        cells.append(TableCell(text=f"span{r}", row=r, col=0, row_span=2))
        cells.append(TableCell(text=f"detail {r} with quite a lot of padding text here", row=r, col=1))
        cells.append(TableCell(text=f"detail {r + 1} with quite a lot of padding text here", row=r + 1, col=1))
        r += 2
    n_rows = r
    md = render_table_markdown(cells, n_rows, 2)
    table = Chunk(
        page_content=md, page_no=1, bbox=[36, 60, 570, 700], chunk_type=ChunkType.TABLE,
        cells=cells, n_rows=n_rows, n_cols=2,
    )
    segments = build_segments([table], chunk_size=600)
    parts = split_members(segments)
    assert len(parts) >= 2
    for m in parts:
        # every part must start on an odd (span-start) source row
        assert m.table_part.row_start % 2 == 0  # body-row coords: spans start at even
        span_rows = m.table_part.row_end - m.table_part.row_start + 1
        assert span_rows % 2 == 0, "rowspan pair separated"


def test_merged_crosspage_table_splits_page_local():
    cells = [TableCell(text="Code", row=0, col=0), TableCell(text="Amount", row=0, col=1)]
    for r in range(1, 25):
        page = 1 if r <= 12 else 2
        cells.append(TableCell(text=f"J{r:04d}", row=r, col=0, page_no=page))
        cells.append(
            TableCell(text=f"Adjustment line item number {r} with detail", row=r, col=1, page_no=page)
        )
    n_rows = 25
    md = render_table_markdown(cells, n_rows, 2)
    merged = Chunk(
        page_content=md, page_no=1, bbox=[36, 300, 570, 700], chunk_type=ChunkType.TABLE,
        cells=cells, n_rows=n_rows, n_cols=2, merged_from_pages=[1, 2],
    )
    segments = build_segments([merged], chunk_size=1000)
    parts = split_members(segments)
    assert len(parts) >= 2
    for m in parts:
        rows = range(m.table_part.row_start + 1, m.table_part.row_end + 2)  # body→abs
        pages = {1 if r <= 12 else 2 for r in rows}
        assert len(pages) == 1, "part spans pages"
        assert m.page_no in pages


def test_headerless_table_splits_without_repeating_data():
    # GFM forces a header slot, so a visually headerless table arrives with a
    # DATA row in row 0. Splitting must not duplicate that row into parts.
    cells = []
    for r in range(40):
        cells.append(TableCell(text=f"J{r + 1:04d}", row=r, col=0))
        cells.append(TableCell(text=f"${r * 13.7 + 40:,.2f}", row=r, col=1))
        cells.append(TableCell(text=f"Professional service line item {r + 1}, extended consult", row=r, col=2))
    md = render_table_markdown(cells, 40, 3)
    table = Chunk(
        page_content=md, page_no=1, bbox=[36, 60, 570, 700], chunk_type=ChunkType.TABLE,
        cells=cells, n_rows=40, n_cols=3,
    )
    segments = build_segments([table], chunk_size=1000)
    parts = sorted(split_members(segments), key=lambda m: m.table_part.index)
    assert len(parts) >= 2
    original_rows = [md.split("\n")[0]] + md.split("\n")[2:]
    reassembled: list[str] = []
    for m in parts:
        lines = m.page_content.split("\n")
        assert lines[1].startswith("| ---"), "part is not a valid table"
        reassembled.extend([lines[0]] + lines[2:])  # header slot holds real data
    assert reassembled == original_rows, "rows duplicated or lost"
    # no row text appears in more than one part
    firsts = [m.page_content.split("\n")[0] for m in parts]
    assert len(firsts) == len(set(firsts)), "a data row was repeated as a header"


def test_gfm_text_table_without_cells_stays_atomic():
    md = "\n".join(["| A | B |", "| --- | --- |"] + [f"| row {i} | {'x' * 60} |" for i in range(40)])
    chunk = Chunk(page_content=md, page_no=1, bbox=[36, 60, 570, 700], chunk_type=ChunkType.TEXT)
    segments = build_segments([chunk], chunk_size=1000)
    assert len(segments) == 1
    assert segments[0].chunks[0].table_part is None  # oversized, intact


# --------------------------------------------------------------------------- #
# request validation + response serialization contract
# --------------------------------------------------------------------------- #


def test_request_validation_gated_on_enable():
    assert ExtractRequest(chunking="semantic", chunk_size=200).chunk_size == 200
    assert ExtractRequest(chunking="semantic", chunk_size=8000).chunk_size == 8000
    with pytest.raises(pydantic.ValidationError):
        ExtractRequest(chunking="semantic", chunk_size=50)
    with pytest.raises(pydantic.ValidationError):
        ExtractRequest(chunking="semantic", chunk_size=100000)
    # disabled: out-of-range values are ignored, never rejected (back-compat)
    assert ExtractRequest(chunking="none", chunk_size=50).chunk_size == 1000
    assert ExtractRequest(chunk_size=50).chunk_size == 1000
    # unknown fields still accepted-and-ignored
    assert ExtractRequest(some_future_option=True).chunking == "none"


def test_default_response_serialization_is_byte_identical():
    chunks = [paragraph(1, 60), make_table(1, 130, body_rows=3), figure(1, 300)]
    response = ExtractResponse(chunks=chunks)
    dumped = response.model_dump(mode="json")
    for key in ("segments", "page_dimensions", "pdf_rendition_url"):
        assert key not in dumped
    # the shape every serialization surface produces (sync route, batch
    # worker, CLI all call model_dump(mode="json")):
    baseline = orjson.dumps({"chunks": [c.model_dump(mode="json") for c in chunks]})
    assert orjson.dumps(dumped) == baseline


# --------------------------------------------------------------------------- #
# The additive contract (ExtractRequest.chunking docs): "'none' (default):
# response unchanged. 'semantic': ADDITIONALLY return `segments`". `chunks` must
# be byte-identical either way. Pinned after a 2026-08-07 production investigation,
# where a 4/23-vs-0/20 correlation was read as the chunker REMOVING chunks; the
# real cause was model-side decode variance (the raw endpoint loses titles 3/24
# at concurrency 1 with no chunker involved). No src/ change was warranted — the
# missing piece was a test, which is why a false belief could stand for hours.
# The geometry below is the shape that was suspected: a short caption whose bbox
# lies INSIDE the bbox of the table it introduces.
# --------------------------------------------------------------------------- #
def _caption_inside_table_page() -> list[Chunk]:
    chunks: list[Chunk] = []
    y = 100.0
    for i in range(1, 6):
        chunks.append(
            text(f"Plan {i}: Choice+ Primary Advantage (2025-2026)", 1, y, h=6.34, w=340)
        )
        chunks.append(make_table(1, y - 2.38, body_rows=4))  # box opens ABOVE the caption
        y += 90.0
    return chunks


@pytest.mark.parametrize("chunk_size", [400, 1000, 2000])
def test_build_segments_never_mutates_its_input(chunk_size):
    chunks = _caption_inside_table_page()
    before = copy.deepcopy(chunks)
    build_segments(chunks, chunk_size=chunk_size)
    assert chunks == before


@pytest.mark.parametrize("chunk_size", [400, 1000, 2000])
def test_chunks_are_byte_identical_with_and_without_semantic_chunking(chunk_size):
    chunks = _caption_inside_table_page()
    plain = ExtractResponse(chunks=chunks).model_dump(mode="json")
    semantic = ExtractResponse(
        chunks=chunks,
        segments=build_segments(chunks, chunk_size=chunk_size),
        page_dimensions=page_dimensions([(612.0, 792.0)]),
    ).model_dump(mode="json")
    assert orjson.dumps(semantic["chunks"]) == orjson.dumps(plain["chunks"])
    assert "segments" not in plain
    assert semantic["segments"]


def test_every_chunk_is_reachable_from_some_segment():
    chunks = _caption_inside_table_page()
    segments = build_segments(chunks, chunk_size=1000)
    covered = {m.source_index for s in segments for m in s.chunks}
    assert covered == set(range(len(chunks)))
    joined = "\n".join(s.content for s in segments)
    for i in range(1, 6):
        assert f"Plan {i}: Choice+ Primary Advantage (2025-2026)" in joined


def test_enabled_response_carries_chunking_fields():
    chunks = [paragraph(1, 60)]
    segments = build_segments(chunks, chunk_size=1000)
    response = ExtractResponse(chunks=chunks, segments=segments, page_dimensions=None)
    dumped = response.model_dump(mode="json")
    assert "chunking_version" not in dumped  # never exposed (owner ruling)
    assert len(dumped["segments"]) == len(segments)
    member = dumped["segments"][0]["chunks"][0]
    assert set(member) >= {"source_index", "chunk_type", "page_no", "bbox"}


def test_telemetry_counters_shape():
    chunks = [paragraph(1, 60), make_table(2, 60, body_rows=40)]
    segments = build_segments(chunks, chunk_size=1000)
    counters = telemetry_counters(segments, chunk_size=1000)
    assert counters["segment_count"] == len(segments)
    assert counters["segment_tables_split"] == 1
    assert 0.0 <= counters["segments_in_band_ratio"] <= 1.0


# --------------------------------------------------------------------------- #
# chunker v2 (production fixes, 2026-07-16)
# --------------------------------------------------------------------------- #


def test_oversized_multiline_prose_splits_at_lines():
    # One dense 40-line block (~1900 chars) — the rajmeet case. Parts must fit
    # the band, tile the source text via char offsets, and share the bbox.
    lines = [f"600000 VENDOR {i:02d} ACME SUPPLY 2026 100 INV P {i * 13}.55" for i in range(40)]
    block = Chunk(page_content="\n".join(lines), page_no=1, bbox=[36, 60, 570, 700])
    segments = build_segments([block], chunk_size=1000)
    assert len(segments) >= 2
    hi = round(1000 * BAND_HIGH)
    source = block.page_content
    reassembled: list[str] = []
    parts = [m for s in segments for m in s.chunks]
    for m in parts:
        assert m.text_part is not None
        assert m.page_content is None  # nothing duplicated: recover by slice
        assert m.bbox == block.bbox
        piece = source[m.text_part.char_start : m.text_part.char_end]
        assert len(piece) <= hi
        reassembled.append(piece)
    assert "\n".join(reassembled) == source, "char offsets do not tile the source"
    counts = {m.text_part.count for m in parts}
    assert counts == {len(parts)}
    for a, b in zip(parts, parts[1:], strict=False):
        assert b.text_part.char_start == a.text_part.char_end + 1
    # segment content must rejoin sibling parts with \n, not a blank line
    for s in segments:
        assert "\n\n" not in s.content


def test_oversized_single_line_prose_stays_atomic():
    block = Chunk(page_content="x" * 2000, page_no=1, bbox=[36, 60, 570, 100])
    segments = build_segments([block], chunk_size=1000)
    assert len(segments) == 1
    assert segments[0].chunks[0].text_part is None


def test_gfm_shaped_text_never_line_splits():
    md = "\n".join(["| A | B |", "| --- | --- |"] + [f"| row {i} | {'x' * 60} |" for i in range(40)])
    chunk = Chunk(page_content=md, page_no=1, bbox=[36, 60, 570, 700], chunk_type=ChunkType.TEXT)
    segments = build_segments([chunk], chunk_size=1000)
    assert len(segments) == 1
    assert segments[0].chunks[0].text_part is None


def test_tiny_fragment_rescued_across_weak_heading():
    # The achyut case: caps-at-body-size form VALUES classify as (weak)
    # headings and used to trap a ~50-char fragment between them.
    chunks = [
        paragraph(1, 60, sentences=6),
        text("MEDICARE PART A AND B", 1, 130, h=12),
        text("1KK2U33QN44", 1, 146, h=12),
        text("Secondary:", 1, 162, h=12),
        text("AARP MEDICARE", 1, 190, h=12),
        paragraph(1, 210, sentences=6),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    for s in segments:
        assert s.char_count == 0 or s.char_count >= 100, (
            f"tiny fragment shipped: {s.char_count}ch {[m.source_index for m in s.chunks]}"
        )


def test_strong_heading_still_blocks_small_section_merge():
    # A short but REAL section under a numbered heading stays its own segment.
    chunks = [
        paragraph(1, 60, sentences=8),
        text("5 Training", 1, 170, h=16),
        text("This section describes the training regime for our models and their data.", 1, 200, h=12),
        text("6 Results", 1, 240, h=16),
        paragraph(1, 270, sentences=8),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    training = next(
        s for s in segments if any(m.source_index == 1 for m in s.chunks)
    )
    members = {m.source_index for m in training.chunks}
    assert members == {1, 2}, "small real section merged across a strong heading"


def test_figure_only_segment_attaches_same_page():
    # The rajmeet 0-char case: a full-page image before the page's title.
    chunks = [
        figure(1, 40),
        text("EXAMPLE COUNTY REPORT", 1, 260, h=16),
        paragraph(1, 290, sentences=6),
    ]
    segments = build_segments(chunks, chunk_size=1000)
    assert all(s.char_count > 0 for s in segments), "empty segment shipped"
    seen = [m.source_index for s in segments for m in s.chunks]
    assert sorted(seen) == [0, 1, 2]


def test_lone_figure_on_own_page_stays_honest():
    chunks = [figure(1, 40), paragraph(2, 60, sentences=6)]
    segments = build_segments(chunks, chunk_size=1000)
    # cross-page attachment is refused; the figure stays a standalone segment
    fig_seg = next(s for s in segments if any(m.chunk_type == ChunkType.IMAGE for m in s.chunks))
    assert fig_seg.char_count == 0 and fig_seg.pages == [1]


def test_oversized_single_line_paragraph_splits_at_sentences():
    # v4: a mega-paragraph with no newlines splits at sentence boundaries.
    sentence = "The committee reviewed the quarterly performance of every region. "
    source = (sentence * 30).strip()  # ~2000 chars, one line
    block = Chunk(page_content=source, page_no=1, bbox=[36, 60, 570, 200])
    segments = build_segments([block], chunk_size=1000)
    parts = [m for s in segments for m in s.chunks]
    assert len(parts) >= 2
    hi = round(1000 * BAND_HIGH)
    for m in parts:
        assert m.text_part is not None
        piece = source[m.text_part.char_start : m.text_part.char_end]
        assert 0 < len(piece) <= hi
        # every part starts at a sentence start, never mid-word
        assert piece[0].isupper()
    for a, b in zip(parts, parts[1:], strict=False):
        gap = source[a.text_part.char_end : b.text_part.char_start]
        assert gap.strip() == "", "non-whitespace lost between parts"


def test_single_sentence_over_budget_stays_whole():
    source = "A single unbroken sentence " + "with many many clauses " * 80 + "ends here."
    block = Chunk(page_content=source, page_no=1, bbox=[36, 60, 570, 120])
    segments = build_segments([block], chunk_size=1000)
    assert len(segments) == 1
    assert segments[0].chunks[0].text_part is None  # atomic: no sentence cut exists
