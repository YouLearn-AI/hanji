"""Plan 077 — opt-in semantic RAG chunking (``chunking="semantic"``).

Groups the flat element list into size-banded, embed-ready ``Segment``s:

* Phase 0 takes the reading-order view the pipeline already produced: the flat
  list arrives in the provider's own emission order (page assembly interleaves
  prose/KV/tables/figures by ``seq``, ``pdf._chunks_from_ocr_page_result``), so
  adjacency reasoning reads it as-is, grouped by page. It used to re-sort each
  page by bbox-center (y, x); that undid the model's column-aware order — see
  ``_ordered_elements``.
* Phase 1 glues runs that must not be separated: heading → following element,
  figure → caption, consecutive list items, KV regions atomic.
* Phase 2 breaks the stream at structural boundaries (page break, heading
  start, large vertical gap).
* Phase 3 splits oversized groups at their weakest internal boundary; the one
  intra-element split is an oversized table, split at body-row boundaries with
  the header repeated so every part is a valid table (§4.1 of the plan).
* Phase 4 merges undersized groups into their lowest-cost neighbor, never
  across a page or heading boundary.

Everything is a pure function of the input chunks and options — no I/O, no
randomness. All size decisions are made on the same rendered text that ships
in ``Segment.content`` (separators included), so ``char_count == len(content)``
and the ±25% band is honored on real output.

Glue/heading heuristics are v1: the serving model emits no heading/list class
(``bbox_2d_json`` has no category field), so headings are inferred from text
shape + bbox geometry. The tunable constants live together below and are
exercised by ``tests/test_chunking.py`` fixtures.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from extract.core.assemble import render_table_markdown
from extract.core.models import (
    Chunk,
    ChunkType,
    PageDimensions,
    Segment,
    SegmentMember,
    TableCell,
    TablePartRef,
    TextPartRef,
)

# Internal algorithm revision (never exposed in the API — owner ruling
# 2026-07-15): bump when segment boundaries may change for identical input,
# so telemetry can distinguish algorithm generations.
# v2 (2026-07-16, from production output review): oversized multi-line prose
# splits at line breaks; tiny segments rescue-merge across heading boundaries;
# figure-only (empty-content) segments attach to a neighbor.
# v3 (2026-07-16, scanned-fax review): headings must contain letters (tall
# handwritten values are larger_type but not headings); trailing-colon labels
# glue forward to their value.
# v4 (2026-07-16, owner-approved): oversized prose cuts at sentence boundaries
# within oversized lines, so single-line mega-paragraphs split too; a single
# sentence over the budget stays whole.
CHUNKING_VERSION = 4

# Size band (plan 077 §2: fixed ±25%, not user-configurable in v1).
BAND_LOW = 0.75
BAND_HIGH = 1.25

# --- heading heuristic (§4 phase 1) ---------------------------------------- #
_HEADING_MAX_CHARS = 80
# per-line bbox height ≥ this × page median ⇒ "larger type" signal
_HEADING_LINE_HEIGHT_RATIO = 1.15
# single-line check: bbox height ≤ this × page median line height
_HEADING_SINGLE_LINE_RATIO = 1.8
_NUMBERED_SECTION_RE = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")
# --- figure caption glue ---------------------------------------------------- #
_CAPTION_MAX_CHARS = 200
_CAPTION_MAX_GAP_PT = 24.0
_CAPTION_MIN_H_OVERLAP = 0.5
_CAPTION_PREFIX_RE = re.compile(r"^(figure|fig\.?|table|chart|exhibit)\s*\d", re.IGNORECASE)
# --- list runs --------------------------------------------------------------- #
_LIST_ITEM_RE = re.compile(r"^([-•*‣▪]|\d{1,3}[.)]|[a-z][.)])\s+\S")
_LIST_X0_ALIGN_PT = 12.0
# --- structural gap boundary (§4 phase 2) ----------------------------------- #
_GAP_BOUNDARY_RATIO = 2.0
_GAP_BOUNDARY_MIN_PT = 12.0
# --- undersized rescue (§4 phase 4) ------------------------------------------ #
# Below this, a segment is a fragment, not a faithful small section: the
# rescue may cross a HEADING boundary to absorb it (never a page boundary).
# Real form pages classify caps-styled VALUES as headings, and without this a
# 52-char fragment ships trapped between two false headings (observed in
# production output, 2026-07-16).
_TINY_RESCUE_MAX_CHARS = 200

_SEPARATOR = "\n\n"


# --------------------------------------------------------------------------- #
# Elements: a real chunk, or a synthesized split-table part
# --------------------------------------------------------------------------- #
@dataclass
class _El:
    source_index: int
    chunk: Chunk
    text: str  # this element's contribution to Segment.content
    # Synthesized split-table parts only:
    part: TablePartRef | None = None
    part_page_no: int | None = None
    part_bbox: list[float] | None = None
    part_cells: list[TableCell] | None = None
    part_n_rows: int | None = None
    part_n_cols: int | None = None
    # Synthesized split-text parts only (oversized multi-line prose):
    text_part: TextPartRef | None = None

    @property
    def page_no(self) -> int:
        return self.part_page_no if self.part_page_no is not None else self.chunk.page_no

    @property
    def bbox(self) -> list[float] | None:
        return self.part_bbox if self.part is not None else self.chunk.bbox


@dataclass
class _Unit:
    """A glue run — indivisible in phases 2–4 (except table dissolution)."""

    els: list[_El]
    heading_start: bool = False
    # "strong" | "weak" | None — see _heading_strength.
    heading_strength: str | None = None


@dataclass
class _Group:
    units: list[_Unit]
    # Boundary kind BEFORE this group: "page" / "heading" are hard (phase 4
    # never merges across them); "gap" / "split" / "start" are soft.
    boundary: str = "start"

    @property
    def els(self) -> list[_El]:
        return [e for u in self.units for e in u.els]


def _sep(a: _El, b: _El) -> str:
    """Separator between two adjacent contributions: sibling text parts of the
    same source chunk rejoin with the single newline the split removed —
    a blank line would rewrite the original text's line structure."""
    if (
        a.text_part is not None
        and b.text_part is not None
        and a.source_index == b.source_index
        and b.text_part.index == a.text_part.index + 1
    ):
        return "\n"
    return _SEPARATOR


def _render_content(els: list[_El]) -> str:
    pieces: list[str] = []
    prev: _El | None = None
    for e in els:
        if not e.text:
            continue
        if prev is not None:
            pieces.append(_sep(prev, e))
        pieces.append(e.text)
        prev = e
    return "".join(pieces)


def _size(els: list[_El]) -> int:
    total = 0
    prev: _El | None = None
    for e in els:
        if not e.text:
            continue
        if prev is not None:
            total += len(_sep(prev, e))
        total += len(e.text)
        prev = e
    return total


# --------------------------------------------------------------------------- #
# Phase 0 — reading-order view
# --------------------------------------------------------------------------- #
def _contribution(c: Chunk) -> str:
    if c.chunk_type == ChunkType.IMAGE:
        # Figures contribute NOTHING to segment content (owner ruling,
        # 2026-07-15): with no description source, any placeholder is pure
        # embedding noise. The figure still rides as a zero-size member
        # (bbox + source_index grounding; caption glue carries the semantics);
        # a future summarize_figures pass changes only this return value.
        return ""
    return c.page_content or ""


def _ordered_elements(chunks: list[Chunk]) -> list[_El]:
    """The chunks in the model's own emission order, grouped by page.

    2026-07-30 reading-order fix, extending 7ccdaa202 to the RAG surface: the
    flat list ALREADY carries the provider's emission order — page assembly
    interleaves prose/KV/tables/figures by their ``seq``
    (``pdf._chunks_from_ocr_page_result``) and the multi-page stitch extends in
    page order, so list position IS the seq order here. Re-deriving order from
    bbox centers undid that for ``segments``/``content`` while ``chunks[]`` in
    the SAME response kept it: replayed over 1,392 saved champion decodes,
    57.0% of pages came back reordered (mean NED 0.0637 against the served
    order, p90 0.2376) — the same −0.137 text-accuracy shape the parse +
    assembly fix removed, on the surface that feeds embeddings.

    ``page_no`` leads the key so the hard page grouping phase 2 relies on
    survives a caller-supplied list that is not page-sorted; the positional
    tiebreak makes the sort a stable no-op within a page.
    """
    order = sorted(range(len(chunks)), key=lambda i: (chunks[i].page_no, i))
    return [_El(source_index=i, chunk=chunks[i], text=_contribution(chunks[i])) for i in order]


# --------------------------------------------------------------------------- #
# Page geometry stats (font-size proxy — no font metadata survives OCR)
# --------------------------------------------------------------------------- #
def _line_height(c: Chunk) -> float | None:
    if not c.bbox or len(c.bbox) != 4:
        return None
    h = c.bbox[3] - c.bbox[1]
    if h <= 0:
        return None
    n_lines = max(1, (c.page_content or "").count("\n") + 1)
    return h / n_lines


def _page_stats(els: list[_El]) -> dict[int, dict[str, float]]:
    """Per page: median text line height and median inter-element vertical gap."""
    by_page: dict[int, list[_El]] = {}
    for e in els:
        by_page.setdefault(e.page_no, []).append(e)
    stats: dict[int, dict[str, float]] = {}
    for page, page_els in by_page.items():
        line_heights = [
            lh
            for e in page_els
            if e.chunk.chunk_type == ChunkType.TEXT and (lh := _line_height(e.chunk))
        ]
        gaps = []
        for a, b in zip(page_els, page_els[1:], strict=False):
            if a.bbox and b.bbox:
                gap = b.bbox[1] - a.bbox[3]
                if gap > 0:
                    gaps.append(gap)
        stats[page] = {
            "line_h": statistics.median(line_heights) if line_heights else 0.0,
            "gap": statistics.median(gaps) if gaps else 0.0,
        }
    return stats


# --------------------------------------------------------------------------- #
# Phase 1 — glue rules
# --------------------------------------------------------------------------- #
def _is_all_caps(text: str) -> bool:
    # ≥2 real words, all upper: "PATIENT INTAKE FORM" yes; "HMO" and
    # "XR-2231-88" no (verified false-positive sources on real form docs —
    # checkbox labels and member IDs are not headings).
    words = [w for w in re.split(r"\s+", text) if sum(ch.isalpha() for ch in w) >= 2]
    return len(words) >= 2 and text.upper() == text


# Trailing-colon labels ("Date:", "Subject:", "Secondary:") — form/fax fields
# whose value follows as the next element. Gluing label→value keeps the pair in
# one segment (observed severed across segments on scanned fax covers).
_LABEL_MAX_CHARS = 40


def _is_label(e: _El) -> bool:
    if e.chunk.chunk_type != ChunkType.TEXT:
        return False
    text = (e.text or "").strip()
    return (
        0 < len(text) <= _LABEL_MAX_CHARS
        and "\n" not in text
        and text.endswith(":")
        and any(ch.isalpha() for ch in text)
    )


def _is_list_item(e: _El) -> bool:
    return e.chunk.chunk_type == ChunkType.TEXT and bool(
        _LIST_ITEM_RE.match((e.text or "").strip())
    )


def _lists_aligned(a: _El, b: _El) -> bool:
    if not (a.bbox and b.bbox):
        return False
    return abs(a.bbox[0] - b.bbox[0]) <= _LIST_X0_ALIGN_PT


def _heading_strength(
    e: _El, stats: dict[int, dict[str, float]], nxt: _El | None
) -> str | None:
    """None (not a heading), "weak", or "strong".

    Strong = numbered-section prefix or visibly larger type — trustworthy
    section starts. Weak = ALL-CAPS at body size only: real headers use it,
    but so do caps-styled form VALUES ("MEDICARE PART A AND B" on a face
    sheet, observed in production), so a weak heading is a boundary the
    tiny-fragment rescue may cross (§ phase 4). Title Case alone is NOT a
    heading signal: form labels ("Name:") and proper nouns are Title Case at
    body size, and treating them as headings shreds form pages.
    """
    c = e.chunk
    if c.chunk_type != ChunkType.TEXT:
        return None
    text = (e.text or "").strip()
    if not text or len(text) > _HEADING_MAX_CHARS or "\n" in text:
        return None
    if text.endswith((".", ",")):
        return None
    if not any(ch.isalpha() for ch in text):
        return None  # digits/dashes only ("##-##-##"): a value, never a heading
    # Numbered-heading vs list disambiguation: two consecutive number-prefixed
    # lines at aligned x0 are a list, not headings.
    if _is_list_item(e) and nxt is not None and _is_list_item(nxt) and _lists_aligned(e, nxt):
        return None
    page = stats.get(e.page_no, {})
    median_lh = page.get("line_h", 0.0)
    lh = _line_height(c)
    if lh is not None and median_lh > 0 and lh > _HEADING_SINGLE_LINE_RATIO * median_lh:
        return None  # taller than a single line of body text ⇒ not a one-line heading
    larger_type = lh is not None and median_lh > 0 and lh >= _HEADING_LINE_HEIGHT_RATIO * median_lh
    if bool(_NUMBERED_SECTION_RE.match(text)) or larger_type:
        return "strong"
    if _is_all_caps(text):
        return "weak"
    return None


def _is_heading_like(e: _El, stats: dict[int, dict[str, float]], nxt: _El | None) -> bool:
    return _heading_strength(e, stats, nxt) is not None


def _is_caption_for(fig: _El, e: _El) -> bool:
    if fig.chunk.chunk_type != ChunkType.IMAGE or e.chunk.chunk_type != ChunkType.TEXT:
        return False
    text = (e.text or "").strip()
    if not text or len(text) > _CAPTION_MAX_CHARS:
        return False
    fb, tb = fig.bbox, e.bbox
    if not (fb and tb):
        return False
    if tb[1] < fb[3] - 2.0:  # caption starts below the figure
        return False
    if tb[1] - fb[3] > _CAPTION_MAX_GAP_PT:
        return False
    overlap = min(fb[2], tb[2]) - max(fb[0], tb[0])
    width = min(fb[2] - fb[0], tb[2] - tb[0])
    if width <= 0 or overlap / width < _CAPTION_MIN_H_OVERLAP:
        return bool(_CAPTION_PREFIX_RE.match(text))
    return True


def _glue(els: list[_El], stats: dict[int, dict[str, float]]) -> list[_Unit]:
    units: list[_Unit] = []
    i = 0
    n = len(els)
    while i < n:
        e = els[i]
        nxt = els[i + 1] if i + 1 < n else None
        heading_strength = _heading_strength(e, stats, nxt)
        heading_start = heading_strength is not None
        run = [e]
        i += 1
        while i < n:
            last, cand = run[-1], els[i]
            if cand.page_no != last.page_no:
                break  # glue is page-local (plan §4 precedence)
            if len(run) == 1 and heading_start:
                run.append(cand)  # heading is never left dangling on its page
            elif _is_caption_for(last, cand) or _is_list_item(last) and _is_list_item(cand) and _lists_aligned(last, cand):
                run.append(cand)
            else:
                break
            i += 1
        units.append(_Unit(els=run, heading_start=heading_start, heading_strength=heading_strength))
    return units


# --------------------------------------------------------------------------- #
# Phase 2 — structural boundaries
# --------------------------------------------------------------------------- #
def _vgap(a: _El, b: _El) -> float | None:
    if a.page_no != b.page_no or not (a.bbox and b.bbox):
        return None
    return b.bbox[1] - a.bbox[3]


def _boundary_between(prev: _Unit, unit: _Unit, stats: dict[int, dict[str, float]]) -> str | None:
    a, b = prev.els[-1], unit.els[0]
    if b.page_no != a.page_no:
        return "page"
    if unit.heading_start:
        return "heading" if unit.heading_strength == "strong" else "heading_weak"
    gap = _vgap(a, b)
    if gap is not None:
        threshold = max(_GAP_BOUNDARY_RATIO * stats.get(a.page_no, {}).get("gap", 0.0),
                        _GAP_BOUNDARY_MIN_PT)
        if gap >= threshold:
            return "gap"
    return None


def _segment_stream(units: list[_Unit], stats: dict[int, dict[str, float]]) -> list[_Group]:
    groups: list[_Group] = []
    for unit in units:
        if groups:
            kind = _boundary_between(groups[-1].units[-1], unit, stats)
            if kind is None:
                groups[-1].units.append(unit)
                continue
            groups.append(_Group(units=[unit], boundary=kind))
        else:
            groups.append(_Group(units=[unit], boundary="start"))
    return groups


# --------------------------------------------------------------------------- #
# §4.1 — table row-split (valid tables on both sides)
# --------------------------------------------------------------------------- #
def _table_dims(chunk: Chunk) -> tuple[int, int]:
    cells = chunk.cells or []
    n_rows = chunk.n_rows or (max((c.row + c.row_span for c in cells), default=0))
    n_cols = chunk.n_cols or (max((c.col + c.col_span for c in cells), default=0))
    return n_rows, n_cols


_DATA_CELL_RE = re.compile(r"[\$€£¥]?\s*[\d.,:/%\-+() ]+%?")


def _row0_looks_like_header(cells: list[TableCell]) -> bool:
    """GFM forces a header row, so a visually headerless table arrives with
    its first DATA row in the header slot. Repeating that row into every
    split part would duplicate data, so only rows that look like label rows
    (some alphabetic text, no purely numeric/currency/date-shaped cells) are
    treated as headers. Conservative direction: when unsure, don't repeat."""
    row0 = [c.text.strip() for c in cells if c.row == 0]
    if not row0:
        return False
    if any(t and _DATA_CELL_RE.fullmatch(t) for t in row0):
        return False
    return any(any(ch.isalpha() for ch in t) for t in row0)


def _split_table(el: _El, max_chars: int) -> list[_El] | None:
    """Split an oversized TABLE element into valid table parts ≤ max_chars
    where possible. Returns None when the table is ineligible (no cells)."""
    chunk = el.chunk
    cells = chunk.cells
    if chunk.chunk_type != ChunkType.TABLE or not cells:
        return None
    n_rows, n_cols = _table_dims(chunk)
    if n_rows <= 0 or n_cols <= 0:
        return None
    # Header = row 0 plus any rows its spans cover — but only when row 0
    # actually looks like a header (see _row0_looks_like_header). Headerless
    # tables split with header_rows=0: no repetition, each part's first row
    # renders in the GFM header slot (how a headerless fragment looks in GFM).
    if _row0_looks_like_header(cells):
        header_rows = max((c.row + c.row_span for c in cells if c.row == 0), default=0)
        header_rows = min(header_rows, n_rows)
    else:
        header_rows = 0
    header_cells = [c for c in cells if c.row < header_rows]
    body_rows = list(range(header_rows, n_rows))
    if len(body_rows) < 2:
        return None  # nothing to split
    # rowspan connectivity: row r joins r+1 when any body cell spans across.
    joined = {
        r: any(c.row <= r < c.row + c.row_span - 1 for c in cells if c.row >= header_rows)
        for r in body_rows
    }
    row_groups: list[list[int]] = []
    for r in body_rows:
        if row_groups and joined.get(r - 1, False):
            row_groups[-1].append(r)
        else:
            row_groups.append([r])
    if len(row_groups) < 2:
        return None  # one rowspan-connected block; ships oversized intact

    def row_page(r: int) -> int:
        for c in cells:
            if c.row <= r < c.row + c.row_span and c.page_no is not None:
                return c.page_no
        return chunk.page_no

    def render_part(groups: list[list[int]]) -> tuple[str, list[TableCell], int]:
        rows = [r for g in groups for r in g]
        remap = {r: header_rows + i for i, r in enumerate(rows)}
        part_cells = [c.model_copy(deep=True) for c in header_cells]
        for c in cells:
            if c.row in remap:
                part_cells.append(c.model_copy(deep=True, update={"row": remap[c.row]}))
        part_n_rows = header_rows + len(rows)
        return render_table_markdown(part_cells, part_n_rows, n_cols), part_cells, part_n_rows

    # Pack row groups into parts: greedy while rendered size ≤ max; a page
    # change forces a part boundary (merged cross-page tables split page-local
    # first — a bbox union across pages is meaningless).
    partitions: list[list[list[int]]] = []
    for g in row_groups:
        page = row_page(g[0])
        if partitions:
            current = partitions[-1]
            same_page = row_page(current[0][0]) == page
            if same_page:
                rendered, _, _ = render_part(current + [g])
                if len(rendered) <= max_chars:
                    current.append(g)
                    continue
        partitions.append([g])

    count = len(partitions)
    if count < 2 and row_page(row_groups[0][0]) == row_page(row_groups[-1][0]):
        return None  # single part ⇒ no split actually needed
    parts: list[_El] = []
    for idx, groups in enumerate(partitions):
        rendered, part_cells, part_n_rows = render_part(groups)
        rows = [r for g in groups for r in g]
        page = row_page(rows[0])
        # Geometry: cell_grid parts union their own cells' true boxes;
        # markdown-derived cells all share the table bbox, so the union
        # degenerates to the table bbox — the honest fidelity floor.
        boxes = [
            c.bbox
            for c in part_cells
            if c.bbox and len(c.bbox) == 4 and (c.page_no is None or c.page_no == page)
        ]
        if boxes:
            bbox = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
        else:
            bbox = chunk.bbox if page == chunk.page_no else None
        parts.append(
            _El(
                source_index=el.source_index,
                chunk=chunk,
                text=rendered,
                part=TablePartRef(
                    index=idx,
                    count=count,
                    row_start=rows[0] - header_rows,
                    row_end=rows[-1] - header_rows,
                ),
                part_page_no=page,
                part_bbox=bbox,
                part_cells=part_cells,
                part_n_rows=part_n_rows,
                part_n_cols=n_cols,
            )
        )
    return parts


# A TEXT chunk whose body is a GFM table (assembly's text-level cross-page
# merge) must not line-split — the trailing part would lose the header row and
# stop being a valid table. Cheap shape check: second line is a GFM delimiter.
_GFM_DELIMITER_LINE_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
# Sentence cut points: whitespace after sentence punctuation, followed by an
# uppercase/digit start (keeps "Dr. Smith" / "1.5" style abbreviations whole
# more often than not; a rare false boundary only adds a cut OPPORTUNITY).
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=["(\[]?[A-Z0-9])')


def _split_text(el: _El, max_chars: int) -> list[_El] | None:
    """Split an oversized TEXT element into parts that each fit ``max_chars``.

    Cut points are line breaks first; a line that alone exceeds the budget
    cuts at sentence boundaries within it (v4). A single sentence over the
    budget stays whole — never cut mid-sentence. Parts are CONTIGUOUS slices
    of the source text (half-open char offsets, nothing duplicated) sharing
    the source chunk's bbox; the gap between consecutive parts is exactly the
    whitespace separator omitted at the cut.
    """
    chunk = el.chunk
    if chunk.chunk_type != ChunkType.TEXT or el.text_part is not None:
        return None
    source = el.text
    lines = source.split("\n")
    if len(lines) >= 2 and "|" in lines[1] and _GFM_DELIMITER_LINE_RE.fullmatch(lines[1]):
        return None  # GFM-shaped text (text-merged table): atomic
    # Atomic unit spans (start, end) in source order: whole lines, or sentence
    # runs within an oversized line.
    units: list[tuple[int, int]] = []
    offset = 0
    for line in lines:
        start, end = offset, offset + len(line)
        offset = end + 1  # skip the newline
        if len(line) <= max_chars:
            if line:
                units.append((start, end))
            continue
        cuts = [start + m.end() for m in _SENTENCE_BOUNDARY_RE.finditer(line)]
        bounds = [start, *cuts, end]
        for a, b in zip(bounds, bounds[1:], strict=False):
            if b > a:
                units.append((a, b))
    if len(units) < 2:
        return None
    # Greedy-pack unit spans into contiguous parts ≤ max_chars where possible
    # (an unavoidable oversized single unit ships whole).
    parts_spans: list[tuple[int, int]] = []
    part_start, part_end = units[0]
    for a, b in units[1:]:
        if b - part_start > max_chars and part_end > part_start:
            parts_spans.append((part_start, part_end))
            part_start = a
        part_end = b
    parts_spans.append((part_start, part_end))
    if len(parts_spans) < 2:
        return None
    parts: list[_El] = []
    for index, (a, b) in enumerate(parts_spans):
        text = source[a:b].rstrip()
        parts.append(
            _El(
                source_index=el.source_index,
                chunk=chunk,
                text=text,
                text_part=TextPartRef(
                    index=index,
                    count=len(parts_spans),
                    char_start=a,
                    char_end=a + len(text),
                ),
            )
        )
    return parts


# --------------------------------------------------------------------------- #
# Phase 3 — fit oversized groups
# --------------------------------------------------------------------------- #
def _split_units(units: list[_Unit], max_chars: int) -> list[list[_Unit]]:
    """Recursively split a unit list at its weakest internal boundary until
    every piece fits, dissolving oversized tables inside glue runs."""
    els = [e for u in units for e in u.els]
    if _size(els) <= max_chars:
        return [units]
    if len(units) == 1:
        unit = units[0]

        def dissolvable(e: _El) -> bool:
            if e.part is not None or e.text_part is not None or len(e.text) <= max_chars:
                return False
            return e.chunk.chunk_type in (ChunkType.TABLE, ChunkType.TEXT)

        if any(dissolvable(e) for e in unit.els):
            new_units: list[_Unit] = []
            for i, e in enumerate(unit.els):
                split = None
                if dissolvable(e):
                    split = (
                        _split_table(e, max_chars)
                        if e.chunk.chunk_type == ChunkType.TABLE
                        else _split_text(e, max_chars)
                    )
                if split:
                    # Glue dissolves at the split: preceding run stays with
                    # part 1, every further part is its own unit.
                    if new_units and i > 0:
                        new_units[-1].els.append(split[0])
                    else:
                        new_units.append(
                            _Unit(
                                els=[split[0]],
                                heading_start=unit.heading_start and i == 0,
                                heading_strength=unit.heading_strength if i == 0 else None,
                            )
                        )
                    new_units.extend(_Unit(els=[p]) for p in split[1:])
                else:
                    if i == 0:
                        new_units.append(
                            _Unit(
                                els=[e],
                                heading_start=unit.heading_start,
                                heading_strength=unit.heading_strength,
                            )
                        )
                    else:
                        new_units.append(_Unit(els=[e]))
            if len(new_units) > 1:
                return _split_units(new_units, max_chars)
        return [units]  # oversized atomic (prose/KV/glue run) — ships intact
    # Split at the widest gap between units (page-change beats everything).
    best_k, best_score = 1, float("-inf")
    for k in range(1, len(units)):
        a, b = units[k - 1].els[-1], units[k].els[0]
        if b.page_no != a.page_no:
            score = float("inf")
        else:
            gap = _vgap(a, b)
            score = gap if gap is not None else 0.0
        if score > best_score:
            best_k, best_score = k, score
    return _split_units(units[:best_k], max_chars) + _split_units(units[best_k:], max_chars)


def _fit(groups: list[_Group], max_chars: int) -> list[_Group]:
    out: list[_Group] = []
    for g in groups:
        pieces = _split_units(g.units, max_chars)
        for i, piece in enumerate(pieces):
            out.append(_Group(units=piece, boundary=g.boundary if i == 0 else "split"))
    return out


# --------------------------------------------------------------------------- #
# Phase 4 — merge undersized (bidirectional, hard boundaries respected)
# --------------------------------------------------------------------------- #
# "page" and strong "heading" are never crossed. "heading_weak" (ALL-CAPS at
# body size — indistinguishable from caps-styled form values) is crossable
# ONLY by the tiny-fragment rescue pass below.
_HARD_BOUNDARIES = frozenset({"page", "heading", "heading_weak"})


def _merge(groups: list[_Group], min_chars: int, max_chars: int, target: int) -> list[_Group]:
    groups = list(groups)
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(groups):
            if _size(g.els) >= min_chars:
                continue
            candidates: list[tuple[int, int]] = []  # (merged_size, neighbor_index)
            if i > 0 and g.boundary not in _HARD_BOUNDARIES:
                merged = _size(groups[i - 1].els + g.els)
                if merged <= max_chars:
                    candidates.append((merged, i - 1))
            if i + 1 < len(groups) and groups[i + 1].boundary not in _HARD_BOUNDARIES:
                merged = _size(g.els + groups[i + 1].els)
                if merged <= max_chars:
                    candidates.append((merged, i + 1))
            if not candidates:
                continue
            _, j = min(candidates, key=lambda t: (abs(t[0] - target), t[1]))
            lo, hi = min(i, j), max(i, j)
            groups[lo] = _Group(
                units=groups[lo].units + groups[hi].units, boundary=groups[lo].boundary
            )
            del groups[hi]
            changed = True
            break
    # Tiny-fragment rescue: a segment under _TINY_RESCUE_MAX_CHARS is a
    # fragment, not a faithful small section. After ordinary merges are
    # exhausted, it may cross a WEAK heading boundary (caps-at-body-size is
    # routinely a form value, not a section start — production 2026-07-16).
    # Strong headings and page boundaries stay uncrossable.
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(groups):
            size = _size(g.els)
            if size >= _TINY_RESCUE_MAX_CHARS:
                continue
            candidates = []
            if i > 0 and g.boundary not in ("page", "heading"):
                merged = _size(groups[i - 1].els + g.els)
                if merged <= max_chars:
                    candidates.append((merged, i - 1))
            if i + 1 < len(groups) and groups[i + 1].boundary not in ("page", "heading"):
                merged = _size(g.els + groups[i + 1].els)
                if merged <= max_chars:
                    candidates.append((merged, i + 1))
            if not candidates:
                continue
            _, j = min(candidates, key=lambda t: (abs(t[0] - target), t[1]))
            lo, hi = min(i, j), max(i, j)
            groups[lo] = _Group(
                units=groups[lo].units + groups[hi].units, boundary=groups[lo].boundary
            )
            del groups[hi]
            changed = True
            break
    return groups


def _attach_empty(groups: list[_Group]) -> list[_Group]:
    """Terminal pass: a segment with no content (figures only — they
    contribute no text by design) attaches to a SAME-PAGE neighbor, following
    first, previous second. Runs after every merge so an erased boundary can
    never be crossed by later merges (codex r2 #4). A figure alone on its
    page stays a standalone empty segment — more honest than binding it to an
    unrelated page."""
    groups = list(groups)
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(groups):
            if len(groups) < 2 or _size(g.els) != 0:
                continue
            pages = {e.page_no for e in g.els}
            j: int | None = None
            if i + 1 < len(groups) and groups[i + 1].els[0].page_no in pages:
                j = i + 1
            elif i > 0 and groups[i - 1].els[-1].page_no in pages:
                j = i - 1
            if j is None:
                continue
            lo, hi = min(i, j), max(i, j)
            groups[lo] = _Group(
                units=groups[lo].units + groups[hi].units, boundary=groups[lo].boundary
            )
            del groups[hi]
            changed = True
            break
    return groups


# --------------------------------------------------------------------------- #
# Segment assembly
# --------------------------------------------------------------------------- #
def _member(e: _El) -> SegmentMember:
    if e.text_part is not None:
        # Split-text part: offsets into the source chunk's text — nothing
        # duplicated (the slice is exactly recoverable via source_index).
        return SegmentMember(
            source_index=e.source_index,
            chunk_type=ChunkType.TEXT,
            page_no=e.page_no,
            bbox=e.bbox,
            text_part=e.text_part,
        )
    if e.part is not None:
        return SegmentMember(
            source_index=e.source_index,
            chunk_type=ChunkType.TABLE,
            page_no=e.page_no,
            bbox=e.bbox,
            table_part=e.part,
            page_content=e.text,
            cells=e.part_cells,
            n_rows=e.part_n_rows,
            n_cols=e.part_n_cols,
        )
    return SegmentMember(
        source_index=e.source_index,
        chunk_type=e.chunk.chunk_type,
        page_no=e.page_no,
        bbox=e.bbox,
    )


def _build_segment(g: _Group) -> Segment:
    els = g.els
    content = _render_content(els)
    return Segment(
        content=content,
        char_count=len(content),
        pages=sorted({e.page_no for e in els}),
        chunks=[_member(e) for e in els],
    )


def build_segments(chunks: list[Chunk], *, chunk_size: int) -> list[Segment]:
    """Group ``chunks`` into segments targeting ``chunk_size`` characters
    (±25%). Pure and deterministic."""
    if not chunks:
        return []
    max_chars = round(chunk_size * BAND_HIGH)
    min_chars = round(chunk_size * BAND_LOW)
    els = _ordered_elements(chunks)
    stats = _page_stats(els)
    units = _glue(els, stats)
    groups = _segment_stream(units, stats)
    groups = _fit(groups, max_chars)
    groups = _merge(groups, min_chars, max_chars, chunk_size)
    groups = _attach_empty(groups)
    return [_build_segment(g) for g in groups]


def render_document_content(chunks: list[Chunk]) -> str:
    """The entire document as one text string, in reading order.

    Reuses phase 0 of the chunker (``_ordered_elements``), so ``content`` ships
    the same order as ``chunks[]`` — which is what keeps a mid-page table with
    its surrounding text instead of displaced. This is not a structured-markdown
    pass — no heading detection or reflow; it is the chunks' own text,
    concatenated. Table chunks carry their existing markdown render; figures
    contribute nothing.
    """
    if not chunks:
        return ""
    return _render_content(_ordered_elements(chunks))


def page_dimensions(page_sizes: list[tuple[float, float]]) -> list[PageDimensions]:
    return [
        PageDimensions(page_no=i + 1, width=w, height=h)
        for i, (w, h) in enumerate(page_sizes)
    ]


def telemetry_counters(segments: list[Segment], *, chunk_size: int) -> dict[str, int | float]:
    max_chars = round(chunk_size * BAND_HIGH)
    min_chars = round(chunk_size * BAND_LOW)
    in_band = sum(1 for s in segments if min_chars <= s.char_count <= max_chars)
    tables_split = len(
        {
            (m.source_index)
            for s in segments
            for m in s.chunks
            if m.table_part is not None
        }
    )
    return {
        "segment_count": len(segments),
        "segments_in_band_ratio": round(in_band / len(segments), 4) if segments else 0.0,
        "segment_tables_split": tables_split,
        "segment_oversize_atomic": sum(1 for s in segments if s.char_count > max_chars),
        "segment_undersize_final": sum(1 for s in segments if s.char_count < min_chars),
    }
