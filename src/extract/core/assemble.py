"""Document-level assembly: deterministic cross-page table merging.

Lane 032 (the internal measurement records §4 Workstream A).
The page-parallel pipeline parses pages independently, so a table spanning a
page boundary ships as two unrelated chunks. This module merges such fragments
with MinerU-derived heuristics, ported to the signals the chunk schema actually
carries (table-level bbox + page rect + column counts + cell/row text — there
is no per-cell geometry, no layout class, and no font size on the live path).
Reference implementation read (not vendored): PaddleX
``paddlex/inference/pipelines/layout_parsing/merge_table.py`` (Apache-2.0,
"adapted from MinerU").

**Two table representations exist on the live path** (measured on prod parses,
2026-06-11): the qwen_lora provider types a record as a TABLE chunk (with
``cells``) only when its GFM is strictly rectangular; a table whose cells
contain newlines fails ``_looks_like_markdown_table`` and ships as a TEXT
chunk whose ``page_content`` IS the GFM markdown. Real multi-page tables are
overwhelmingly the latter. The assembly layer therefore works on a
:class:`TableView` abstraction: ``cells`` views (typed TABLE chunks) and
``gfm_text`` views (TEXT chunks whose content is GFM-table-shaped). ``cells``+``cells``
fragments merge with full cell structure; any pair involving a ``gfm_text``
fragment merges at the GFM text level. The live parser (``qwen_lora.py``) is
read, never edited (plan §7).

Pure functions only — no I/O, no provider calls, no settings reads. The one
call site is ``extract_pdf`` (``pdf.py``), gated by
``Settings.EXTRACT_ASSEMBLY_MODE`` (``off`` | ``shadow`` | ``on``, default
``off``): ``shadow`` computes :class:`BoundaryDecision`s without mutating
output; ``on`` replaces fragments with the merged chunk.

Gates per boundary (page *p* → *p+1*), in order:

1. **Geometry pre-gate** — the lowest table on *p* ends within
   ``margin · page_height`` of the page bottom AND the highest table on *p+1*
   starts within ``margin · page_height`` of the top. Boundaries passing this
   gate are the ``pages_merge_candidate`` population.
2. **No intervening content** — nothing sits between the fragment and the page
   edge except page furniture: anything inside the extreme ``furniture_band``
   of the page, a pure page-number line, or a short "(continued)" caption
   (also counted as a positive signal).
3. **Width ratio** — fragment bbox widths within ``width_ratio_max``
   (MinerU's conditional 10% threshold).
4. **Column compatibility** — equal column counts, OR fragment 2's leading
   rows repeat fragment 1's leading rows (header repetition — also the dedup
   rule), OR the seam rows match (last row of fragment 1 and first data row of
   fragment 2 have the same width). Fragment KIND is parser noise (the strict
   router types small clean fragments and leaves big ones as text), so mixed
   kinds merge at the text level rather than rejecting.

**Header-only continuation path** (measured on Clyde referral packets,
2026-07-31): when fragment 1 is a *header-only* table — one row, the header,
zero data rows — the parser failed to structure that table's rows (they ship
as loose text spans below it), and the normal gates are meaningless: the
sliver ends mid-page (gate 1), its orphaned rows read as intervening content
(gate 2), and a continuation with empty trailing columns is narrower (gate 3).
A header-only table is itself the anomaly signal, so such boundaries take a
dedicated gate set instead:

a. everything below fragment 1 on page *p* (excluding furniture) is TEXT
   confined to the fragment's x-range in short spans
   (``ROW_SPAN_MAX_HEIGHT_FRAC``), the spans are *many* and *narrow*
   (``ROWLIKE_MIN_SPANS``, ``ROWLIKE_MAX_MEDIAN_WIDTH_FRAC`` — orphaned rows
   are many small fragments, a paragraph is one-to-few table-width blocks),
   and that text reaches into the bottom ``margin`` band (evidence the rows
   run off the page); when fragment 1 itself ends inside the bottom band and
   nothing sits below it, this check passes vacuously *but* the merge then
   demands corroboration (see below);
b. fragment 2 starts within the top ``margin`` band of page *p+1*, with no
   intervening non-furniture content above it (as the normal path's
   ``intervening_text_b`` — a caption above fragment 2 means a new table);
c. fragment 2 is left-aligned with fragment 1 (``x0`` within
   ``ALIGN_TOL_FRAC`` of fragment 1's width) and no wider;
d. fragment 2 has no more columns than fragment 1's header (continuations
   drop empty trailing columns);
e. in the vacuous case of (a) there is zero orphaned-row evidence, so the
   merge additionally requires a repeated header, a "(continued)" marker, or
   the normal path's width-ratio + column-compatibility evidence
   (``continuation_uncorroborated`` otherwise) — without this, any two
   left-aligned tables straddling a boundary would fuse.

Fragment 2's first row is DATA here (its GFM "header" position is parser
syntax, not semantics), so the merged emission keeps it; a fragment 2 that
does repeat the header still dedups via ``leading_header_overlap``.

Emission trade-off (explicit decision, 2026-07-31): the merged chunk sits at
fragment 1's position while the orphaned row spans stay put below it, so
fragment 2's rows precede page *p*'s soup rows in the text stream. True
reading order needs the row-folding follow-up
(the internal measurement records); the
measured effect of merging alone is a deterministic recovery of the
continuation rows, which dominates the ordering cost.

Merged emission (plan §4 A3): the merged chunk replaces its fragments;
``page_no``/``bbox`` come from the first fragment; fragment-2's repeated
header rows are dropped; ``Chunk.merged_from_pages`` carries provenance; and
``confidence`` is the min of the parents (fail-conservative — agreed with
lane 035). ``cells`` fragments re-index fragment-2's rows and re-render GFM
through :func:`render_table_markdown` (with ``TableCell.page_no`` per cell);
``gfm_text`` fragments splice fragment-2's raw data lines under fragment 1 so
cell content is preserved byte-for-byte.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from extract.core.models import Chunk, ChunkType, TableCell

DEFAULT_MARGIN = 0.12          # geometry pre-gate band, fraction of page height
WIDTH_RATIO_MAX = 0.10         # MinerU width-ratio threshold
# Extreme top/bottom of the page treated as running headers/footers. Measured
# on PubMed pages: journal running heads sit at ~4.5-5% of page height, so 4%
# misses them and blocks nearly every true continuation boundary; 7% covers
# them while staying inside the 12% geometry margin. (No layout classes exist
# on the live path — this geometric proxy replaces MinerU's header/footer
# labels.)
FURNITURE_BAND = 0.07
MAX_HEADER_ROWS = 5            # leading rows compared for header repetition
MAX_MARKER_CHARS = 60          # "(continued)" captions are short single lines
# Header-only continuation path: fragment-2 left-edge alignment tolerance as a
# fraction of fragment 1's width (floored at 6pt — sub-column-width jitter).
ALIGN_TOL_FRAC = 0.03
ALIGN_TOL_MIN_PTS = 6.0
# Orphaned table rows ship as many SHORT spans (a few wrapped lines at most);
# a flowing paragraph is one tall block. 8% of page height ≈ 63pt on letter —
# double the tallest row span measured on the Clyde packets (38pt).
ROW_SPAN_MAX_HEIGHT_FRAC = 0.08
# ...and they are MANY and NARROW: the Clyde shape is 24 spans, none wider
# than 43% of the table. A paragraph block is one-to-few spans at table width.
ROWLIKE_MIN_SPANS = 3
ROWLIKE_MAX_MEDIAN_WIDTH_FRAC = 0.8

_CONTINUED_RE = re.compile(r"\(?\bcont(?:inued|'?d)?\b\.?\)?", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[-–—]?\s*\d{1,4}\s*(?:(?:of|/)\s*\d{1,4})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)
_GFM_DELIM_CELL_RE = re.compile(r":?-+:?")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_WS = re.compile(r"\s+")


@dataclass
class BoundaryDecision:
    """The assembly verdict for one page boundary that had a table view on both
    sides. ``rejection`` is the FIRST failed gate (``None`` when merged):
    ``geometry`` | ``intervening_text_a`` | ``intervening_text_b`` |
    ``width_ratio`` | ``column_mismatch`` — or, on the header-only
    continuation path, ``continuation_rows_below`` | ``continuation_geometry``
    | ``intervening_text_b`` | ``continuation_alignment`` |
    ``continuation_columns`` | ``continuation_uncorroborated``. PHI-safe by
    construction (no text content) — safe for counters; the sidecar diagnostic
    carries content."""

    page_a: int
    page_b: int
    merged: bool
    rejection: str | None = None
    geometry_candidate: bool = False     # passed gate 1 (the candidate universe)
    continued_marker: bool = False
    signals: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Table rendering — the single home (pdf.py imports these; plan §9: merged
# tables re-render through the same canonical renderer as page-level tables).
#
# GFM and HTML are SIBLING serializations of one cell model, never derived from
# each other: `_cell_grid` fills the square grid once and both renderers format
# it. Deriving HTML by re-parsing our own markdown would give two grammars that
# drift, and markdown cannot express spans at all.
# --------------------------------------------------------------------------- #
def _cell_grid(cells: list[TableCell], n_rows: int, n_cols: int) -> list[list[str]] | None:
    """Fill the square grid once. Merged cells repeat their text across every
    spanned position, so both renderers see a rectangular table."""
    if n_rows <= 0 or n_cols <= 0:
        return None
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in cells:
        for dr in range(c.row_span):
            for dc in range(c.col_span):
                r, k = c.row + dr, c.col + dc
                if 0 <= r < n_rows and 0 <= k < n_cols and not grid[r][k]:
                    grid[r][k] = c.text
    return grid


def render_table_html(cells: list[TableCell], n_rows: int, n_cols: int) -> str:
    """Render cells as an HTML table (HTML-table compatibility mode).

    Row 0 is the header row and is emitted as ``<th>`` **whatever it contains,
    including when it is empty**. It is NEVER synthesized by promoting the first
    body row: on a line-numbered transcript the customer contract (rule 4,
    owner + customer 2026-08-02) mandates an EMPTY header, and promoting line 1
    to a heading is precisely the defect measured on the reference vendor (242 of
    256 gutter pages). An empty header therefore renders as
    ``<tr><th></th><th></th></tr>`` and every printed line stays a body row.

    Shape matches the reference vendor's: bare ``<table>``/``<tr>``, no
    ``<thead>``/``<tbody>`` wrapper. Spans are not emitted — a table sourced from
    the markdown cell floor has ``row_span``/``col_span`` of 1 by construction
    (``qwen_lora._table_from_markdown``), and repeating a spanned cell's text
    across its covered positions keeps the grid rectangular and lossless.
    """
    grid = _cell_grid(cells, n_rows, n_cols)
    if grid is None:
        return ""
    out = ["<table>"]
    for i, row in enumerate(grid):
        tag = "th" if i == 0 else "td"
        out.append(
            "<tr>" + "".join(f"<{tag}>{_html_escape(v)}</{tag}>" for v in row) + "</tr>"
        )
    out.append("</table>")
    return "".join(out)


def _html_escape(text: str) -> str:
    """Escape the five XML entities. Legal text carries `&`, `<` and quotes
    (``Smith & Jones``, ``<redacted>``), so an unescaped render is malformed
    markup, not a cosmetic issue."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_table_markdown(cells: list[TableCell], n_rows: int, n_cols: int) -> str:
    """Render cells as a GitHub-flavored markdown table. Merged cells repeat
    their text across the spanned grid positions so the rendering is square."""
    grid = _cell_grid(cells, n_rows, n_cols)
    if grid is None:
        return ""
    esc = [[v.replace("|", r"\|").replace("\n", " ") for v in row] for row in grid]
    lines = ["| " + " | ".join(esc[0]) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
    for row in esc[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# table views — one abstraction over typed TABLE chunks and GFM-shaped TEXT
# --------------------------------------------------------------------------- #
def _norm_cell(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text or "")).casefold().strip()


def _split_gfm_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]
    return [c.strip() for c in _UNESCAPED_PIPE_RE.split(s)]


def _is_gfm_delimiter(line: str) -> bool:
    cells = _split_gfm_row(line)
    return bool(cells) and all(_GFM_DELIM_CELL_RE.fullmatch(c.strip()) for c in cells)


@dataclass
class TableView:
    """A table fragment as the merge gates see it, independent of whether the
    parser typed it (``cells``) or left it as GFM text (``gfm_text``)."""

    chunk: Chunk
    kind: str                              # "cells" | "gfm_text"
    row_sigs: list[tuple]                  # normalized per-row signatures
    row_widths: list[int]                  # per-row visual width
    row_cells: list[list[str]]             # raw cell texts per row (delimiters dropped)
    n_cols: int
    # gfm_text only: raw content lines + the line index where each row starts
    raw_lines: list[str] | None = None
    row_line_starts: list[int] | None = None


def _cells_view(chunk: Chunk) -> TableView:
    rows: dict[int, list[TableCell]] = {}
    for c in chunk.cells or []:
        rows.setdefault(c.row, []).append(c)
    ordered = [sorted(cs, key=lambda c: c.col) for _, cs in sorted(rows.items())]
    n_rows, n_cols = len(ordered), chunk.n_cols or 0
    # square text grid (render-style fill) — the canonical per-row cell texts
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for i, r in enumerate(ordered):
        for c in r:
            for dr in range(c.row_span):
                for dc in range(c.col_span):
                    rr, kk = i + dr, c.col + dc
                    if 0 <= rr < n_rows and 0 <= kk < n_cols and not grid[rr][kk]:
                        grid[rr][kk] = c.text
    return TableView(
        chunk=chunk, kind="cells",
        row_sigs=[tuple((_norm_cell(c.text), c.col_span) for c in r) for r in ordered],
        row_widths=[sum(c.col_span for c in r) for r in ordered],
        row_cells=grid,
        n_cols=n_cols,
    )


def _gfm_text_view(chunk: Chunk) -> TableView | None:
    """Lenient GFM parse: a TEXT chunk whose content STARTS as a GFM table
    (header row + delimiter row). Lines not starting with ``|`` are cell
    continuations (the exact shape the strict typed-table router rejects) and
    fold into the current row."""
    lines = (chunk.page_content or "").splitlines()
    if len(lines) < 2 or not lines[0].lstrip().startswith("|"):
        return None
    header = _split_gfm_row(lines[0])
    if len(header) < 2 or not _is_gfm_delimiter(lines[1]):
        return None
    rows: list[list[str]] = []
    starts: list[int] = []
    for i, line in enumerate(lines):
        if _is_gfm_delimiter(line):
            continue
        if line.lstrip().startswith("|"):
            rows.append(_split_gfm_row(line))
            starts.append(i)
        elif rows:
            # cell continuation: the row's terminating pipe lands at the end of
            # the last continuation line — strip it, it's syntax not content
            cont = line.strip()
            if cont.endswith("|") and not cont.endswith(r"\|"):
                cont = cont[:-1].rstrip()
            rows[-1][-1] = f"{rows[-1][-1]} {cont}".strip()
    n_cols = len(header)
    return TableView(
        chunk=chunk, kind="gfm_text",
        row_sigs=[tuple((_norm_cell(c), 1) for c in r) for r in rows],
        row_widths=[len(r) for r in rows],
        row_cells=[r + [""] * (n_cols - len(r)) if len(r) < n_cols else r for r in rows],
        n_cols=n_cols,
        raw_lines=lines, row_line_starts=starts,
    )


def table_view(chunk: Chunk) -> TableView | None:
    """The chunk's table view, or ``None`` if it isn't table-shaped."""
    if chunk.chunk_type == ChunkType.TABLE:
        return _cells_view(chunk)
    if chunk.chunk_type == ChunkType.TEXT:
        return _gfm_text_view(chunk)
    return None


def leading_header_overlap(a: TableView, b: TableView,
                           *, max_rows: int = MAX_HEADER_ROWS) -> int:
    """How many of fragment B's leading rows repeat fragment A's leading rows
    (text + span identical, per the reference's ``detect_table_headers``)."""
    n = 0
    for sig_a, sig_b in zip(a.row_sigs[:max_rows], b.row_sigs[:max_rows], strict=False):
        if sig_a != sig_b or not any(t for t, _ in sig_b):
            break
        n += 1
    return n


def _columns_compatible(a: TableView, b: TableView, header_rows: int) -> bool:
    if a.n_cols and a.n_cols == b.n_cols:
        return True
    if header_rows > 0:
        return True
    if not a.row_widths or not b.row_widths:
        return False
    # seam-row match: last row of A vs B's first data row (reference rows_match)
    first_data = b.row_widths[header_rows] if header_rows < len(b.row_widths) else b.row_widths[0]
    return a.row_widths[-1] == first_data


# --------------------------------------------------------------------------- #
# furniture / intervening-content rules
# --------------------------------------------------------------------------- #
def _is_furniture(chunk: Chunk, page_height: float, *, band: float) -> bool:
    bbox = chunk.bbox or [0.0, 0.0, 0.0, 0.0]
    y0, y1 = bbox[1], bbox[3]
    if y1 <= page_height * band or y0 >= page_height * (1 - band):
        return True
    if chunk.chunk_type != ChunkType.TEXT:
        return False
    text = (chunk.page_content or "").strip()
    if _PAGE_NUMBER_RE.match(text):
        return True
    # "(continued)" captions only: short, single-line. A paragraph that merely
    # contains the word "continued" is real content and must block the merge.
    return ("\n" not in text and len(text) <= MAX_MARKER_CHARS
            and bool(_CONTINUED_RE.search(text)))


def _intervening(chunks: list[Chunk], *, below_y: float | None, above_y: float | None,
                 page_height: float, band: float) -> bool:
    """Non-furniture content between a fragment and its page edge."""
    for c in chunks:
        if c.bbox is None:
            continue
        center = (c.bbox[1] + c.bbox[3]) / 2.0
        between = ((below_y is not None and center > below_y)
                   or (above_y is not None and center < above_y))
        if between and not _is_furniture(c, page_height, band=band):
            return True
    return False


def _orphaned_rows_below(chunks: list[Chunk], tbl: Chunk, page_height: float,
                         *, band: float) -> tuple[bool, list[list[float]]]:
    """Header-only continuation gate (a), first half: is everything below
    ``tbl`` on its page (excluding furniture) TEXT confined to the table's
    x-range in short spans? Returns ``(ok, span_bboxes)``; the count / width /
    reach requirements on those spans are the caller's second half."""
    bbox = tbl.bbox or [0.0, 0.0, 0.0, 0.0]
    x_tol = max(ALIGN_TOL_MIN_PTS, ALIGN_TOL_FRAC * (bbox[2] - bbox[0]))
    spans: list[list[float]] = []
    for c in chunks:
        if c is tbl or c.bbox is None:
            continue
        center = (c.bbox[1] + c.bbox[3]) / 2.0
        if center <= bbox[3] or _is_furniture(c, page_height, band=band):
            continue
        if (c.chunk_type != ChunkType.TEXT
                or c.bbox[0] < bbox[0] - x_tol or c.bbox[2] > bbox[2] + x_tol
                or c.bbox[3] - c.bbox[1] > page_height * ROW_SPAN_MAX_HEIGHT_FRAC):
            return False, spans
        spans.append(list(c.bbox))
    return True, spans


def _has_continued_marker(chunks: list[Chunk], above_y: float) -> bool:
    return any(
        c.chunk_type == ChunkType.TEXT
        and c.bbox is not None
        and (c.bbox[1] + c.bbox[3]) / 2.0 < above_y
        and len((c.page_content or "").strip()) <= MAX_MARKER_CHARS
        and _CONTINUED_RE.search(c.page_content or "")
        for c in chunks
    )


# --------------------------------------------------------------------------- #
# merge emission
# --------------------------------------------------------------------------- #
def _merged_provenance(a: Chunk, b: Chunk) -> list[int]:
    return sorted(set(a.merged_from_pages or [a.page_no])
                  | set(b.merged_from_pages or [b.page_no]))


def _merged_confidence(a: Chunk, b: Chunk) -> float | None:
    confidences = [v for v in (a.confidence, b.confidence) if v is not None]
    return min(confidences) if confidences else None


def _with_page_no(cells: list[TableCell] | None, page_no: int) -> list[TableCell]:
    return [c.model_copy(update={"page_no": c.page_no if c.page_no is not None else page_no})
            for c in cells or []]


def _merged_table_output_format(a: Chunk, b: Chunk) -> str | None:
    """The merged table's honest representation: the WEAKER of its fragments.

    A merged table holds both fragments' cells, so if either side fell back to
    markdown-derived cells the merged table contains coarse table-level boxes and
    must say ``markdown`` — even when the other side localized perfectly. Dropping
    this (it used to be omitted) let the ``cell_grid`` post-pass stamp the merged
    chunk ``cell_grid``, advertising repeated coarse boxes as true per-cell
    coordinates. ``None`` is preserved so the post-pass can still fill it in.
    """
    formats = {a.table_output_format, b.table_output_format}
    if "markdown" in formats:
        return "markdown"
    return a.table_output_format or b.table_output_format


def merge_table_chunks(a: Chunk, b: Chunk, *, header_rows: int) -> Chunk:
    """Merged TABLE chunk from two typed (``cells``) fragments (plan §4 A3)."""
    a_rows = a.n_rows or (max((c.row for c in a.cells or []), default=-1) + 1)
    cells = _with_page_no(a.cells, a.page_no)
    for c in _with_page_no(b.cells, b.page_no):
        if c.row < header_rows:
            continue                      # deduped repeated header
        cells.append(c.model_copy(update={"row": c.row - header_rows + a_rows}))
    n_rows = a_rows + max((b.n_rows or 0) - header_rows, 0)
    n_cols = max(a.n_cols or 0, b.n_cols or 0)
    return Chunk(
        page_content=render_table_markdown(cells, n_rows, n_cols),
        page_no=a.page_no,
        bbox=list(a.bbox) if a.bbox else None,
        chunk_type=ChunkType.TABLE,
        confidence=_merged_confidence(a, b),
        cells=cells,
        n_rows=n_rows,
        n_cols=n_cols,
        merged_from_pages=_merged_provenance(a, b),
        table_output_format=_merged_table_output_format(a, b),
    )


def _data_lines(view: TableView, *, skip_rows: int) -> list[str]:
    """Fragment 2's contribution as GFM lines, after dropping ``skip_rows``
    deduped header rows and any delimiter row. ``gfm_text`` fragments splice
    their raw lines verbatim (multi-line cell content survives byte-for-byte);
    ``cells`` fragments render row lines from the cell grid."""
    if view.kind == "gfm_text":
        starts = view.row_line_starts or []
        start = starts[skip_rows] if skip_rows < len(starts) else len(view.raw_lines or [])
        return [ln for ln in (view.raw_lines or [])[start:] if not _is_gfm_delimiter(ln)]
    return ["| " + " | ".join(c.replace("|", r"\|").replace("\n", " ") for c in row) + " |"
            for row in view.row_cells[skip_rows:]]


def merge_text_level(a_view: TableView, b_view: TableView, *, header_rows: int) -> Chunk:
    """Merged TEXT chunk for ``gfm_text``+``gfm_text`` or mixed-kind fragments:
    fragment 2's data lines (deduped header + delimiter dropped) splice under
    fragment 1's GFM content. The conservative common denominator — a ``cells``
    fragment 1's ``page_content`` already IS the canonical GFM render."""
    a, b = a_view.chunk, b_view.chunk
    contribution = _data_lines(b_view, skip_rows=header_rows)
    merged_text = (a.page_content or "").rstrip("\n")
    if contribution:
        merged_text += "\n" + "\n".join(contribution)
    return Chunk(
        page_content=merged_text,
        page_no=a.page_no,
        bbox=list(a.bbox) if a.bbox else None,
        chunk_type=ChunkType.TEXT,
        confidence=_merged_confidence(a, b),
        merged_from_pages=_merged_provenance(a, b),
    )


# --------------------------------------------------------------------------- #
# the assembly pass
# --------------------------------------------------------------------------- #
def _decide(chunks_a: list[Chunk], chunks_b: list[Chunk], va: TableView, vb: TableView,
            h_a: float, h_b: float, *, margin: float, width_ratio_max: float,
            band: float, max_header_rows: int) -> BoundaryDecision:
    tbl_a, tbl_b = va.chunk, vb.chunk
    d = BoundaryDecision(page_a=tbl_a.page_no, page_b=tbl_b.page_no, merged=False)
    d.signals.update({"kind_a": va.kind, "kind_b": vb.kind})
    bbox_a, bbox_b = tbl_a.bbox or [0, 0, 0, 0], tbl_b.bbox or [0, 0, 0, 0]
    d.continued_marker = _has_continued_marker(chunks_b, bbox_b[1])

    if len(va.row_sigs) == 1 and va.row_sigs[0] and any(t for t, _ in va.row_sigs[0]):
        # fragment 1 is header-only: the anomaly path (module docstring)
        return _decide_headerless_continuation(
            d, chunks_a, chunks_b, va, vb, h_a, h_b,
            margin=margin, width_ratio_max=width_ratio_max, band=band,
            max_header_rows=max_header_rows)

    near_bottom = bbox_a[3] >= h_a * (1 - margin)
    near_top = bbox_b[1] <= h_b * margin
    if not (near_bottom and near_top):
        d.rejection = "geometry"
        return d
    d.geometry_candidate = True

    others_a = [c for c in chunks_a if c is not tbl_a]
    others_b = [c for c in chunks_b if c is not tbl_b]
    if _intervening(others_a, below_y=bbox_a[3], above_y=None, page_height=h_a, band=band):
        d.rejection = "intervening_text_a"
        return d
    if _intervening(others_b, below_y=None, above_y=bbox_b[1], page_height=h_b, band=band):
        d.rejection = "intervening_text_b"
        return d

    w_a, w_b = bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0]
    if w_a <= 0 or w_b <= 0 or abs(w_a - w_b) / min(w_a, w_b) >= width_ratio_max:
        d.rejection = "width_ratio"
        if min(w_a, w_b) > 0:
            d.signals["width_ratio"] = round(abs(w_a - w_b) / min(w_a, w_b), 4)
        return d

    header_rows = leading_header_overlap(va, vb, max_rows=max_header_rows)
    d.signals.update({"n_cols_a": va.n_cols, "n_cols_b": vb.n_cols,
                      "header_rows_deduped": header_rows})
    if not _columns_compatible(va, vb, header_rows):
        d.rejection = "column_mismatch"
        return d

    d.merged = True
    return d


def _decide_headerless_continuation(
        d: BoundaryDecision, chunks_a: list[Chunk], chunks_b: list[Chunk],
        va: TableView, vb: TableView, h_a: float, h_b: float, *, margin: float,
        width_ratio_max: float, band: float,
        max_header_rows: int) -> BoundaryDecision:
    """Gates a-e of the header-only continuation path (module docstring)."""
    d.signals["variant"] = "headerless_continuation"
    tbl_a, tbl_b = va.chunk, vb.chunk
    bbox_a, bbox_b = tbl_a.bbox or [0, 0, 0, 0], tbl_b.bbox or [0, 0, 0, 0]
    others_a = [c for c in chunks_a if c is not tbl_a]
    others_b = [c for c in chunks_b if c is not tbl_b]

    # geometry-candidate flag first (shadow-mode near-miss visibility): does
    # page-a content — the sliver or ANY non-furniture text below it,
    # row-shaped or not — reach the bottom band, with fragment 2 opening its
    # page? The shape verdicts below refine, never widen, this population.
    below_reach = max((c.bbox[3] for c in others_a
                       if c.bbox is not None
                       and (c.bbox[1] + c.bbox[3]) / 2.0 > bbox_a[3]
                       and not _is_furniture(c, h_a, band=band)),
                      default=0.0)
    a_reach = max(bbox_a[3], below_reach)
    near_top_b = bbox_b[1] <= h_b * margin
    d.geometry_candidate = a_reach >= h_a * (1 - margin) and near_top_b

    # (a) orphaned rows: only short row-like text below the header sliver
    rows_ok, spans = _orphaned_rows_below(others_a, tbl_a, h_a, band=band)
    if not rows_ok:
        d.rejection = "continuation_rows_below"
        return d
    vacuous = not spans
    if spans:
        rows_y1 = max(s[3] for s in spans)
        if max(bbox_a[3], rows_y1) < h_a * (1 - margin):
            d.rejection = "continuation_geometry"
            return d
        widths = sorted(s[2] - s[0] for s in spans)
        median_w = widths[len(widths) // 2]
        tbl_w = bbox_a[2] - bbox_a[0]
        if (len(spans) < ROWLIKE_MIN_SPANS
                or (tbl_w > 0 and median_w > tbl_w * ROWLIKE_MAX_MEDIAN_WIDTH_FRAC)):
            d.rejection = "continuation_rows_below"
            return d
    elif bbox_a[3] < h_a * (1 - margin):
        d.rejection = "continuation_geometry"
        return d

    # (b) fragment 2 opens its page, nothing above it but furniture
    if not near_top_b:
        d.rejection = "continuation_geometry"
        return d
    if _intervening(others_b, below_y=None, above_y=bbox_b[1], page_height=h_b, band=band):
        d.rejection = "intervening_text_b"
        return d

    # (c) left-aligned, no wider
    x_tol = max(ALIGN_TOL_MIN_PTS, ALIGN_TOL_FRAC * (bbox_a[2] - bbox_a[0]))
    if abs(bbox_b[0] - bbox_a[0]) > x_tol or bbox_b[2] > bbox_a[2] + x_tol:
        d.rejection = "continuation_alignment"
        return d

    # (d) continuations drop empty trailing columns, never add them
    if vb.n_cols > va.n_cols:
        d.rejection = "continuation_columns"
        return d

    header_rows = leading_header_overlap(va, vb, max_rows=max_header_rows)
    d.signals.update({"n_cols_a": va.n_cols, "n_cols_b": vb.n_cols,
                      "header_rows_deduped": header_rows})

    # (e) with zero orphaned-row evidence, demand corroboration: a repeated
    # header, a "(continued)" marker, or the normal path's width + column
    # evidence — otherwise any two left-aligned tables at a boundary fuse.
    if vacuous and header_rows == 0 and not d.continued_marker:
        w_a, w_b = bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0]
        width_ok = (w_a > 0 and w_b > 0
                    and abs(w_a - w_b) / min(w_a, w_b) < width_ratio_max)
        if not (width_ok and _columns_compatible(va, vb, 0)):
            d.rejection = "continuation_uncorroborated"
            return d

    d.merged = True
    return d


def assemble_chunks(
    chunks: list[Chunk],
    page_heights_pts: list[float],
    *,
    margin: float = DEFAULT_MARGIN,
    width_ratio_max: float = WIDTH_RATIO_MAX,
    furniture_band: float = FURNITURE_BAND,
    max_header_rows: int = MAX_HEADER_ROWS,
) -> tuple[list[Chunk], list[BoundaryDecision]]:
    """The assembly pass. Returns ``(assembled_chunks, decisions)``; the input
    list is never mutated, and with no merge the output is the same objects in
    the same order (the no-op property). ``page_heights_pts[i]`` is page
    ``i+1``'s height in PDF points (chunk bboxes are in points).

    Boundaries are processed LAST page first (the reference's order) so a table
    spanning 3+ pages cascades: (p+1,p+2) merges into the p+1 fragment, which
    then merges into p. Decisions are recorded for every boundary with a table
    view on both sides.
    """
    working = list(chunks)
    decisions: list[BoundaryDecision] = []
    n_pages = len(page_heights_pts)

    def height(page_no: int) -> float:
        return page_heights_pts[page_no - 1] if 0 < page_no <= n_pages else 792.0

    for p in range(n_pages - 1, 0, -1):           # boundary p -> p+1, 1-based
        on_a = [c for c in working if c.page_no == p]
        on_b = [c for c in working if c.page_no == p + 1]
        views_a = [v for v in (table_view(c) for c in on_a) if v is not None]
        views_b = [v for v in (table_view(c) for c in on_b) if v is not None]
        if not views_a or not views_b:
            continue
        va = max(views_a, key=lambda v: (v.chunk.bbox or [0, 0, 0, 0])[3])
        vb = min(views_b, key=lambda v: (v.chunk.bbox or [0, 0, 0, 0])[1])
        decision = _decide(on_a, on_b, va, vb, height(p), height(p + 1),
                           margin=margin, width_ratio_max=width_ratio_max,
                           band=furniture_band, max_header_rows=max_header_rows)
        decisions.append(decision)
        if decision.merged:
            header_rows = decision.signals.get("header_rows_deduped", 0)
            if va.kind == "cells" and vb.kind == "cells":
                # typed+typed keeps full cell structure + canonical re-render
                merged = merge_table_chunks(va.chunk, vb.chunk, header_rows=header_rows)
            else:
                # gfm+gfm and mixed-kind merge at the GFM text level
                merged = merge_text_level(va, vb, header_rows=header_rows)
            working = [merged if c is va.chunk else c
                       for c in working if c is not vb.chunk]
    decisions.reverse()                            # report in page order
    return working, decisions


def decision_counters(decisions: list[BoundaryDecision]) -> dict[str, int]:
    """Low-cardinality, PHI-safe counters for ``timer.meta`` (plan §4 A4)."""
    candidates = [d for d in decisions if d.geometry_candidate]
    return {
        "pages_merge_candidate": len(candidates),
        "tables_merged": sum(1 for d in decisions if d.merged),
        "merge_gate_rejections": sum(1 for d in candidates if not d.merged),
    }
