"""OCR provider protocol. Keeps ``core`` free of boto3 (or any other SDK)
unless the user actually selects a provider that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class OCRBlock:
    """A single OCR result — text plus its bbox on the page.

    ``confidence`` is ``None`` when the provider has no real signal (plan 028
    B3); providers must never emit a placeholder 0.0 — a real 0.0 is a
    legitimate (terrible) score and survives the pipeline mapping.

    ``seq`` is the element's position in the provider's own emission order
    (2026-07-30 reading-order fix): the fine-tuned model emits records in
    reading order (column-aware, tables in place), which a bbox-center sort
    provably destroys on multi-column/table pages. Providers that have a
    meaningful emission order stamp it; ``None`` (other providers) keeps the
    legacy type-bucketed assembly byte-identical.
    """

    text: str
    bbox: list[float]  # [x0, y0, x1, y1] in page points
    confidence: float | None = None
    seq: int | None = None


@dataclass
class OCRTableCell:
    """One cell of a table detected by OCR."""

    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    bbox: list[float] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class OCRTable:
    """A table detected by OCR — rows × cols of cells, plus the table bbox."""

    cells: list[OCRTableCell]
    bbox: list[float]
    n_rows: int
    n_cols: int
    confidence: float | None = None
    # Transient (never serialized): set True by the cell-grid sidecar for ANY
    # table that keeps its markdown-derived cells on a ``cell_grid`` request —
    # ladder failure, over-cap, or degenerate sliver. The chunk builder reads it
    # to echo table_output_format="markdown" on that one table, so the response
    # marks exactly which tables fell back. Unconditional on every surface (an
    # unmarked fallback would be stamped "cell_grid" and read as true per-cell
    # coordinates). Default (markdown) runs never set it.
    cell_grid_degraded: bool = False
    # Provider emission-order index (see OCRBlock.seq).
    seq: int | None = None


@dataclass
class OCRFigure:
    """A figure region detected by OCR (e.g. logo, chart, diagram)."""

    bbox: list[float]
    confidence: float | None = None
    # Provider emission-order index (see OCRBlock.seq).
    seq: int | None = None


@dataclass
class OCRKeyValue:
    """A key-value form region (attribute panel / boxed field group). ``text`` is
    the pinned line grammar (``Key: Value`` per line, ``Key: <empty>`` blank,
    ``[x]``/``[ ]`` checkbox lines, optional leading title line); ``bbox`` is the
    region box. Per-pair boxes are a v2 concern — v1 is region-level.
    See .claude/skills/data-curation/references/kv-region-contract.md."""

    text: str
    bbox: list[float]  # [x0, y0, x1, y1] in page points — the region box
    confidence: float | None = None
    # Provider emission-order index (see OCRBlock.seq).
    seq: int | None = None


@dataclass
class OCRPageResult:
    """What an OCR provider returns for one page: prose blocks, tables, figures,
    and key-value regions. Checkbox / selection marks ride inside block or KV-region
    text as ``[x]``/``[ ]`` glyphs — there is no separate selection element."""

    blocks: list[OCRBlock] = field(default_factory=list)
    tables: list[OCRTable] = field(default_factory=list)
    figures: list[OCRFigure] = field(default_factory=list)
    key_values: list[OCRKeyValue] = field(default_factory=list)
    # True when the provider's decode hit the token cap mid-output (page incomplete):
    # the pipeline treats a truncated page as unusable and routes it to the fallback.
    truncated: bool = False
    # Optional raw provider response (envelope dict or raw model text) kept ONLY
    # for internal diagnostics — e.g. the Gemini-fallback review bundle. The
    # customer-facing extraction response never reads this field (it serializes
    # blocks/tables/figures), and it must never reach logs or metrics: raw model
    # text is page content (PHI). Providers set it best-effort; ``None`` when the
    # provider did not (or could not) expose it.
    raw: object | None = None


class OCRProvider(Protocol):
    """Every OCR backend implements this. Providers receive a rendered page
    image and the page's dimensions and return text blocks plus any detected
    tables as an ``OCRPageResult``.
    """

    name: str

    async def ocr_page(
        self,
        image_bytes: bytes,
        *,
        page_width: float,
        page_height: float,
    ) -> OCRPageResult: ...
