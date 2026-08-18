"""Pipe-tabular block promotion — recover tables the parser typed as TEXT.

The OCR champion frequently reads a table correctly but serializes it as
pipe-delimited text WITHOUT the GFM delimiter row (``|---|---|``), so
``_table_from_markdown`` never types it as a table and the block ships as a
plain text chunk — no ``cells``, no ``n_rows``/``n_cols``. Measured on the
production slice: 3 of 6 pages carried ALL their tables in that shape (typed
table coverage 17→23 of 24 gold tables with promotion; grits_top 0.145→0.392).

This module deterministically re-types such blocks after the page's OCR result
is served: a block whose text is pipe-tabular (>= 2 pipe rows, each >= 2 cells,
pipe rows the majority of the block) becomes an ``OCRTable`` whose cells carry
the production text and share the block bbox — exactly the markdown-derived
table semantics — and its ``page_content`` re-renders as canonical GFM.

The strict GFM delimiter-row check in ``_looks_like_markdown_table`` stays as-is
for single-shot typing; the guards HERE are what keep prose out of promotion.
No-reg receipts (2026-07-07, prod OCR endpoint, paired 130-page extract-bench):
fire rate 1/130 pages; the affected page improved (text_accuracy +0.217); all
other deltas within the measured serving-noise floor. See plan 058.
"""

from __future__ import annotations

from typing import Any

from extract.core.ocr.base import OCRTable, OCRTableCell

#: A block must contain at least this many pipe rows to be tabular.
_MIN_PIPE_ROWS = 2


def _pipe_rows(text: str) -> list[list[str]] | None:
    """Parse a pipe-tabular block into rows of cells, or ``None`` if the text
    is not pipe-tabular (needs >= ``_MIN_PIPE_ROWS`` CONTENT pipe rows, each
    splitting into >= 2 cells, and a majority of non-empty lines being pipe rows).

    A GFM delimiter row (``| --- | --- |``) is parsed like any other row — it
    still counts as a pipe row for the majority test — but it is excluded from
    the CONTENT count that gates promotion. Counting it made promotion depend on
    whether the model happened to emit a separator, flipping the chunk type of an
    unchanged document between identical requests (observed in production
    2026-08-06 production evaluation, 2/10 then 9/40 runs).
    """
    from extract.core.ocr.qwen_lora import _is_delimiter_row, _split_markdown_row

    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < _MIN_PIPE_ROWS:
        return None
    rows: list[list[str]] = []
    pipe_lines = 0
    content_pipe_lines = 0
    for ln in lines:
        if ln.count("|") >= 2:
            cells = _split_markdown_row(ln)
            if len(cells) >= 2:
                pipe_lines += 1
                if not _is_delimiter_row(ln):
                    content_pipe_lines += 1
                rows.append(cells)
                continue
        rows.append([ln.strip()])
    if content_pipe_lines < _MIN_PIPE_ROWS or pipe_lines * 2 < len(lines):
        return None
    return rows


def promote_tabular_blocks(page_result: Any) -> int:
    """Promote pipe-tabular TEXT blocks to typed tables, in place.

    Every promoted table carries the production text as its cell grid (cells
    share the block bbox, exactly like a markdown-derived table). Runs on the
    served ``OCRPageResult`` for every parse; prose blocks are untouched.
    Returns the number of blocks promoted (telemetry:
    ``timer.meta["tables_pipe_promoted"]``)."""
    blocks = list(getattr(page_result, "blocks", None) or [])
    promoted = 0
    kept_blocks = []
    for block in blocks:
        rows = _pipe_rows(getattr(block, "text", "") or "")
        if rows is None or not getattr(block, "bbox", None):
            kept_blocks.append(block)
            continue
        n_cols = max(len(r) for r in rows)
        cells = [
            OCRTableCell(
                text=cell_text,
                row=ri,
                col=ci,
                bbox=list(block.bbox),
                confidence=getattr(block, "confidence", None),
            )
            for ri, r in enumerate(rows)
            for ci, cell_text in enumerate(r)
        ]
        page_result.tables.append(
            OCRTable(
                cells=cells,
                bbox=list(block.bbox),
                n_rows=len(rows),
                n_cols=n_cols,
                confidence=getattr(block, "confidence", None),
                # Promotion must not move the table in reading order: the
                # promoted table keeps the source block's emission position.
                seq=getattr(block, "seq", None),
            )
        )
        promoted += 1
    if promoted:
        page_result.blocks = kept_blocks
    return promoted
