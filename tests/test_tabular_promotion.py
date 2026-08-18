"""Pipe-tabular block promotion (`extract.core.ocr.tabular_promotion`).

The OCR champion frequently serializes a real table as pipe-delimited text
WITHOUT the GFM delimiter row, so it was typed ``text`` and shipped without
``cells``. Promotion deterministically re-types such blocks on every parse.
Receipts for the behavior change live in plan 058 (extract-bench paired
no-reg + typed-table recovery 17→23/24 on a production slice).
"""

from __future__ import annotations

from extract.core.models import ExtractRequest
from extract.core.ocr.base import OCRBlock, OCRPageResult, OCRTable, OCRTableCell
from extract.core.ocr.tabular_promotion import _pipe_rows, promote_tabular_blocks


def test_pipe_rows_parses_delimiterless_tabular_text():
    text = (
        "| Patient Name | Bowen, Michael | Gender | MALE |\n"
        "| DOB | 09/29/1958 (67 yr) |  |  |\n"
        "| Home Phone | 555-0101 | MRN | 3728 |"
    )
    rows = _pipe_rows(text)
    assert rows is not None and len(rows) == 3
    assert rows[0][0] == "Patient Name"


def test_pipe_rows_rejects_prose_and_single_rows():
    assert _pipe_rows("Plain prose.\nMore prose lines here.") is None
    assert _pipe_rows("| one | row |") is None  # single line — not tabular
    # Majority-prose block with two incidental pipe lines stays prose.
    assert (
        _pipe_rows(
            "prose line one\nprose line two\nprose three\nprose four\n"
            "| a | b |\n| c | d |"
        )
        is None
    )


def test_pipe_rows_ignores_delimiter_rows_in_the_content_count():
    """A GFM delimiter row must not buy a block its promotion.

    Regression for a production table-count flip (2026-08-06): a
    header + ``| --- |`` block carries ONE content row, so it is not tabular —
    but the delimiter row used to count toward ``_MIN_PIPE_ROWS``, so the block
    promoted only on the runs where the model happened to emit the separator,
    flipping the chunk type of an unchanged document between identical requests.

    REVERT-PROOF: against the unfixed ``_pipe_rows`` (``pipe_lines <
    _MIN_PIPE_ROWS``) the first assertion below fails — the header+delimiter
    block parses to 2 rows and promotes.
    """
    # Header + delimiter only: one content row → NOT tabular.
    assert _pipe_rows("| Item | Amount |\n| --- | --- |") is None
    # Alignment-marker delimiters (``:---:``) are delimiter rows too.
    assert _pipe_rows("| Item | Amount |\n| :--- | ---: |") is None
    # Header + delimiter + data rows → tabular (the delimiter is still parsed).
    rows = _pipe_rows(
        "| Item | Amount |\n| --- | --- |\n| Widget | 12 |\n| Gadget | 7 |"
    )
    assert rows is not None
    assert ["Widget", "12"] in rows and ["Gadget", "7"] in rows
    # A real delimiter-less table is unaffected by the content-count change.
    assert _pipe_rows("| Item | Amount |\n| Widget | 12 |") is not None


def test_promote_tabular_blocks_leaves_header_delimiter_block_as_text():
    """The production shape: a stray header+delimiter block stays a TEXT chunk
    instead of becoming a phantom table that swallows the following text."""
    page = OCRPageResult(
        blocks=[
            OCRBlock(text="| Charge Description | Amount |\n| --- | --- |",
                     bbox=[10, 20, 500, 60]),
        ],
    )
    assert promote_tabular_blocks(page) == 0
    assert page.tables == []
    assert len(page.blocks) == 1


def test_promote_tabular_blocks_types_pipe_text_and_keeps_prose():
    page = OCRPageResult(
        blocks=[
            OCRBlock(text="Patient Information - Bowen, Michael", bbox=[10, 5, 300, 15]),
            OCRBlock(
                text=(
                    "| Patient Name | Bowen, Michael | Gender | MALE |\n"
                    "| DOB | 09/29/1958 (67 yr) |  |  |"
                ),
                bbox=[10, 20, 500, 60],
            ),
        ],
    )
    assert promote_tabular_blocks(page) == 1
    assert len(page.tables) == 1
    assert len(page.blocks) == 1  # the prose header stays a block
    assert page.blocks[0].text.startswith("Patient Information")
    table = page.tables[0]
    assert table.n_rows == 2 and table.n_cols == 4
    assert {c.text for c in table.cells} >= {"Patient Name", "Bowen, Michael", "DOB"}
    # Markdown-derived semantics: cells share the block bbox.
    assert all(c.bbox == [10, 20, 500, 60] for c in table.cells)


def test_promote_tabular_blocks_ignores_boxless_and_typed_tables():
    typed = OCRTable(
        cells=[OCRTableCell(text="x", row=0, col=0, bbox=[1, 1, 2, 2])],
        bbox=[1, 1, 2, 2], n_rows=1, n_cols=1,
    )
    page = OCRPageResult(
        blocks=[OCRBlock(text="| a | b |\n| c | d |", bbox=[])],  # no bbox — kept
        tables=[typed],
    )
    assert promote_tabular_blocks(page) == 0
    assert len(page.blocks) == 1
    assert page.tables == [typed]


def _one_page_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


async def test_extract_pdf_promotes_pipe_tables_end_to_end(monkeypatch):
    """A delimiter-less pipe block ships as a TYPED table chunk with cells and
    a canonical GFM render (delimiter row added) through the full pipeline."""
    from extract.core import pdf as pdf_module
    from extract.observability.timing import StageTimer

    class _PipeBlockOCR:
        name = "stub-pipe"

        async def ocr_page(self, image_bytes, *, page_width, page_height):
            return OCRPageResult(
                blocks=[
                    OCRBlock(
                        text=(
                            "| Patient Name | Bowen, Michael |\n"
                            "| DOB | 09/29/1958 (67 yr) |"
                        ),
                        bbox=[50.0, 50.0, 550.0, 120.0],
                    )
                ]
            )

    monkeypatch.setattr(pdf_module, "get_ocr_provider", lambda _name: _PipeBlockOCR())
    timer = StageTimer()
    response = await pdf_module.extract_pdf(
        ExtractRequest(), data=_one_page_pdf(), timer=timer
    )
    tables = [c for c in response.chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    assert tables[0].n_rows == 2 and tables[0].n_cols == 2
    assert "| --- |" in tables[0].page_content  # canonical GFM delimiter row
    assert {c.text for c in tables[0].cells} == {
        "Patient Name", "Bowen, Michael", "DOB", "09/29/1958 (67 yr)",
    }
    assert timer.meta["tables_pipe_promoted"] == 1


async def test_extract_pdf_prose_pages_emit_no_promotion_metric(monkeypatch):
    from extract.core import pdf as pdf_module
    from extract.observability.timing import StageTimer

    class _ProseOCR:
        name = "stub-prose"

        async def ocr_page(self, image_bytes, *, page_width, page_height):
            return OCRPageResult(
                blocks=[OCRBlock(text="Just a paragraph.", bbox=[50, 50, 550, 80])]
            )

    monkeypatch.setattr(pdf_module, "get_ocr_provider", lambda _name: _ProseOCR())
    timer = StageTimer()
    response = await pdf_module.extract_pdf(
        ExtractRequest(), data=_one_page_pdf(), timer=timer
    )
    assert [c.chunk_type.value for c in response.chunks] == ["text"]
    assert "tables_pipe_promoted" not in timer.meta
