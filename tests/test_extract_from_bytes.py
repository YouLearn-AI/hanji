"""Integration tests for ``Extractor.aextract_from_bytes``.

These tests read the in-repo fixture files into memory and assert the
bytes-based entry point produces the same page count and first-text-chunk
content as ``aextract_from_path``. They don't hit the network (no ``live``
marker) but do run the full extraction pipeline locally.

Office tests require LibreOffice (``soffice``) on the path; they skip
otherwise, the same way ``test_extract_live.py`` guards its office tests.
"""

from __future__ import annotations

from pathlib import Path
from shutil import which

import pytest

from extract.core import ChunkType, ExtractRequest, Extractor, UnsupportedInput

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_FIXTURE = REPO_ROOT / "web" / "public" / "demo" / "attention.pdf"
DOCX_FIXTURE = REPO_ROOT / "evals" / "corpus" / "yl" / "cache" / "test-docx.docx"
PPTX_FIXTURE = REPO_ROOT / "evals" / "corpus" / "yl" / "cache" / "test-pptx.pptx"


def _has_soffice() -> bool:
    return which("soffice") is not None


def _first_text(chunks) -> str | None:
    for c in chunks:
        if c.chunk_type == ChunkType.TEXT and c.page_content:
            return c.page_content
    return None


def _request() -> ExtractRequest:
    # Keep tests fast + storage-agnostic: no images, no OCR.
    return ExtractRequest(extract_images=False, ocr="never")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_pdf_bytes_matches_path():
    data = PDF_FIXTURE.read_bytes()

    by_bytes = await Extractor().aextract_from_bytes(
        data, filename=PDF_FIXTURE.name, request=_request()
    )
    by_path = await Extractor().aextract_from_path(
        str(PDF_FIXTURE), request=_request()
    )

    assert by_bytes.page_count == by_path.page_count
    assert by_bytes.page_count > 0
    assert _first_text(by_bytes.chunks) == _first_text(by_path.chunks)


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_pdf_bytes_ignores_misleading_filename():
    # Magic bytes wins: a `.docx` filename with PDF bytes must still
    # extract as a PDF and not be routed to LibreOffice.
    data = PDF_FIXTURE.read_bytes()
    res = await Extractor().aextract_from_bytes(
        data, filename="pretend.docx", request=_request()
    )
    assert res.page_count > 0


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_pdf_bytes_without_filename():
    # Browser / agent uploads that omit the filename still work when
    # magic bytes identify the format.
    data = PDF_FIXTURE.read_bytes()
    res = await Extractor().aextract_from_bytes(data, request=_request())
    assert res.page_count > 0


# ---------------------------------------------------------------------------
# Unsupported / garbage bytes
# ---------------------------------------------------------------------------


async def test_bytes_unknown_kind_raises():
    with pytest.raises(UnsupportedInput):
        await Extractor().aextract_from_bytes(
            b"not a real document", filename=None, request=_request()
        )


async def test_bytes_empty_raises():
    with pytest.raises(UnsupportedInput):
        await Extractor().aextract_from_bytes(
            b"", filename=None, request=_request()
        )


# ---------------------------------------------------------------------------
# Office — require soffice
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_soffice(), reason="soffice not installed")
@pytest.mark.skipif(not DOCX_FIXTURE.exists(), reason=f"fixture missing: {DOCX_FIXTURE}")
async def test_docx_bytes_matches_path():
    data = DOCX_FIXTURE.read_bytes()

    by_bytes = await Extractor().aextract_from_bytes(
        data, filename=DOCX_FIXTURE.name, request=_request()
    )
    by_path = await Extractor().aextract_from_path(
        str(DOCX_FIXTURE), request=_request()
    )

    assert by_bytes.page_count == by_path.page_count
    assert by_bytes.page_count > 0
    assert _first_text(by_bytes.chunks) == _first_text(by_path.chunks)


@pytest.mark.skipif(not _has_soffice(), reason="soffice not installed")
@pytest.mark.skipif(not PPTX_FIXTURE.exists(), reason=f"fixture missing: {PPTX_FIXTURE}")
async def test_pptx_bytes_matches_path():
    data = PPTX_FIXTURE.read_bytes()

    by_bytes = await Extractor().aextract_from_bytes(
        data, filename=PPTX_FIXTURE.name, request=_request()
    )
    by_path = await Extractor().aextract_from_path(
        str(PPTX_FIXTURE), request=_request()
    )

    assert by_bytes.page_count == by_path.page_count
    assert by_bytes.page_count > 0
    assert _first_text(by_bytes.chunks) == _first_text(by_path.chunks)
