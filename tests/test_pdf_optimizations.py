import asyncio
import io

import pymupdf
from PIL import Image

from extract.core.models import ChunkType, ExtractRequest
from extract.core.ocr.base import (
    OCRBlock,
    OCRFigure,
    OCRKeyValue,
    OCRPageResult,
    OCRTable,
    OCRTableCell,
)
from extract.core.pdf import (
    OCR_DEFAULT_DPI,
    OCR_MAX_IMAGE_PIXELS,
    PageOcrSignals,
    _finalize_image_bytes,
    _is_table_like_vector_candidate,
    _merge_fragmented_text_chunks,
    _needs_ocr,
    _ocr_raster_zoom,
    _rasterize_page_for_ocr,
    _scan_fast_path,
    _text_chunk,
    extract_pdf,
)
from extract.storage.inline import InlineStorage


def test_scan_fast_path_without_text_uses_single_large_image():
    fast = _scan_fast_path(
        has_text=False,
        image_list=[(7, 0, 5712, 4284)],
        page_rect=pymupdf.Rect(0, 0, 612, 816),
    )
    assert fast == (7, 0, 5712, 4284)


def test_scan_fast_path_rejects_pages_with_text():
    fast = _scan_fast_path(
        has_text=True,
        image_list=[(7, 0, 5712, 4284)],
        page_rect=pymupdf.Rect(0, 0, 612, 816),
    )
    assert fast is None


def test_scan_fast_path_allows_dominant_page_image_with_small_decoration():
    fast = _scan_fast_path(
        has_text=False,
        image_list=[(7, 0, 2284, 3096), (8, 0, 479, 54)],
        page_rect=pymupdf.Rect(0, 0, 612, 816),
    )
    assert fast == (7, 0, 2284, 3096)


def test_default_ocr_raster_keeps_150_dpi_for_small_pages():
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=72, height=72)
        image_bytes = _rasterize_page_for_ocr(page)
    finally:
        doc.close()

    image = Image.open(io.BytesIO(image_bytes))
    assert OCR_DEFAULT_DPI == 150
    assert image.size == (150, 150)


def test_default_ocr_raster_caps_letter_page_to_2mp():
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=612, height=792)
        zoom = _ocr_raster_zoom(page)
        image_bytes = _rasterize_page_for_ocr(page)
    finally:
        doc.close()

    image = Image.open(io.BytesIO(image_bytes))
    assert OCR_MAX_IMAGE_PIXELS == 2_000_000
    assert image.width * image.height <= OCR_MAX_IMAGE_PIXELS
    assert image.size != (1275, 1650)
    assert 145 <= zoom * 72 <= 150


def test_finalize_image_bytes_keeps_large_embedded_image_direct_for_s3():
    original = b"x" * (600 * 1024)
    payload, mime, width, height = _finalize_image_bytes(
        {
            "img_bytes": original,
            "img_ext": "jpg",
            "has_mask": False,
            "width": 1200,
            "height": 900,
        },
        "s3",
    )

    assert payload == original
    assert mime == "image/jpeg"
    assert width == 1200
    assert height == 900


def test_ocr_classifier_detects_gibberish_text():
    signals = PageOcrSignals(
        page_area=100_000.0,
        chars_seen=100,
        replacement_chars=80,
    )

    assert _needs_ocr(signals=signals, mode="auto")


def test_ocr_classifier_skips_existing_glyphless_ocr_layer():
    signals = PageOcrSignals(
        page_area=100_000.0,
        text_chars=120,
        body_text_chars=120,
        glyphless_font_chars=120,
        image_area=100_000.0,
        dominant_image_area=100_000.0,
        image_rects=[pymupdf.Rect(0, 0, 200, 500)],
    )

    assert not _needs_ocr(signals=signals, mode="auto")


def test_vector_images_reject_text_heavy_form_grids():
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=360, height=300)
        for y in range(40, 220, 12):
            page.draw_line((30, y), (330, y), width=0.5)
        for x in range(30, 331, 60):
            page.draw_line((x, 40), (x, 220), width=0.5)
        for row in range(12):
            for col in range(4):
                page.insert_text((36 + col * 60, 50 + row * 12), f"F{row}-{col}", fontsize=6)
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    result = asyncio.run(
        extract_pdf(
            ExtractRequest(ocr="never", extract_images=True),
            data=pdf_bytes,
            storage=InlineStorage(),
        )
    )

    assert sum(1 for c in result.chunks if c.chunk_type == ChunkType.IMAGE) == 0


def test_vector_table_exclusion_allows_small_false_positive_inside_chart():
    chart = pymupdf.Rect(100, 100, 500, 240)
    small_false_table = pymupdf.Rect(145, 130, 305, 220)
    actual_table = pymupdf.Rect(110, 105, 490, 230)

    assert not _is_table_like_vector_candidate(chart, [small_false_table])
    assert _is_table_like_vector_candidate(chart, [actual_table])


def test_ocr_text_and_table_coexist(monkeypatch):
    # Under pure-OCR there is no native span text: the model supplies BOTH the
    # prose blocks and the tables. Verify an OCR result carrying a text block
    # plus a table yields both a TEXT chunk and a TABLE chunk.
    def _stub(_name):
        class _TextAndTableOCR:
            name = "stub"

            async def ocr_page(self, image_bytes, *, page_width, page_height):
                return OCRPageResult(
                    blocks=[
                        OCRBlock(text="Revenue", bbox=[40, 20, 160, 40], confidence=0.0),
                        OCRBlock(text="Cost", bbox=[40, 44, 160, 64], confidence=0.0),
                    ],
                    tables=[
                        OCRTable(
                            cells=[
                                OCRTableCell(text="Revenue", row=0, col=0, bbox=[40, 60, 160, 100], confidence=0.0),
                                OCRTableCell(text="100", row=0, col=1, bbox=[160, 60, 280, 100], confidence=0.0),
                            ],
                            bbox=[40, 60, 280, 140],
                            n_rows=1,
                            n_cols=2,
                            confidence=0.0,
                        )
                    ],
                )

        return _TextAndTableOCR()

    monkeypatch.setattr("extract.core.pdf.get_ocr_provider", _stub)

    doc = pymupdf.open()
    try:
        doc.new_page(width=320, height=220)
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    result = asyncio.run(
        extract_pdf(
            ExtractRequest(extract_images=False),
            data=pdf_bytes,
            storage=InlineStorage(),
        )
    )

    text = "\n".join(c.page_content for c in result.chunks if c.chunk_type == ChunkType.TEXT)
    tables = [c for c in result.chunks if c.chunk_type == ChunkType.TABLE]

    assert "Revenue" in text  # prose block from the OCR model
    assert "Cost" in text
    assert tables  # table supplied by the OCR model
    assert "Revenue" in tables[0].page_content


def test_ocr_kv_region_becomes_key_value_chunk(monkeypatch):
    # A KV region from the OCR model yields a KEY_VALUE chunk whose page_content is
    # the pinned line grammar verbatim (never overwritten by native-word text).
    def _stub(_name):
        class _KvOCR:
            name = "stub"

            async def ocr_page(self, image_bytes, *, page_width, page_height):
                return OCRPageResult(
                    blocks=[OCRBlock(text="# Facesheet", bbox=[40, 10, 280, 30], confidence=None)],
                    key_values=[
                        OCRKeyValue(
                            text="Next of Kin\nName: Karen Nolan\nWork Phone: <empty>\n[ ] POA",
                            bbox=[40, 40, 280, 180],
                            confidence=None,
                        )
                    ],
                )

        return _KvOCR()

    monkeypatch.setattr("extract.core.pdf.get_ocr_provider", _stub)

    doc = pymupdf.open()
    try:
        doc.new_page(width=320, height=220)
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    result = asyncio.run(
        extract_pdf(
            ExtractRequest(extract_images=False),
            data=pdf_bytes,
            storage=InlineStorage(),
        )
    )

    kv = [c for c in result.chunks if c.chunk_type == ChunkType.KEY_VALUE]
    assert len(kv) == 1
    assert kv[0].page_content.startswith("Next of Kin")
    assert "Work Phone: <empty>" in kv[0].page_content and "[ ] POA" in kv[0].page_content
    # The region text did NOT leak into a TEXT or TABLE chunk.
    assert all("Next of Kin" not in c.page_content for c in result.chunks if c.chunk_type != ChunkType.KEY_VALUE)


def test_fragmented_native_text_chunks_merge_touching_word_spans():
    chunks = [
        _text_chunk(text="This document de", page_no=1, bbox=[10, 10, 92, 20]),
        _text_chunk(text="\ufb01", page_no=1, bbox=[92, 10, 98, 20]),
        _text_chunk(text="nes transport", page_no=1, bbox=[98, 10, 170, 20]),
        _text_chunk(text="\ufb02", page_no=1, bbox=[10, 24, 16, 34]),
        _text_chunk(text="ow", page_no=1, bbox=[16, 24, 30, 34]),
        _text_chunk(text="control", page_no=1, bbox=[36, 24, 70, 34]),
    ]

    merged = _merge_fragmented_text_chunks(chunks)

    assert [c.page_content for c in merged] == [
        "This document defines transport",
        "flow",
        "control",
    ]
    assert merged[0].bbox == [10.0, 10.0, 170.0, 20.0]


def test_vector_images_can_be_disabled_for_bisecting():
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=320, height=240)
        page.insert_text((40, 38), "Text remains text.", fontsize=10)
        for i in range(10):
            page.draw_line((60, 130 + i * 5), (180, 120 + i * 4), width=0.8)
            page.draw_line((60 + i * 10, 180), (70 + i * 10, 120), width=0.8)
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    result = asyncio.run(
        extract_pdf(
            ExtractRequest(ocr="never", extract_images=True),
            data=pdf_bytes,
            storage=InlineStorage(),
            enable_vector_images=False,
        )
    )

    assert sum(1 for c in result.chunks if c.chunk_type == ChunkType.IMAGE) == 0


def test_pure_ocr_emits_text_table_and_image_chunks(monkeypatch):
    # The pure-OCR assembly path: the model returns one prose block, one table,
    # and one figure region for the page; extract_pdf must turn them into the
    # corresponding TEXT, TABLE, and IMAGE chunks.
    def _stub(_name):
        class _RichOCR:
            name = "stub"

            async def ocr_page(self, image_bytes, *, page_width, page_height):
                return OCRPageResult(
                    blocks=[
                        OCRBlock(text="Body paragraph from OCR.", bbox=[40, 20, 280, 48], confidence=0.0),
                    ],
                    tables=[
                        OCRTable(
                            cells=[
                                OCRTableCell(text="Revenue", row=0, col=0, bbox=[40, 60, 160, 100], confidence=0.0),
                                OCRTableCell(text="100", row=0, col=1, bbox=[160, 60, 280, 100], confidence=0.0),
                            ],
                            bbox=[40, 60, 280, 100],
                            n_rows=1,
                            n_cols=2,
                            confidence=0.0,
                        )
                    ],
                    figures=[OCRFigure(bbox=[40, 120, 280, 200], confidence=0.0)],
                )

        return _RichOCR()

    monkeypatch.setattr("extract.core.pdf.get_ocr_provider", _stub)

    doc = pymupdf.open()
    try:
        doc.new_page(width=320, height=220)
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    result = asyncio.run(
        extract_pdf(
            ExtractRequest(extract_images=True),
            data=pdf_bytes,
            storage=InlineStorage(),
        )
    )

    text_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.TEXT]
    table_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.TABLE]
    image_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.IMAGE]

    assert text_chunks
    assert "Body paragraph from OCR." in text_chunks[0].page_content
    assert table_chunks
    assert "Revenue" in table_chunks[0].page_content
    assert image_chunks
    assert image_chunks[0].bbox is not None
