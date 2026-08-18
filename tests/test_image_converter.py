"""Unit tests for ``extract.core.converters.image``.

The converter wraps raster images (PNG/JPEG/WebP/TIFF/HEIC/BMP) into PDF
bytes so the standard PDF OCR pipeline handles them like scanned pages.
Fixtures are generated in-memory with Pillow so signatures are authentic.
"""

from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image

from extract.core.converters import image as image_mod
from extract.core.converters.image import convert_to_pdf
from extract.core.errors import DocumentTooLarge, UnsupportedInput
from extract.core.pdf import OCR_DEFAULT_DPI
from extract.observability.timing import StageTimer


def _encode(img: Image.Image, fmt: str, **save_kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _open_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=pdf_bytes, filetype="pdf")


def _expected_pts(pixels: int) -> float:
    return pixels * 72.0 / OCR_DEFAULT_DPI


# ---------------------------------------------------------------------------
# Happy paths, one per format
# ---------------------------------------------------------------------------


async def test_png_converts_to_one_page_pdf():
    data = _encode(Image.new("RGB", (400, 300), (200, 30, 30)), "PNG")
    pdf = await convert_to_pdf(data=data)
    with _open_pdf(pdf) as doc:
        assert len(doc) == 1
        assert doc[0].rect.width == pytest.approx(_expected_pts(400))
        assert doc[0].rect.height == pytest.approx(_expected_pts(300))


async def test_jpeg_exif_orientation_transposes_dimensions():
    exif = Image.Exif()
    exif[274] = 6  # rotate 90° CW on display — width/height swap
    data = _encode(Image.new("RGB", (300, 200), (5, 5, 5)), "JPEG", exif=exif)
    pdf = await convert_to_pdf(data=data)
    with _open_pdf(pdf) as doc:
        assert len(doc) == 1
        assert doc[0].rect.width == pytest.approx(_expected_pts(200))
        assert doc[0].rect.height == pytest.approx(_expected_pts(300))


async def test_transparent_png_flattens_onto_white():
    # Fully transparent red — a naive embed would render red or black.
    data = _encode(Image.new("RGBA", (50, 50), (255, 0, 0, 0)), "PNG")
    pdf = await convert_to_pdf(data=data)
    with _open_pdf(pdf) as doc:
        pix = doc[0].get_pixmap()
        assert pix.pixel(pix.width // 2, pix.height // 2) == (255, 255, 255)


async def test_static_webp_converts():
    data = _encode(Image.new("RGB", (120, 80), (10, 200, 100)), "WEBP")
    pdf = await convert_to_pdf(data=data)
    with _open_pdf(pdf) as doc:
        assert len(doc) == 1


async def test_bmp_converts():
    data = _encode(Image.new("RGB", (120, 80), (255, 255, 0)), "BMP")
    pdf = await convert_to_pdf(data=data)
    with _open_pdf(pdf) as doc:
        assert len(doc) == 1


async def test_multipage_tiff_converts_one_page_per_frame():
    frames = [Image.new("RGB", (200, 150), (i * 60 % 255, 100, 50)) for i in range(4)]
    data = _encode(frames[0], "TIFF", save_all=True, append_images=frames[1:])
    timer = StageTimer()
    pdf = await convert_to_pdf(data=data, timer=timer)
    with _open_pdf(pdf) as doc:
        assert len(doc) == 4
    assert timer.meta["image_input_format"] == "TIFF"
    assert timer.meta["image_input_frames"] == 4
    assert timer.meta["image_input_max_frame_pixels"] == 200 * 150
    assert timer.meta["image_input_total_pixels"] == 4 * 200 * 150


async def test_heic_converts():
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    try:
        data = _encode(Image.new("RGB", (400, 300), (120, 60, 180)), "HEIF")
    except Exception as e:  # pragma: no cover - encoder not built locally
        pytest.skip(f"pillow-heif cannot encode a HEIC fixture here: {e}")
    pdf = await convert_to_pdf(data=data)
    with _open_pdf(pdf) as doc:
        assert len(doc) == 1


async def test_grayscale_and_palette_modes_convert():
    gray = _encode(Image.new("L", (60, 60), 128), "PNG")
    palette = _encode(Image.new("RGB", (60, 60), (1, 2, 3)).convert("P"), "PNG")
    bilevel = _encode(Image.new("1", (60, 60), 1), "PNG")
    for data in (gray, palette, bilevel):
        with _open_pdf(await convert_to_pdf(data=data)) as doc:
            assert len(doc) == 1


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


async def test_oversize_frame_pixels_rejected(monkeypatch):
    monkeypatch.setattr(image_mod, "IMAGE_INPUT_MAX_FRAME_PIXELS", 100)
    data = _encode(Image.new("RGB", (50, 50)), "PNG")  # 2500 px > 100
    with pytest.raises(DocumentTooLarge, match="frame limit"):
        await convert_to_pdf(data=data)


async def test_frame_count_over_page_limit_rejected_during_conversion(monkeypatch):
    # The pixel caps don't bound frame COUNT (tiny frames are ~free pixels);
    # the page limit must fire inside the conversion loop, not after all
    # frames have been decoded and embedded.
    import extract.core.pdf as pdf_mod
    from extract.core.errors import PageLimitExceeded

    monkeypatch.setattr(pdf_mod, "PAGE_LIMIT", 5)
    frames = [Image.new("L", (1, 1), i % 255) for i in range(8)]
    data = _encode(frames[0], "TIFF", save_all=True, append_images=frames[1:])
    with pytest.raises(PageLimitExceeded, match="frames"):
        await convert_to_pdf(data=data)


async def test_oversize_total_pixels_rejected(monkeypatch):
    monkeypatch.setattr(image_mod, "IMAGE_INPUT_MAX_TOTAL_PIXELS", 5000)
    frames = [Image.new("RGB", (50, 50)) for _ in range(3)]  # 7500 px total
    data = _encode(frames[0], "TIFF", save_all=True, append_images=frames[1:])
    with pytest.raises(DocumentTooLarge, match="total pixels"):
        await convert_to_pdf(data=data)


async def test_converted_pdf_over_max_size_rejected():
    # Lossy WebP of noise is tiny; the PNG-backed PDF re-encode is much
    # bigger. Pick a cap between the two so only the OUTPUT trips it.
    import random

    rng = random.Random(42)
    noise = Image.new("RGB", (200, 200))
    noise.putdata(
        [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(200 * 200)]
    )
    data = _encode(noise, "WEBP", quality=10)
    pdf = await convert_to_pdf(data=data)
    assert len(data) < len(pdf)
    cap = len(pdf) - 1
    assert len(data) < cap
    with pytest.raises(DocumentTooLarge, match="Converted image PDF"):
        await convert_to_pdf(data=data, max_size=cap)


async def test_animated_webp_rejected():
    frames = [Image.new("RGB", (40, 40), (i * 40, 0, 0)) for i in range(4)]
    data = _encode(
        frames[0], "WEBP", save_all=True, append_images=frames[1:], duration=100
    )
    with pytest.raises(UnsupportedInput, match="Animated"):
        await convert_to_pdf(data=data)


async def test_gif_rejected_even_when_static():
    data = _encode(Image.new("RGB", (40, 40)), "GIF")
    with pytest.raises(UnsupportedInput, match="Unsupported image format"):
        await convert_to_pdf(data=data)


async def test_corrupt_image_bytes_rejected():
    with pytest.raises(UnsupportedInput):
        await convert_to_pdf(data=b"\x89PNG\r\n\x1a\n" + b"garbage" * 16)


async def test_non_image_bytes_rejected():
    with pytest.raises(UnsupportedInput):
        await convert_to_pdf(data=b"this is not an image at all")


async def test_source_bytes_over_max_size_rejected():
    data = _encode(Image.new("RGB", (400, 300)), "PNG")
    with pytest.raises(DocumentTooLarge):
        await convert_to_pdf(data=data, max_size=len(data) - 1)
