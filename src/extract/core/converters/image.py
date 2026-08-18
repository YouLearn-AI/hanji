"""Raster image → PDF via Pillow + PyMuPDF.

PNG / JPEG / WebP / TIFF / HEIC-HEIF / BMP inputs are normalized with
Pillow (EXIF orientation, alpha flattened onto white, mode conversion to
RGB or L) and wrapped into a PDF — one page per frame — so the standard
PDF OCR pipeline treats them like scanned pages. Deliberately a converter,
not a second extraction engine: auth, billing, OCR, fallback handling,
schema extraction, async batch, and observability all stay on the existing
PDF path.

Page geometry uses ``pixels * 72 / OCR_DEFAULT_DPI`` points so the OCR
rasterizer renders the page at approximately the source's native pixel
dimensions instead of accidentally upsampling.
"""

from __future__ import annotations

import asyncio
import io

import pymupdf
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from extract.core.errors import DocumentTooLarge, PageLimitExceeded, UnsupportedInput
from extract.core.io import load_bytes
from extract.observability.timing import StageTimer

# Decoded-pixel guardrails, checked per frame and across all frames before
# any PDF page is built. Above either cap the request fails with a 413.
IMAGE_INPUT_MAX_FRAME_PIXELS = 50_000_000
IMAGE_INPUT_MAX_TOTAL_PIXELS = 250_000_000

# Pillow can decode more than we accept (GIF, ICO, …). Only the formats the
# kind detector routes here are allowed; anything else fails closed even if
# it slipped in via a misnamed extension.
_ACCEPTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "TIFF", "HEIF", "BMP", "MPO"})

# MPO is a JPEG carrying more than one image: iPhones emit it for HDR/portrait
# shots, where frame 0 is the full-resolution photo and the extra frame is a
# half-resolution grayscale gain map. Pillow reports format "MPO" (not "JPEG")
# and sets is_animated=True, so before this it was refused twice over — a real
# user's iPhone photo of a document was rejected in 0ms as an unsupported
# format (2026-07-29). Treat it as a single-frame JPEG: decode frame 0 only.
# The auxiliary frame is NOT a page — rendering it would both corrupt the
# output and bill an extra page.
_SINGLE_FRAME_FORMATS = frozenset({"MPO"})

_JPEG_QUALITY = 95

# PDF pages below ~3pt are rejected by viewers/MuPDF; clamp tiny images up.
_MIN_PAGE_PTS = 3.0

_heif_registered = False


def _ensure_heif_registered() -> None:
    """Idempotently register the pillow-heif opener with Pillow."""
    global _heif_registered
    if not _heif_registered:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _heif_registered = True


def _normalize_frame(frame: Image.Image) -> tuple[Image.Image, bool]:
    """Return an embeddable RGB/L frame and whether PNG must be used.

    Applies EXIF orientation, flattens transparency onto white (a document
    page is a white sheet, not transparent ink over a default background),
    and converts palette/bilevel/16-bit/CMYK modes.
    """
    frame = ImageOps.exif_transpose(frame)
    force_png = False
    if frame.mode == "P":
        frame = frame.convert("RGBA" if "transparency" in frame.info else "RGB")
        force_png = True
    if frame.mode in ("RGBA", "LA", "PA"):
        rgba = frame.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        frame = Image.alpha_composite(background, rgba).convert("RGB")
        force_png = True
    elif frame.mode == "1":
        frame = frame.convert("L")
    elif frame.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
        # 16-bit grayscale: scale to 8-bit instead of convert("L")'s truncation.
        frame = frame.convert("I").point(lambda v: v / 256).convert("L")
    elif frame.mode == "F":
        frame = frame.convert("L")
    elif frame.mode not in ("L", "RGB"):
        # CMYK, YCbCr, LAB, HSV, …
        frame = frame.convert("RGB")
    if frame.mode == "L":
        force_png = True
    return frame, force_png


def _encode_frame(frame: Image.Image, *, force_png: bool) -> bytes:
    """PNG for bilevel/gray/palette/transparent/low-color frames; JPEG q95
    for opaque photographic RGB so large camera scans don't balloon into
    enormous PNG-backed PDFs."""
    buf = io.BytesIO()
    if force_png or frame.getcolors(256) is not None:
        frame.save(buf, format="PNG")
    else:
        frame.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _convert_sync(input_bytes: bytes, timer: StageTimer | None) -> bytes:
    from extract.core.pdf import OCR_DEFAULT_DPI, PAGE_LIMIT

    _ensure_heif_registered()
    try:
        img = Image.open(io.BytesIO(input_bytes))
    except Image.DecompressionBombError as e:
        raise DocumentTooLarge(f"Image is too large to decode: {e}") from e
    except UnidentifiedImageError as e:
        raise UnsupportedInput("Bytes are not a decodable image.") from e
    except (OSError, ValueError, SyntaxError) as e:
        raise UnsupportedInput(f"Could not decode image: {e}") from e

    with img:
        fmt = (img.format or "").upper()
        if fmt not in _ACCEPTED_FORMATS:
            raise UnsupportedInput(
                f"Unsupported image format: {fmt or 'unknown'}. "
                "Supported: PNG, JPEG, WebP, TIFF, HEIC/HEIF, BMP."
            )
        # Multi-frame is a TIFF-only feature (one PDF page per frame).
        # Animated GIF/WebP/APNG need a frame-selection + billing decision
        # we have deliberately not made — fail closed.
        if (
            fmt != "TIFF"
            and fmt not in _SINGLE_FRAME_FORMATS
            and getattr(img, "is_animated", False)
        ):
            raise UnsupportedInput(
                "Animated images are not supported. Send a static image."
            )

        doc = pymupdf.open()
        try:
            n_frames = 0
            total_pixels = 0
            max_frame_pixels = 0
            try:
                frames = (
                    [img] if fmt in _SINGLE_FRAME_FORMATS else ImageSequence.Iterator(img)
                )
                for raw_frame in frames:
                    # Enforce the page limit DURING conversion: the pixel caps
                    # don't bound frame count (1×1 frames are ~free pixels but
                    # each pays a decode + encode + PDF page), so a
                    # million-frame TIFF must fail here, not after minutes of
                    # work at extract_pdf's downstream check.
                    if n_frames >= PAGE_LIMIT:
                        raise PageLimitExceeded(
                            f"Image has more than {PAGE_LIMIT} frames, "
                            f"exceeds the {PAGE_LIMIT}-page limit."
                        )
                    width, height = raw_frame.size
                    if width <= 0 or height <= 0:
                        raise UnsupportedInput(
                            f"Image frame {n_frames + 1} has zero width or height."
                        )
                    pixels = width * height
                    if pixels > IMAGE_INPUT_MAX_FRAME_PIXELS:
                        raise DocumentTooLarge(
                            f"Image frame is {pixels} pixels, exceeds the "
                            f"{IMAGE_INPUT_MAX_FRAME_PIXELS}-pixel frame limit."
                        )
                    total_pixels += pixels
                    if total_pixels > IMAGE_INPUT_MAX_TOTAL_PIXELS:
                        raise DocumentTooLarge(
                            f"Image decodes to over {IMAGE_INPUT_MAX_TOTAL_PIXELS} "
                            "total pixels across frames."
                        )
                    max_frame_pixels = max(max_frame_pixels, pixels)

                    frame, force_png = _normalize_frame(raw_frame)
                    stream = _encode_frame(frame, force_png=force_png)
                    # EXIF orientation may have swapped the dimensions.
                    width, height = frame.size

                    page = doc.new_page(
                        width=max(width * 72.0 / OCR_DEFAULT_DPI, _MIN_PAGE_PTS),
                        height=max(height * 72.0 / OCR_DEFAULT_DPI, _MIN_PAGE_PTS),
                    )
                    page.insert_image(page.rect, stream=stream)
                    n_frames += 1
            except Image.DecompressionBombError as e:
                raise DocumentTooLarge(f"Image is too large to decode: {e}") from e
            except (OSError, ValueError, SyntaxError, EOFError) as e:
                raise UnsupportedInput(f"Could not decode image: {e}") from e

            if n_frames == 0:
                raise UnsupportedInput("Image contains no decodable frames.")

            if timer is not None:
                timer.meta["image_input_format"] = fmt
                timer.meta["image_input_frames"] = n_frames
                timer.meta["image_input_max_frame_pixels"] = max_frame_pixels
                timer.meta["image_input_total_pixels"] = total_pixels

            return doc.tobytes(garbage=4, deflate=True)
        finally:
            doc.close()


def convert_bytes_to_pdf(data: bytes, *, timer: StageTimer | None = None) -> bytes:
    """Blocking twin of :func:`convert_to_pdf`, for callers already on a thread.

    Same conversion, same guardrails, same exceptions — it exists so a caller
    that is ALREADY off the event loop (the API's page-render prefetch, which
    runs in a worker thread) can reuse this converter instead of growing a
    second, subtly different image decode path. Nothing about the output
    differs: the pixels a caller renders from this PDF are the pixels the parse
    pipeline read, EXIF rotation, flattened alpha, HEIC/MPO frame choice and all.
    """
    return _convert_sync(data, timer)


async def convert_to_pdf(
    *,
    url: str | None = None,
    path: str | None = None,
    data: bytes | None = None,
    max_size: int | None = None,
    timer: StageTimer | None = None,
    block_private: bool = True,
) -> bytes:
    """Convert raster image bytes (URL / path / data) into PDF bytes.

    Raises ``UnsupportedInput`` for undecodable/animated/disallowed images
    and ``DocumentTooLarge`` when the source bytes, decoded pixels, or the
    converted PDF exceed their caps.
    """
    input_bytes = await load_bytes(
        url=url, path=path, data=data, max_size=max_size, block_private=block_private
    )
    pdf_bytes = await asyncio.to_thread(_convert_sync, input_bytes, timer)
    if max_size is not None and len(pdf_bytes) > max_size:
        raise DocumentTooLarge(
            f"Converted image PDF is {len(pdf_bytes)} bytes, exceeds max_size={max_size}."
        )
    return pdf_bytes
