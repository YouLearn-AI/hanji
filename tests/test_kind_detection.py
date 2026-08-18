"""Unit tests for ``extract.core.kind``.

Covers the three detector entry points:

- ``detect_kind_from_name`` — extension sniffing on URLs, paths, filenames.
- ``detect_kind_from_magic`` — first-few-bytes sniff (PDF, OOXML archives,
  raster-image signatures).
- ``detect_kind`` — the composite used by callers; magic wins on conflict,
  filename fills in when magic is inconclusive.
"""

from __future__ import annotations

import io
import zipfile

from extract.core.kind import (
    detect_kind,
    detect_kind_from_magic,
    detect_kind_from_name,
)
from extract.core.models import InputKind


def _make_zip(entries: list[tuple[str, bytes]]) -> bytes:
    """Build a minimal in-memory ZIP with the given entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in entries:
            zf.writestr(name, payload)
    return buf.getvalue()


PDF_MAGIC = b"%PDF-1.7\n%..."
# Real, minimally valid OOXML archives — a PK local file header plus a
# central directory listing an office-specific entry. `detect_kind_from_magic`
# reads the central directory to identify them.
DOCX_MAGIC = _make_zip([
    ("[Content_Types].xml", b"<xml/>"),
    ("word/document.xml", b"<w:document/>"),
])
PPTX_MAGIC = _make_zip([
    ("[Content_Types].xml", b"<xml/>"),
    ("ppt/presentation.xml", b"<p:presentation/>"),
])
# Valid ZIP with neither office marker — the fallback path should treat
# this as inconclusive and defer to the filename detector.
AMBIGUOUS_ZIP = _make_zip([("hello.txt", b"hi")])


# ---------------------------------------------------------------------------
# detect_kind_from_name
# ---------------------------------------------------------------------------


def test_name_plain_filenames():
    assert detect_kind_from_name("report.pdf") == InputKind.PDF
    assert detect_kind_from_name("deck.pptx") == InputKind.PPTX
    assert detect_kind_from_name("letter.docx") == InputKind.DOCX


def test_name_case_insensitive():
    assert detect_kind_from_name("REPORT.PDF") == InputKind.PDF
    assert detect_kind_from_name("Deck.PptX") == InputKind.PPTX


def test_name_legacy_doc_routes_to_docx_path():
    # LibreOffice handles .doc via the same convert route as .docx; the
    # detector maps the extension to the DOCX pipeline.
    assert detect_kind_from_name("letter.doc") == InputKind.DOCX


def test_name_alternate_office_extensions():
    assert detect_kind_from_name("slides.ppsx") == InputKind.PPTX


def test_name_url_with_query_string():
    url = "https://example.com/content/paper.pdf?signature=abc&expires=123"
    assert detect_kind_from_name(url) == InputKind.PDF


def test_name_url_with_percent_encoded_path():
    url = "https://example.com/folder/My%20Slides.pptx"
    assert detect_kind_from_name(url) == InputKind.PPTX


def test_name_unknown_extension_returns_none():
    assert detect_kind_from_name("archive.zip") is None
    assert detect_kind_from_name("notes.txt") is None
    assert detect_kind_from_name("photo.gif") is None
    assert detect_kind_from_name("vector.svg") is None


def test_name_image_extensions():
    # Every v1 image extension routes to the image converter.
    for name in (
        "scan.png",
        "photo.jpg",
        "photo.jpeg",
        "photo.jpe",
        "shot.webp",
        "fax.tif",
        "fax.tiff",
        "camera.heic",
        "camera.heif",
        "legacy.bmp",
    ):
        assert detect_kind_from_name(name) == InputKind.IMAGE, name
    assert detect_kind_from_name("SCAN.PNG") == InputKind.IMAGE
    assert (
        detect_kind_from_name("https://example.com/scan.png?token=abc")
        == InputKind.IMAGE
    )


def test_name_empty_or_none():
    assert detect_kind_from_name(None) is None
    assert detect_kind_from_name("") is None


def test_name_no_extension():
    assert detect_kind_from_name("just-a-filename") is None


# ---------------------------------------------------------------------------
# detect_kind_from_magic
# ---------------------------------------------------------------------------


def test_magic_pdf():
    assert detect_kind_from_magic(PDF_MAGIC) == InputKind.PDF


def test_magic_pdf_requires_signature_prefix():
    # Leading whitespace/garbage defeats the sniff — PDFs must start with %PDF-.
    assert detect_kind_from_magic(b" %PDF-1.7") is None


def test_magic_docx():
    assert detect_kind_from_magic(DOCX_MAGIC) == InputKind.DOCX


def test_magic_pptx():
    assert detect_kind_from_magic(PPTX_MAGIC) == InputKind.PPTX


def test_magic_zip_without_office_markers_is_none():
    # A plain zip archive with neither `word/` nor `ppt/` in the first
    # 4 KiB is rejected — we let the filename fallback take over.
    assert detect_kind_from_magic(AMBIGUOUS_ZIP) is None


def test_magic_empty_bytes():
    assert detect_kind_from_magic(b"") is None


def test_magic_short_bytes():
    # Too short to contain any full signature.
    assert detect_kind_from_magic(b"%") is None
    assert detect_kind_from_magic(b"PK") is None


def test_magic_detects_ooxml_regardless_of_entry_position():
    # Central-directory based detection is position-independent — padding
    # the archive with large unrelated entries before the `word/` entry
    # must not fool the sniffer.
    padded = _make_zip([
        ("padding.bin", b"\x00" * 8192),
        ("also-padding.bin", b"\x00" * 8192),
        ("word/document.xml", b"<w:document/>"),
    ])
    assert detect_kind_from_magic(padded) == InputKind.DOCX


def test_magic_malformed_zip_returns_none():
    # A PK signature followed by garbage is not a parseable ZIP — the
    # sniffer should fall through cleanly rather than raise.
    truncated = b"PK\x03\x04" + b"\x00" * 200
    assert detect_kind_from_magic(truncated) is None


def test_magic_zip_with_both_markers_is_inconclusive():
    # Pathological archive containing both `word/` and `ppt/` entries —
    # refuse to guess and let the filename break the tie.
    mixed = _make_zip([
        ("word/document.xml", b"<w:document/>"),
        ("ppt/presentation.xml", b"<p:presentation/>"),
    ])
    assert detect_kind_from_magic(mixed) is None


def _real_image_bytes(fmt: str) -> bytes:
    """Encode a real 4x4 image with Pillow so the signature is authentic."""
    from PIL import Image

    if fmt == "HEIF":
        import pillow_heif

        pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format=fmt)
    return buf.getvalue()


def test_magic_png():
    assert detect_kind_from_magic(_real_image_bytes("PNG")) == InputKind.IMAGE


def test_magic_jpeg():
    assert detect_kind_from_magic(_real_image_bytes("JPEG")) == InputKind.IMAGE


def test_magic_webp():
    assert detect_kind_from_magic(_real_image_bytes("WEBP")) == InputKind.IMAGE


def test_magic_tiff_both_endians():
    assert detect_kind_from_magic(_real_image_bytes("TIFF")) == InputKind.IMAGE
    # Explicit signature checks for both byte orders.
    assert detect_kind_from_magic(b"II*\x00" + b"\x00" * 8) == InputKind.IMAGE
    assert detect_kind_from_magic(b"MM\x00*" + b"\x00" * 8) == InputKind.IMAGE


def test_magic_bmp():
    assert detect_kind_from_magic(_real_image_bytes("BMP")) == InputKind.IMAGE


def test_magic_bmp_prefix_alone_is_inconclusive():
    # "BM" is a weak 2-byte signature — a payload shorter than a full BMP
    # file header must not classify as an image.
    assert detect_kind_from_magic(b"BM") is None


def test_magic_heic_brands():
    assert detect_kind_from_magic(_real_image_bytes("HEIF")) == InputKind.IMAGE
    # Hand-built ftyp boxes covering major-brand and compatible-brand hits.
    major = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"
    assert detect_kind_from_magic(major) == InputKind.IMAGE
    compat_only = b"\x00\x00\x00\x18ftypXXXX\x00\x00\x00\x00XXXXmif1"
    assert detect_kind_from_magic(compat_only) == InputKind.IMAGE
    # An mp4 ftyp (no HEIF brand anywhere) stays inconclusive.
    mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
    assert detect_kind_from_magic(mp4) is None


def test_magic_gif_is_not_detected():
    # GIF is deliberately unsupported — its signature must stay inconclusive.
    assert detect_kind_from_magic(b"GIF89a" + b"\x00" * 16) is None


# ---------------------------------------------------------------------------
# detect_kind — composite
# ---------------------------------------------------------------------------


def test_composite_magic_wins_over_filename():
    # A `.docx` filename whose bytes are actually a PDF → PDF.
    assert (
        detect_kind(filename="report.docx", data=PDF_MAGIC) == InputKind.PDF
    )


def test_composite_pdf_bytes_named_png_stay_pdf():
    # Magic-first: `pretend.png` containing PDF bytes routes as PDF.
    assert detect_kind(filename="pretend.png", data=PDF_MAGIC) == InputKind.PDF


def test_composite_image_bytes_with_wrong_filename_route_as_image():
    png = _real_image_bytes("PNG")
    assert detect_kind(filename="scan.pdf", data=png) == InputKind.IMAGE
    assert detect_kind(filename=None, data=png) == InputKind.IMAGE


def test_composite_plain_zip_is_not_image():
    # A plain ZIP with an image-ish filename still resolves via the filename
    # fallback (IMAGE from extension), but the bytes alone are inconclusive.
    assert detect_kind_from_magic(AMBIGUOUS_ZIP) is None
    assert detect_kind(filename="archive.zip", data=AMBIGUOUS_ZIP) is None


def test_composite_filename_fills_in_when_magic_is_inconclusive():
    # No bytes supplied → falls through to filename.
    assert detect_kind(filename="report.pdf", data=None) == InputKind.PDF
    # Ambiguous zip bytes (no office marker) → fall through to filename.
    assert (
        detect_kind(filename="letter.docx", data=AMBIGUOUS_ZIP)
        == InputKind.DOCX
    )


def test_composite_returns_none_when_both_inconclusive():
    assert detect_kind(filename=None, data=None) is None
    assert detect_kind(filename="unknown.bin", data=b"???") is None


def test_composite_magic_only_without_filename():
    # No filename at all — browser uploaded raw bytes. Magic must carry the day.
    assert detect_kind(filename=None, data=PDF_MAGIC) == InputKind.PDF
    assert detect_kind(filename=None, data=DOCX_MAGIC) == InputKind.DOCX
