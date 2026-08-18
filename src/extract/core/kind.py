"""Document kind detection — magic bytes first, filename extension second.

Centralizes the logic that used to live as ``_detect_kind`` inside
``extractor.py``. Accepting raw bytes via the multipart route means we can't
trust the filename alone: browsers commonly send ``application/octet-stream``
for unknown uploads, and users rename PDFs to ``.docx`` all the time. We
sniff the first few bytes and only fall back to the filename when the magic
is inconclusive.
"""

from __future__ import annotations

import io
import zipfile
from urllib.parse import unquote, urlparse

from extract.core.models import InputKind

_PDF_EXTS = (".pdf",)
_PPTX_EXTS = (".pptx", ".ppsx")
_DOCX_EXTS = (".docx", ".doc")
_IMAGE_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".bmp",
)

# ISO BMFF brands (the 4 bytes after ``ftyp``) that identify HEIC/HEIF stills.
# ``mif1``/``msf1`` are the generic HEIF image/sequence brands pillow-heif
# decodes; the rest are HEVC-coded still/sequence brands.
_HEIF_BRANDS = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"heif"}
)


def detect_kind_from_name(source: str | None) -> InputKind | None:
    """Detect kind from a URL, filesystem path, or bare filename.

    Query strings and fragments are stripped via ``urlparse`` so that
    ``https://example.com/report.pdf?token=abc`` still resolves to ``PDF``.
    """
    if not source:
        return None
    parsed = urlparse(source)
    path = unquote(parsed.path or source).lower()
    if path.endswith(_PDF_EXTS):
        return InputKind.PDF
    if path.endswith(_PPTX_EXTS):
        return InputKind.PPTX
    if path.endswith(_DOCX_EXTS):
        return InputKind.DOCX
    if path.endswith(_IMAGE_EXTS):
        return InputKind.IMAGE
    return None


def _is_heif_magic(data: bytes) -> bool:
    """ISO BMFF sniff: a ``ftyp`` box at offset 4 whose major or compatible
    brands include a HEIC/HEIF still-image brand. The box size (bytes 0-4)
    bounds the compatible-brand scan; we cap it defensively as well."""
    if len(data) < 12 or data[4:8] != b"ftyp":
        return False
    if data[8:12] in _HEIF_BRANDS:
        return True
    box_size = int.from_bytes(data[0:4], "big")
    end = min(len(data), box_size if 16 <= box_size <= 256 else 32)
    return any(data[off : off + 4] in _HEIF_BRANDS for off in range(16, end - 3, 4))


def detect_kind_from_magic(data: bytes) -> InputKind | None:
    """Detect kind from a document's signature bytes.

    - PDF: starts with ``%PDF-``.
    - PPTX / DOCX: both are ZIP archives (``PK\\x03\\x04``). We open the
      archive and inspect its central directory — an OOXML document is
      identified by having entries under ``word/`` (DOCX) or ``ppt/``
      (PPTX). The central directory lives at the end of a ZIP, so the
      caller must pass the full bytes; partial reads will be treated as
      inconclusive.
    - Raster images: PNG / JPEG / WebP / TIFF / HEIC-HEIF / BMP signatures
      route to the image→PDF converter. Only formats Pillow (+ pillow-heif)
      can decode are sniffed — no ``python-magic``/``filetype`` dependency.

    A ZIP we can't parse (truncated, corrupt, encrypted header) is
    treated as inconclusive so the filename fallback gets a chance.
    """
    if data.startswith(b"%PDF-"):
        return InputKind.PDF
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
        except (zipfile.BadZipFile, OSError, ValueError):
            return None
        has_word = any(n.startswith("word/") for n in names)
        has_ppt = any(n.startswith("ppt/") for n in names)
        if has_word and not has_ppt:
            return InputKind.DOCX
        if has_ppt and not has_word:
            return InputKind.PPTX
        # Both markers, or neither — let the filename break the tie.
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return InputKind.IMAGE
    if data.startswith(b"\xff\xd8\xff"):
        return InputKind.IMAGE
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return InputKind.IMAGE
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return InputKind.IMAGE
    if _is_heif_magic(data):
        return InputKind.IMAGE
    # ``BM`` alone is a weak 2-byte signature; require the full 14-byte BMP
    # file header so arbitrary text starting with "BM" stays inconclusive.
    if data.startswith(b"BM") and len(data) >= 14:
        return InputKind.IMAGE
    return None


def detect_kind(
    *,
    filename: str | None = None,
    data: bytes | None = None,
) -> InputKind | None:
    """Magic-byte first, filename second.

    Magic wins on disagreement — a file named ``report.docx`` whose bytes
    start with ``%PDF-`` is treated as a PDF. Falls back to the filename
    only when magic is inconclusive (e.g., legacy ``.doc`` where we don't
    attempt OLE sniffing and rely on the extension to route to LibreOffice).
    """
    if data is not None:
        kind = detect_kind_from_magic(data)
        if kind is not None:
            return kind
    return detect_kind_from_name(filename)
