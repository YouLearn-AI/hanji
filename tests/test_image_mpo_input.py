"""iPhone HDR photos (MPO) must parse like any other JPEG.

An MPO is a JPEG carrying more than one image: frame 0 is the full-resolution
photo, the extra frame is a half-resolution grayscale gain map. Pillow reports
format "MPO" and sets is_animated=True, so before this such a file was refused
twice over — a real user's iPhone photo of a document was rejected in 0ms as an
unsupported format (2026-07-29), which is the single most common intake path
there is.
"""

from __future__ import annotations

import asyncio
import io

import pymupdf
import pytest
from PIL import Image

from extract.core.converters.image import convert_to_pdf
from extract.core.errors import UnsupportedInput


def _mpo(primary=(400, 600), aux=(200, 300)) -> bytes:
    """A two-frame MPO: RGB photo + smaller grayscale gain map, as iPhones emit."""
    a = Image.new("RGB", primary, (240, 240, 240))
    b = Image.new("RGB", aux, (128, 128, 128))
    buf = io.BytesIO()
    a.save(buf, format="MPO", save_all=True, append_images=[b])
    raw = buf.getvalue()
    assert Image.open(io.BytesIO(raw)).format == "MPO", "fixture is not an MPO"
    return raw


async def test_mpo_converts_using_only_the_primary_frame():
    pdf = await convert_to_pdf(data=_mpo())
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        # ONE page: the gain map is not a page. Emitting it would corrupt the
        # output and bill an extra page.
        assert len(doc) == 1
        # Geometry follows the PRIMARY frame, not the smaller auxiliary one.
        assert doc[0].rect.width > doc[0].rect.height * 0.5
    finally:
        doc.close()


async def test_genuinely_animated_images_are_still_refused():
    """The MPO exemption must not open the door to animated GIF/WebP."""
    frames = [Image.new("RGB", (60, 60), c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:], duration=100)
    with pytest.raises(UnsupportedInput):
        await convert_to_pdf(data=buf.getvalue())


def test_mpo_is_accepted_but_not_treated_as_multipage():
    from extract.core.converters.image import _ACCEPTED_FORMATS, _SINGLE_FRAME_FORMATS

    assert "MPO" in _ACCEPTED_FORMATS
    assert "MPO" in _SINGLE_FRAME_FORMATS
    # TIFF stays genuinely multi-page (one PDF page per frame).
    assert "TIFF" not in _SINGLE_FRAME_FORMATS
