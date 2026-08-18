"""Tests for the URL-extension fallback in ``Extractor.aextract``.

When a URL lacks a recognized extension (e.g. ``arxiv.org/pdf/<id>``), the
extractor should download once, magic-sniff the bytes, and still route to
the right pipeline instead of raising ``UnsupportedInput``. These tests use
an ``httpx.MockTransport`` so they don't hit the network.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from extract.core import ExtractRequest, Extractor, UnsupportedInput

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_FIXTURE = REPO_ROOT / "web" / "public" / "demo" / "attention.pdf"


def _request() -> ExtractRequest:
    # The `url` on the request is what kind detection looks at.
    # Use arxiv's real no-extension pattern.
    return ExtractRequest(
        url="https://arxiv.org/pdf/1706.03762",
        extract_images=False,
        ocr="never",
    )


def _mock_client(body: bytes) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason=f"fixture missing: {PDF_FIXTURE}")
async def test_no_extension_url_falls_back_to_magic_sniff():
    pdf_bytes = PDF_FIXTURE.read_bytes()
    async with _mock_client(pdf_bytes) as client:
        res = await Extractor(download_client=client).aextract(_request())
    assert res.page_count > 0


async def test_no_extension_url_with_garbage_raises():
    async with _mock_client(b"not a real document") as client:
        with pytest.raises(UnsupportedInput):
            await Extractor(download_client=client).aextract(_request())
