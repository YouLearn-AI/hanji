"""Test fixtures.

Live tests hit real URLs from the reference repo. They're marked so they can
be skipped on CI or when offline. Run the full suite with::

    uv run pytest -m ""            # everything
    uv run pytest -m "not live"    # local/offline only
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test that hits a live URL (downloads PDFs, may call Textract)",
    )


def pytest_collection_modifyitems(config, items):
    # By default, skip live tests unless explicitly requested with `-m live`
    # or `-m ""`. The CI / Makefile uses `-m live` to run them intentionally.
    keyword = config.getoption("-m", default="")
    if keyword:
        return  # user asked for a specific marker; don't override
    skip_live = pytest.mark.skip(reason="live test; run with `-m live`.")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _stub_ocr_provider(monkeypatch, request):
    """OCR now runs on every page, so every extraction invokes the OCR provider.

    Stub it to an empty result by default so offline unit tests don't hit the
    network. Tests that exercise OCR behavior (e.g. the fallback tests) override
    ``extract.core.pdf.get_ocr_provider`` in their own body, which runs after this
    fixture and therefore wins. Live tests opt out.
    """
    if "live" in request.keywords:
        return
    from extract.core.ocr.base import OCRPageResult

    class _EmptyOCR:
        name = "stub"

        async def ocr_page(self, image_bytes, *, page_width, page_height):
            return OCRPageResult()

    monkeypatch.setattr(
        "extract.core.pdf.get_ocr_provider", lambda _name: _EmptyOCR(), raising=False
    )
