"""A bad PDF is the customer's problem to fix, so it must not read as ours.

PyMuPDF signals an unusable stream two ways — it raises (``EmptyFileError`` on
zero bytes, ``FileDataError`` on corruption), or it returns a zero-page
document. ``_open_or_repair`` only handled the second, so an empty or corrupt
upload escaped as an unhandled exception and became a bare HTTP 500.

Verified against production on 2026-07-31: a 0-byte file, a header-only PDF and
a non-PDF all returned ``500 Internal Server Error`` with
``error_customer_actionable=0``. Six of them came from one customer that day.
Two costs: the customer is told to contact support about a file only they can
fix, and every one of these pollutes the 5xx signal used to spot real outages.

Which typed error is the contract, per ``mintlify/guides/batch.mdx``: an empty
file is ``unsupported_input``; one we cannot read is ``extraction_failed``
("corrupted PDF, missing fonts"). Both already render in the dashboard, so this
makes behaviour match the documented contract rather than changing it.
"""

from __future__ import annotations

import pymupdf
import pytest

from extract.core.errors import ExtractionFailed, UnsupportedInput
from extract.core.pdf import _open_or_repair


def _valid_pdf(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


# --- the three inputs that returned 500 in production -----------------------


def test_empty_file_is_unsupported_input_not_a_server_error():
    with pytest.raises(UnsupportedInput):
        _open_or_repair(b"")


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("header only", b"%PDF-1.4\n"),
        ("not a pdf at all", b"not a pdf at all"),
        ("truncated mid-object", b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog"),
        ("empty-ish whitespace", b"   \n\n  "),
    ],
)
def test_malformed_bytes_raise_a_client_error(label: str, data: bytes):
    """Whatever the flavour of broken, it must be an ExtractError — never an
    unhandled PyMuPDF exception, which is what FastAPI turns into a 500."""
    # Non-empty but broken is ExtractionFailed exactly — `unsupported_input` is
    # reserved for an empty file, which batch.mdx now documents that way.
    with pytest.raises(ExtractionFailed) as exc:
        _open_or_repair(data)
    assert not isinstance(exc.value, pymupdf.FileDataError), label


def test_no_pymupdf_exception_escapes():
    """The regression guard: the bug was an exception type crossing the core
    boundary. Nothing PyMuPDF raises may reach the API layer uncaught."""
    for data in (b"", b"%PDF-1.4\n", b"garbage", b"\x00\x01\x02"):
        try:
            _open_or_repair(data)
        except (UnsupportedInput, ExtractionFailed):
            pass
        except pymupdf.FileDataError as e:  # pragma: no cover - the bug itself
            pytest.fail(f"PyMuPDF exception escaped for {data!r}: {type(e).__name__}")


# --- and the working path is untouched --------------------------------------


def test_a_valid_pdf_still_opens():
    doc = _open_or_repair(_valid_pdf(3))
    try:
        assert len(doc) == 3
    finally:
        doc.close()


def test_valid_pdf_never_touches_the_repair_path(monkeypatch):
    """A healthy document must not be relinearized on the way in. Asserting the
    page count alone would pass either way, so the repair hook is replaced with
    one that fails the test if it is ever reached."""
    import extract.core.pdf as pdf_mod

    def must_not_run(_: bytes):  # pragma: no cover - fires only on regression
        pytest.fail("repair_pdf_bytes was called for a healthy PDF")

    monkeypatch.setattr(pdf_mod, "repair_pdf_bytes", must_not_run)
    doc = _open_or_repair(_valid_pdf(2))
    try:
        assert len(doc) == 2
    finally:
        doc.close()


def _bad_page_count() -> bytes:
    """A PDF that OPENS fine but whose page tree lies about /Count.

    Found by adversarial review, then reproduced: `pymupdf.open()` succeeds and
    `len(doc)` raises RuntimeError("code=7: Invalid number of pages"). The first
    cut of this fix only guarded `open()`, so this input still escaped untyped
    (another 500) AND leaked the Document, since close() sat after the len().
    """
    doc = pymupdf.open()
    doc.new_page()
    raw = doc.tobytes()
    doc.close()
    return raw.replace(b"/Count 1", b"/Count 9")


def test_lying_page_count_is_a_client_error_not_a_500():
    data = _bad_page_count()
    # Precondition: this really is the open-ok / len-raises shape.
    probe = pymupdf.open(stream=data, filetype="pdf")
    try:
        with pytest.raises(RuntimeError):
            len(probe)
    finally:
        probe.close()
    with pytest.raises(ExtractionFailed):
        _open_or_repair(data)


def test_unusable_documents_are_always_closed():
    """Every rejection path must close what it opened — a leaked Document holds
    MuPDF memory for the life of the process."""
    opened: list = []
    real_open = pymupdf.open

    def tracking_open(*a, **kw):
        doc = real_open(*a, **kw)
        opened.append(doc)
        return doc

    pymupdf.open = tracking_open
    try:
        for data in (b"%PDF-1.4\n", b"garbage", _bad_page_count()):
            with pytest.raises((UnsupportedInput, ExtractionFailed)):
                _open_or_repair(data)
    finally:
        pymupdf.open = real_open
    assert opened, "expected at least one Document to have been opened"
    assert all(d.is_closed for d in opened), "a rejected Document was left open"



