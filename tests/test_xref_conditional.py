"""Conditional cross-read: which pages earn a secondary read."""

import pathlib

from extract.config import settings
from extract.core.textract_page import PageRead
from extract.core.xref_context import page_needs_xref, xref_chunks


def _clean(words=200):
    return PageRead(text="clean print", words=[], total_words=words)


# --- page selection ---------------------------------------------------------


def test_clean_printed_page_is_skipped():
    assert page_needs_xref(_clean()) is False


def test_handwriting_selects_the_page():
    read = PageRead(text="x", total_words=142, handwriting_words=7)
    assert page_needs_xref(read) is True


def test_a_stray_handwritten_mark_does_not_select():
    """A signature squiggle scores 1-2 words; that is not written content."""
    read = PageRead(text="x", total_words=155, handwriting_words=1)
    assert page_needs_xref(read) is False


def test_low_confidence_alone_does_not_select():
    """A badly scanned page is not the same as a page holding a value at risk;
    selecting on it cost a 28-row table and bought no field. Diagnostic only."""
    read = PageRead(text="x", total_words=239, low_confidence_words=125)  # 52%
    assert page_needs_xref(read) is False


# --- fail-open --------------------------------------------------------------


def test_unreadable_page_is_selected():
    assert page_needs_xref(PageRead(error="ThrottlingException")) is True


def test_missing_read_is_selected():
    assert page_needs_xref(None) is True


def test_page_with_no_words_is_selected():
    assert page_needs_xref(PageRead(text="", total_words=0)) is True


# --- wiring -----------------------------------------------------------------


class _Cache:
    """Stands in for TextractPageCache: page 2 is handwritten, the rest are clean."""

    def __init__(self):
        self.reads = 0

    def read(self, image: bytes, size):
        self.reads += 1
        page = int(image.decode())
        if page == 2:
            return PageRead(text=f"secondary text p{page}", total_words=140, handwriting_words=9)
        return PageRead(text=f"secondary text p{page}", total_words=200)


def test_only_the_page_that_needs_it_is_cross_read(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "")  # isolate the Textract half
    cache = _Cache()
    imgs = {p: str(p).encode() for p in range(1, 5)}
    chunks = xref_chunks(imgs, cache=cache, page_sizes={p: (100, 100) for p in imgs})
    assert [c.page_no for c in chunks] == [2]
    # Every page is still READ — that is how we know which ones to keep, and the
    # box tightener reuses the same cache — only the context injection is scoped.
    assert cache.reads == 4


def test_explicit_reader_bypasses_selection():
    """No Textract response to select on, so an injected reader covers everything."""
    imgs = {p: str(p).encode() for p in range(1, 4)}
    chunks = xref_chunks(imgs, reader=lambda img: f"read {img.decode()}")
    assert [c.page_no for c in chunks] == [1, 2, 3]


# --- always-on cross-read (EXTRACT_XREF_ALWAYS_CUSTOMERS) -------------------



def test_always_keeps_every_page(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "")
    imgs = {p: str(p).encode() for p in range(1, 5)}
    chunks = xref_chunks(
        imgs, cache=_Cache(), page_sizes={p: (100, 100) for p in imgs}, always=True
    )
    assert [c.page_no for c in chunks] == [1, 2, 3, 4]
    assert all("SECONDARY OCR SOURCE" in c.page_content for c in chunks)


def test_always_widens_the_second_reader_too(monkeypatch):
    """EXP-A: on printed thermal receipts the handwriting gate never fires, and the
    recognizer read is the strongest single witness there (+0.035 dev micro, CI-strong).
    So the always-customers pay for it on every page, not only the handwriting ones."""
    from unittest.mock import patch

    monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
    imgs = {p: str(p).encode() for p in range(1, 5)}
    with patch("extract.core.xref_context._gemini_read", return_value="second read") as g:
        chunks = xref_chunks(
            imgs, cache=_Cache(), page_sizes={p: (100, 100) for p in imgs}, always=True
        )
    assert g.call_count == 4  # every page, not just the handwriting one
    assert sum("gemini" in c.page_content for c in chunks) == 4
    assert sum("textract" in c.page_content for c in chunks) == 4
    # Same label, same chunk shape, same whole-page bbox as the gated path.
    assert all(c.bbox == [0.0, 0.0, 1000.0, 1000.0] for c in chunks)
    assert all("SECONDARY OCR SOURCE" in c.page_content for c in chunks)


def test_non_always_customer_keeps_the_handwriting_gate_on_the_second_reader(monkeypatch):
    """The widening rides the always-customers list and nothing else: for everyone
    else the billed reader still sees only the pages that clear the gate."""
    from unittest.mock import patch

    monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
    imgs = {p: str(p).encode() for p in range(1, 5)}
    with patch("extract.core.xref_context._gemini_read", return_value="second read") as g:
        chunks = xref_chunks(imgs, cache=_Cache(), page_sizes={p: (100, 100) for p in imgs})
    assert g.call_count == 1  # only the handwriting page
    assert [c.page_no for c in chunks] == [2, 2]
    assert sum("gemini" in c.page_content for c in chunks) == 1
    assert sum("textract" in c.page_content for c in chunks) == 1


def test_always_second_reader_failure_is_fail_open(monkeypatch):
    """A widened read is still a secondary source: every page raising must cost the
    extraction nothing but those chunks."""
    from unittest.mock import patch

    monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
    imgs = {p: str(p).encode() for p in range(1, 5)}
    with patch(
        "extract.core.xref_context._gemini_read", side_effect=RuntimeError("vertex down")
    ) as g:
        chunks = xref_chunks(
            imgs, cache=_Cache(), page_sizes={p: (100, 100) for p in imgs}, always=True
        )
    assert g.call_count == 4
    assert [c.page_no for c in chunks] == [1, 2, 3, 4]  # the Textract half survives
    assert all("textract" in c.page_content for c in chunks)





def test_route_hands_its_warmed_cache_to_the_extraction_pass():
    """The prefetch warms one TextractPageCache; the pass must receive THAT cache.
    A stray reassignment used to null it, so xref_chunks built a fresh one and paid
    Textract a second time for every page."""
    src = pathlib.Path("src/extract/api/routes/v1.py").read_text()
    body = src[src.index("_xref_task = ("):src.index("xref_cache=_tx_cache")]
    assert "_tx_cache = None" not in body


def test_the_cross_read_stops_between_pages_when_the_request_is_over(monkeypatch):
    """Every page here is a paid call — Textract's $0.0015 and the recognizer's
    $0.00337 — and this runs in a worker thread for a request that may already
    have failed. Checking only at the top would let a 20-page document buy 19
    more reads for an answer nobody receives."""
    from extract.core import xref_context

    seen: list[int] = []
    stop_after = {"n": 1}

    def _reader(_img):
        seen.append(len(seen) + 1)
        return f"page {len(seen)}"

    def _should_stop():
        return len(seen) >= stop_after["n"]

    pages = {n: b"png" for n in range(1, 6)}
    out = xref_context.xref_chunks(pages, reader=_reader, should_stop=_should_stop)
    assert len(seen) == 1  # stopped before the second page
    assert len(out) == 1  # ...and kept what it had


def test_no_should_stop_reads_every_page(monkeypatch):
    """The normal path is untouched: no callback, no checks."""
    from extract.core import xref_context

    seen: list[int] = []

    def _reader(_img):
        seen.append(len(seen) + 1)
        return f"page {len(seen)}"

    out = xref_context.xref_chunks({n: b"png" for n in range(1, 6)}, reader=_reader)
    assert len(seen) == 5
    assert len(out) == 5
