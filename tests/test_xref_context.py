"""Cross-engine secondary context. Must never break an extraction."""
from unittest.mock import patch

from extract.core.xref_context import xref_chunks


class _FakeCache:
    """Stands in for TextractPageCache so no AWS call is made."""

    def __init__(self, text):
        self._text = text

    def read(self, image, size):
        from extract.core.textract_page import PageRead

        return PageRead(text=self._text)


class TestXrefChunks:
    def test_returns_nothing_without_images(self):
        assert xref_chunks(None) == []
        assert xref_chunks({}) == []

    def test_one_labelled_chunk_per_page(self):
        out = xref_chunks({1: b"a", 2: b"b"}, reader=lambda img: "TEXT")
        assert len(out) == 2
        assert [c.page_no for c in out] == [1, 2]
        assert all("SECONDARY OCR SOURCE" in c.page_content for c in out)
        assert all(c.page_content.rstrip().endswith("TEXT") for c in out)

    def test_reader_failure_is_swallowed_per_page(self):
        def boom(img):
            if img == b"bad":
                raise RuntimeError("provider down")
            return "OK"
        out = xref_chunks({1: b"bad", 2: b"good"}, reader=boom)
        assert len(out) == 1 and out[0].page_no == 2

    def test_empty_read_produces_no_chunk(self):
        assert xref_chunks({1: b"x"}, reader=lambda i: "   ") == []

    def test_budget_truncates(self):
        # count only the READ, not the label (which itself contains lowercase 'y')
        out = xref_chunks({1: b"x"}, reader=lambda i: "ç" * 10_000, budget=100)
        assert out[0].page_content.count("ç") == 100

    def test_bbox_is_the_whole_page(self):
        out = xref_chunks({1: b"x"}, reader=lambda i: "T")
        assert out[0].bbox == [0.0, 0.0, 1000.0, 1000.0]


class TestSecondReader:
    """A second recognizer alongside Textract. Measured: Member ID 88% -> 98%, NPI 86% -> 92%."""

    def test_disabled_when_no_model_configured(self, monkeypatch):
        from extract.config import settings

        monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "")
        with patch("extract.core.xref_context._gemini_read") as g:
            out = xref_chunks({1: b"x"}, cache=_FakeCache("textract text"))
        g.assert_not_called()
        assert len(out) == 1

    def test_adds_one_chunk_per_page_when_configured(self, monkeypatch):
        from extract.config import settings

        monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
        with patch("extract.core.xref_context._gemini_read", return_value="Member ID: UL0024967"):
            out = xref_chunks({1: b"a", 2: b"b"}, cache=_FakeCache("textract text"))
        assert len(out) == 4  # 2 pages x 2 readers
        assert sum("gemini-3.5-flash" in c.page_content for c in out) == 2
        assert sum("textract" in c.page_content for c in out) == 2

    def test_second_reader_failure_does_not_lose_the_first(self, monkeypatch):
        from extract.config import settings

        monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
        with patch("extract.core.xref_context._gemini_read", side_effect=RuntimeError("vertex down")):
            out = xref_chunks({1: b"x"}, cache=_FakeCache("textract text"))
        assert len(out) == 1 and "textract" in out[0].page_content

    def test_empty_second_read_adds_nothing(self, monkeypatch):
        from extract.config import settings

        monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
        with patch("extract.core.xref_context._gemini_read", return_value="   "):
            out = xref_chunks({1: b"x"}, cache=_FakeCache("textract text"))
        assert len(out) == 1

    def test_second_read_is_budget_truncated(self, monkeypatch):
        from extract.config import settings

        monkeypatch.setattr(settings, "EXTRACT_XREF_GEMINI_MODEL", "gemini-3.5-flash")
        with patch("extract.core.xref_context._gemini_read", return_value="ç" * 9000):
            out = xref_chunks({1: b"x"}, cache=_FakeCache("t"), budget=50)
        assert [c for c in out if "gemini" in c.page_content][0].page_content.count("ç") == 50
