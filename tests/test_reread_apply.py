"""apply_reread: the extraction-side pass that gates high-stakes fields on
parse confidence and re-reads their crops. Stubbed reader + generated PNG — no
network, no PHI.
"""
import io
from dataclasses import dataclass, field

from PIL import Image

from extract.core.ocr.reread import apply_reread, field_confidence, is_high_stakes


@dataclass
class Ev:
    page: int
    bbox: list | None
    confidence: float | None
    needs_review: bool = False
    suggested_value: str | None = None


@dataclass
class Fld:
    path: str
    value: object
    evidence: list = field(default_factory=list)


def _png() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (80, 24), "white").save(b, "PNG")
    return b.getvalue()


def _pages(_page):  # every page returns the same tiny blank png
    return _png()


def test_high_stakes_matcher_and_rollup():
    assert is_high_stakes("discoveredInsurances[0].member_id", (".member_id",))
    assert not is_high_stakes("patient_info.first_name", (".member_id",))
    assert field_confidence([Ev(1, [0, 0, 10, 10], 0.9), Ev(1, [0, 0, 10, 10], 0.3)]) == 0.3
    assert field_confidence([Ev(1, None, None)]) is None


def test_member_id_below_gate_disagrees_flags_with_suggestion():
    f = Fld("discoveredInsurances[0].member_id", "1EG4TE5MK72",
            [Ev(2, [580, 448, 895, 475], 0.285)])
    n = apply_reread([f], _pages, 
                     reader=lambda crop: "1EG4-TE5-MK73", gate=0.80)
    assert n == 1
    assert f.evidence[0].needs_review is True
    assert f.evidence[0].suggested_value == "1EG4-TE5-MK73"


def test_member_id_reader_agrees_confirms_no_flag():
    f = Fld("discoveredInsurances[0].member_id", "2P17XX8XK68",
            [Ev(1, [10, 10, 90, 30], 0.78)])
    n = apply_reread([f], _pages, 
                     reader=lambda crop: "2P17XX8XK68", gate=0.80)
    assert n == 1                       # gate tripped (re-read happened)
    assert f.evidence[0].needs_review is False   # but readers agreed -> no flag


def test_non_member_id_field_is_never_reread():
    f = Fld("patient_info.first_name", "Wright", [Ev(2, [10, 10, 90, 30], 0.2)])
    n = apply_reread([f], _pages, 
                     reader=lambda crop: "SHOULD_NOT_BE_CALLED", gate=0.80)
    assert n == 0
    assert f.evidence[0].needs_review is False


def test_member_id_above_gate_is_not_reread():
    f = Fld("discoveredInsurances[0].member_id", "90278471E", [Ev(4, [10, 10, 90, 30], 0.999)])
    n = apply_reread([f], _pages, 
                     reader=lambda crop: "SHOULD_NOT_BE_CALLED", gate=0.80)
    assert n == 0
    assert f.evidence[0].needs_review is False


def test_reader_exception_never_breaks_extraction():
    def boom(crop):
        raise RuntimeError("gemini down")
    f = Fld("discoveredInsurances[0].member_id", "1EG4TE5MK72", [Ev(2, [580, 448, 895, 475], 0.285)])
    n = apply_reread([f], _pages, reader=boom, gate=0.80)
    assert n == 1
    assert f.evidence[0].needs_review is False   # failure swallowed, field left as-is


def test_reread_evidence_end_to_end_flags():
    """Real tiny PDF through the pipeline seam: low-conf member_id -> flagged."""
    import pymupdf

    from extract.core.models import FieldEvidence
    from extract.core.ocr.reread import reread_evidence

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((100, 100), "Member ID: 1EG4TE5MK72")
    pdf_bytes = doc.tobytes()

    ev = FieldEvidence(page=1, bbox=[150, 110, 450, 140], text="1EG4TE5MK72", confidence=0.28)
    evidence = {"discoveredInsurances[0].member_id": [ev]}
    values = {"discoveredInsurances": [{"member_id": "1EG4TE5MK72"}]}

    n = reread_evidence(evidence, values, pdf_bytes, 
                        reader=lambda crop: "1EG4-TE5-MK73", gate=0.80)
    assert n == 1
    assert ev.needs_review is True
    assert ev.suggested_value == "1EG4-TE5-MK73"


def test_reread_evidence_skips_without_opening_pdf_when_above_gate():
    from extract.core.models import FieldEvidence
    from extract.core.ocr.reread import reread_evidence

    ev = FieldEvidence(page=1, bbox=[150, 110, 450, 140], text="90278471E", confidence=0.999)
    evidence = {"discoveredInsurances[0].member_id": [ev]}
    values = {"discoveredInsurances": [{"member_id": "90278471E"}]}
    # doc_bytes is garbage; if the gate is respected it's NEVER opened -> no error
    n = reread_evidence(evidence, values, b"not-a-real-pdf", 
                        reader=lambda crop: "X", gate=0.80)
    assert n == 0
    assert ev.needs_review is False
