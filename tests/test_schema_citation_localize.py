"""Unit tests for the image citation-localization core lib (plan 057).

No network: localizer results are constructed directly. Covers target building
from nested objects/arrays, candidate-page selection (from first-pass evidence and
from {value,pages} page hints), deterministic occurrence fallback, partitioning
(fields_per_call 1 / 4 / monolithic), bbox validation (range/order/degenerate/
too-broad/clip), source-text support, and evidence merge with parse fallback.
"""

from __future__ import annotations

from extract.core.models import Chunk, FieldEvidence
from extract.core.schema_citation_localize import (
    CitationLocalizationResult,
    CitationPageTarget,
    build_citation_targets,
    expand_page_targets,
    merge_localized_evidence,
    parse_occurrence_evidence,
    partition_citation_targets,
    validate_localized_evidence,
    value_supported_by_text,
)


def _chunk(text, page, bbox):
    return Chunk(page_content=text, page_no=page, bbox=bbox)


# --- target building --------------------------------------------------------


def test_build_targets_from_nested_objects_and_arrays():
    values = {
        "patient": {"name": "Jane Doe", "dob": None},
        "claims": [{"amount": "$124.00"}, {"amount": "$5.00"}],
    }
    evidence = {
        "patient.name": [FieldEvidence(page=1, bbox=[10, 10, 50, 20], text="Jane Doe")],
        "claims[0].amount": [FieldEvidence(page=3, bbox=[700, 600, 790, 640], text="$124.00")],
        "claims[1].amount": [FieldEvidence(page=4, bbox=[700, 600, 790, 640], text="$5.00")],
    }
    targets = build_citation_targets(values, evidence, [], [])
    paths = {t.path for t in targets}
    assert paths == {"patient.name", "claims[0].amount", "claims[1].amount"}  # dob is null → skipped
    by_path = {t.path: t for t in targets}
    assert by_path["patient.name"].candidate_pages == [1]
    assert by_path["claims[0].amount"].candidate_pages == [3]
    assert by_path["patient.name"].first_pass_quote == "Jane Doe"


def test_candidate_pages_from_page_hints_for_value_pages_mode():
    values = {"patient": {"name": "Jane Doe"}, "amount": "$5"}
    hints = {"patient.name": [1, 2], "amount": [7]}
    targets = build_citation_targets(values, {}, [], [], page_hints=hints)
    by_path = {t.path: t for t in targets}
    assert by_path["patient.name"].candidate_pages == [1, 2]
    assert by_path["amount"].candidate_pages == [7]


def test_occurrence_fallback_when_no_evidence_or_hints():
    values = {"patient": {"mrn": "ABC-77"}}
    chunks = [
        _chunk("header", 1, [0, 0, 10, 10]),
        _chunk("Patient MRN ABC-77 active", 4, [0, 0, 100, 20]),
    ]
    page_sizes = [(100, 100), (100, 100), (100, 100), (100, 100)]
    targets = build_citation_targets(values, {}, chunks, page_sizes)
    assert len(targets) == 1
    assert targets[0].candidate_pages == [4]  # found by deterministic occurrence search


def test_invalid_page_hint_falls_through_to_occurrence_search():
    values = {"x": "ABC-77"}
    chunks = [_chunk("value ABC-77 here", 2, [0, 0, 100, 20])]
    page_sizes = [(100, 100), (100, 100)]
    # page hint 99 is out of range → ignored → occurrence search finds page 2
    targets = build_citation_targets(values, {}, chunks, page_sizes, page_hints={"x": [99]})
    assert targets[0].candidate_pages == [2]


def test_expand_multi_page_value_into_separate_tasks():
    values = {"amount": "$124"}
    targets = build_citation_targets(values, {}, [], [], page_hints={"amount": [3, 4]})
    page_targets = expand_page_targets(targets)
    assert [(t.path, t.page) for t in page_targets] == [("amount", 3), ("amount", 4)]


# --- partitioning -----------------------------------------------------------


def _pt(path, page):
    return CitationPageTarget(path=path, value="v", page=page)


def test_partition_fields_per_call_1_4_and_monolithic():
    targets = [_pt(f"f{i}", page=1) for i in range(5)] + [_pt("g", page=2)]
    n1 = partition_citation_targets(targets, 1)
    assert len(n1) == 6 and all(len(c) == 1 for c in n1)
    n4 = partition_citation_targets(targets, 4)
    # page 1 (5 fields) → 4+1; page 2 (1 field) → 1  ==> 3 calls
    assert len(n4) == 3
    assert sorted(len(c) for c in n4) == [1, 1, 4]
    mono = partition_citation_targets(targets, 0)
    assert len(mono) == 2  # one call per page
    assert sorted(len(c) for c in mono) == [1, 5]
    # every call is single-page
    for call in n1 + n4 + mono:
        assert len({t.page for t in call}) == 1


# --- bbox validation --------------------------------------------------------


def test_validate_rejects_not_found_and_bad_box():
    t = _pt("x", 1)
    assert not validate_localized_evidence(t, CitationLocalizationResult("x", 1, found=False)).valid
    bad_shape = CitationLocalizationResult("x", 1, found=True, bbox=[1, 2, 3], source_text="x")
    assert validate_localized_evidence(t, bad_shape).reason == "bad_bbox_shape"


def test_validate_rejects_degenerate_but_accepts_broad():
    t = CitationPageTarget(path="x", value="abc", page=1)
    degen = CitationLocalizationResult("x", 1, found=True, bbox=[10, 10, 10, 50], source_text="abc")
    assert validate_localized_evidence(t, degen).reason == "degenerate"
    # MINIMAL policy (2026-07-01 audit): too_broad never fired in 1,240 real
    # localizations — a large box is accepted, not rejected.
    broad = CitationLocalizationResult("x", 1, found=True, bbox=[0, 0, 1000, 1000], source_text="abc")
    assert validate_localized_evidence(t, broad).valid


def test_validate_orders_and_clips_box():
    t = CitationPageTarget(path="x", value="abc", page=1)
    # reversed + slight overflow → ordered and clipped to grid
    r = CitationLocalizationResult("x", 1, found=True, bbox=[80, 40, 20, 10], source_text="abc here")
    out = validate_localized_evidence(t, r)
    assert out.valid and out.evidence.bbox == [20.0, 10.0, 80.0, 40.0]
    over = CitationLocalizationResult("x", 1, found=True, bbox=[10, 10, 1010, 60], source_text="abc")
    assert validate_localized_evidence(t, over).evidence.bbox[2] == 1000.0


def test_source_text_no_longer_gates_validation():
    # value_supported_by_text still exists as a helper...
    assert value_supported_by_text("Jane Doe", "Patient: Jane Doe (DOB...)")
    assert value_supported_by_text("$124.00", "Billed 124 00 USD")  # digit support
    assert not value_supported_by_text("Jane Doe", "totally unrelated text")
    # ...but no longer rejects a localization (2026-07-01 audit: 152/152 sampled
    # rejections were normalized-value false positives — '1957-06-03' vs boxed
    # '06-03-1957', 'False' vs boxed 'No' — stripping citations from the
    # date/boolean/enum fields that need them most).
    t = CitationPageTarget(path="x", value="1957-06-03", page=1)
    printed = CitationLocalizationResult("x", 1, found=True, bbox=[10, 10, 50, 20],
                                         source_text="06-03-1957")
    assert validate_localized_evidence(t, printed).valid


# --- merge ------------------------------------------------------------------


def test_merge_keeps_valid_localization_and_falls_back_to_parse():
    values = {"a": "Jane", "b": "Bob"}
    old = {
        "a": [FieldEvidence(page=1, bbox=[0, 0, 100, 100], text="Jane block")],
        "b": [FieldEvidence(page=2, bbox=[0, 0, 100, 100], text="Bob block")],
    }
    # 'a' localizes valid+tight; 'b' fails validation → fallback to parse box
    out_a = validate_localized_evidence(
        CitationPageTarget("a", "Jane", 1),
        CitationLocalizationResult("a", 1, found=True, bbox=[10, 10, 40, 20], source_text="Jane"))
    out_b = validate_localized_evidence(
        CitationPageTarget("b", "Bob", 2),
        CitationLocalizationResult("b", 2, found=False))
    merged = merge_localized_evidence(values, old, [out_a, out_b], fallback_policy="parse")
    assert merged["a"][0].bbox == [10.0, 10.0, 40.0, 20.0]  # tight localized box wins
    assert merged["b"] == old["b"]  # parse fallback


def test_merge_localized_only_drops_failed_localization():
    old = {"b": [FieldEvidence(page=2, bbox=[0, 0, 100, 100], text="Bob block")]}
    out_b = validate_localized_evidence(
        CitationPageTarget("b", "Bob", 2), CitationLocalizationResult("b", 2, found=False))
    merged = merge_localized_evidence({"b": "Bob"}, old, [out_b], fallback_policy="localized-only")
    assert "b" not in merged  # no valid box and no parse fallback


def test_parse_occurrence_evidence_builds_boxes_from_chunks():
    values = {"x": "ABC-77"}
    chunks = [_chunk("row with ABC-77 inside", 2, [0, 0, 100, 20])]
    ev = parse_occurrence_evidence(values, chunks, [(100, 100), (100, 100)])
    assert ev["x"][0].page == 2
    assert ev["x"][0].bbox == [0.0, 0.0, 1000.0, 200.0]  # normalized 0-1000


# --- generalized box tightener (2026-07-02) ----------------------------------


def test_tighten_locates_iso_date_by_printed_form():
    from extract.core.schema_citation_localize import tighten_value_box

    # "DOB: 06-03-1957" on one line; value is the ISO-normalized form.
    box = [100.0, 500.0, 260.0, 514.0]
    out = tighten_value_box("1957-06-03", "DOB: 06-03-1957", box,
                            med_line_h=14.0, med_char_w=10.0)
    assert out is not None
    assert out[1] == 500.0 and out[3] == 514.0            # y untouched
    assert out[0] > 130.0 and out[2] <= 260.0             # x narrowed past "DOB: "
    assert (out[2] - out[0]) < (box[2] - box[0]) * 0.8    # genuinely tighter


def test_tighten_bails_on_ambiguity_multiline_and_sparse_rows():
    from extract.core.schema_citation_localize import tighten_value_box

    # two occurrences on one line -> ambiguous, keep coarse
    assert tighten_value_box("93030", "93030 zip 93030", [0, 0, 200, 12],
                             med_line_h=12.0, med_char_w=10.0) is None
    # multi-line chunk (height >> doc median line height)
    assert tighten_value_box("93030", "zip 93030", [0, 0, 100, 60],
                             med_line_h=12.0, med_char_w=1.0) is None
    # sparse row: implied per-char width >> doc median -> hidden gaps, bail
    assert tighten_value_box("93030", "zip 93030", [0, 0, 900, 12],
                             med_line_h=12.0, med_char_w=1.0) is None
    # dense wide line (width fits its text) DOES tighten — the absolute
    # 500-width rule would have refused this
    long_text = "Patient Zip 93030 " + "x" * 80
    out = tighten_value_box("93030", long_text, [0, 0, 600, 12],
                            med_line_h=12.0, med_char_w=6.0)
    assert out is not None and (out[2] - out[0]) < 120


def test_tighten_boolean_targets_printed_word():
    from extract.core.schema_citation_localize import tighten_value_box

    out = tighten_value_box(False, "Auto Accident? No", [0.0, 0.0, 170.0, 12.0],
                            med_line_h=12.0, med_char_w=10.0)
    assert out is not None and out[0] > 120.0             # the "No" at line end
