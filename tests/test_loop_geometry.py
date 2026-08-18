"""Geometry-aware degenerate-loop detection (experiment 130).

The class this replaces the old width thresholds for is the WIDE MOSTLY-EMPTY GRID: a
medication-administration row with 24 blank cells is 25 pipes on one markdown line and
is a *correct* extraction. The predecessor fired on it. Every test below is written
against a real measured page shape, not an invented one.
"""

from __future__ import annotations

import json

import pytest

from extract.core import loop_geometry as lg
from extract.core import pdf



#: a real markdown newline inside a chunk text
NL = chr(10)
#: the two-character JSON escape a raw completion carries instead
ESC_NL = chr(92) + "n"
PREFIX = '[{"bbox_2d": [0, 0, 9, 9], "text_content": "'

# --------------------------------------------------------------------------- #
# helpers — build raws in the exact shape production sees them: a JSON array of
# chunk records, so a markdown row break is the TWO-CHARACTER escape ``\n``.
# --------------------------------------------------------------------------- #
def _raw(*text_contents: str, terminated: bool = True) -> str:
    recs = [{"bbox_2d": [0, 0, 100, 100], "text_content": t} for t in text_contents]
    s = json.dumps(recs)
    return s if terminated else s[: s.rindex('"}]')] + "  |  |  |  |"


def _mar_row(n_blank: int, label: str = "Metoprolol 25mg") -> str:
    """One legitimate wide-grid row: a label then ``n_blank`` blank cells."""
    return f"| {label} |" + "  |" * n_blank


def _grid(n_cols: int, n_body: int, n_blank_rows: int = 0) -> str:
    """A well-formed GFM table: header, separator, body rows, then blank rows."""
    head = "|" + "|".join(f" c{i} " for i in range(n_cols)) + "|"
    sep = "|" + "|".join(" --- " for _ in range(n_cols)) + "|"
    body = ["|" + "|".join(f" v{r}{c} " for c in range(n_cols)) + "|" for r in range(n_body)]
    blank = ["|" + "|".join("  " for _ in range(n_cols)) + "|" for _ in range(n_blank_rows)]
    return "\n".join([head, sep, *body, *blank])


# --------------------------------------------------------------------------- #
# the false-positive class the redesign exists for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_blank", [12, 16, 24, 30, 36])
def test_wide_grid_row_is_not_a_loop(n_blank):
    """A single wide MAR row is one table row and must never be called a loop.

    These widths are the measured plan-114 range (gold same-line runs 18-33 pipes);
    the predecessor fires on every one of them from 16 pipes up.
    """
    raw = _raw(_mar_row(n_blank))
    assert lg.is_degenerate_loop(raw) is False
    assert lg.is_degenerate_loop_strict(raw) is False


def test_wide_grid_row_fires_the_frozen_v1_predicate():
    """Pin the defect itself, so a reviewer can see what changed and a regression
    back to width thresholds fails loudly."""
    raw = _raw(_mar_row(24))
    assert pdf._has_degenerate_empty_cell_loop(raw) is True   # v1: false positive
    assert lg.is_degenerate_loop(raw) is False                # v2: correct


def test_blank_rows_within_a_real_grid_are_not_a_loop():
    """A form with more blank rows than filled ones is ordinary (an unfilled MAR
    sheet). Measured max over the correct corpus: 17 consecutive blank rows."""
    raw = _raw(_grid(n_cols=8, n_body=3, n_blank_rows=17))
    assert lg.max_blank_row_run(raw) == 17
    assert lg.is_degenerate_loop(raw) is False


# --------------------------------------------------------------------------- #
# the loop class
# --------------------------------------------------------------------------- #
def test_unterminated_tail_of_blank_cells_is_a_loop():
    """The p0186 signature: the decode never closed its JSON and the last thing it
    emitted was blank cells. No width threshold is consulted."""
    raw = _raw("| FULL DESC: Y25-97 | 3,995.00 |", terminated=False)
    assert lg.is_unterminated(raw) is True
    assert lg.ends_in_blank_cells(raw) is True
    assert lg.is_degenerate_loop(raw) is True


def test_short_unterminated_tail_still_fires():
    """No minimum run length: an unterminated decode stuck on blank cells is a loop
    whether it managed 3 cells or 3,000. Length was never the evidence."""
    raw = '[{"bbox_2d": [0, 0, 9, 9], "text_content": "| a | b |  |  |'
    assert lg.is_degenerate_loop(raw) is True


def test_blank_row_run_beyond_the_measured_maximum_fires_when_the_decode_never_CLOSED():
    """The row-run trip needs BOTH limbs since v3: deep blank rows AND no closure.

    Before codex rule 4.14 the depth alone was the evidence, because no correct page
    had ever produced 18 consecutive blank rows. 4.14 makes that shape legal, so the
    surviving discriminator is closure — see the companion test below.
    """
    grid = _grid(n_cols=6, n_body=2, n_blank_rows=lg.BLANK_ROW_RUN_MAX)
    raw = _raw(grid, terminated=False)
    assert lg.max_blank_row_run(raw) >= lg.BLANK_ROW_RUN_MAX
    assert lg.is_unterminated(raw) is True
    assert lg.is_degenerate_loop(raw) is True


def test_blank_row_run_one_below_the_bound_does_not_fire():
    raw = _raw(_grid(n_cols=6, n_body=2, n_blank_rows=lg.BLANK_ROW_RUN_MAX - 1))
    assert lg.is_degenerate_loop(raw) is False


# --------------------------------------------------------------------------- #
# codex GT rule 4.14 — a blank grid transcribed WHOLE is a correct answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_blank_rows", [18, 24, 40])
def test_a_closed_printed_blank_grid_is_not_a_loop(n_blank_rows):
    """THE DEGENERATE-LOOP FIXTURE. Owner ruling 2026-08-06 (codex 4.14): a genuinely
    blank printed grid is emitted as the FULL table at its printed row count, cells
    empty, and then the record ENDS.

    plan-128's B1 probe is why: masking the image out of attention BROKE 3/3 banked loop pages
    loops and the model resumed normal records, so the model is faithfully transcribing
    real emptiness with no learned stopping rule. Condemning the correct transcription
    at serve would re-bill a right answer to the fallback AND keep pushing labelers
    back toward the truncated grid that caused the loop.
    """
    raw = _raw(_grid(n_cols=6, n_body=1, n_blank_rows=n_blank_rows))
    assert lg.max_blank_row_run(raw) == n_blank_rows >= lg.BLANK_ROW_RUN_MAX
    assert lg.is_unterminated(raw) is False
    assert lg.is_degenerate_loop(raw) is False


def test_the_same_grid_unclosed_is_still_a_loop():
    """The exemption is closure, not depth: the identical grid that never closed its
    JSON is exactly the runaway the detector exists for."""
    grid = _grid(n_cols=6, n_body=1, n_blank_rows=40)
    assert lg.is_degenerate_loop(_raw(grid)) is False
    assert lg.is_degenerate_loop(_raw(grid, terminated=False)) is True


# --------------------------------------------------------------------------- #
# the self-relative (constant-free) rules, used only on the restore path
# --------------------------------------------------------------------------- #
def test_supported_grid_width_needs_the_width_twice():
    """A GFM table emits its column count repeatedly; a runaway row is a singleton."""
    raw = _raw(_grid(n_cols=5, n_body=3))
    assert lg.supported_grid_width(raw) == 5


def test_run_wider_than_the_pages_own_grid_is_strict_only():
    """A 400-cell run on a page whose tables are 5 wide is off its own grid — but the
    serving path still does not fire, because a terminated decode is not a loop and
    the FP cost there dominates. The restore path does fire."""
    raw = _raw(_grid(n_cols=5, n_body=3) + "\\n| x |" + "  |" * 400)
    assert lg.supported_grid_width(raw) == 5
    assert lg.max_blank_cell_run(raw) == 400
    assert lg.is_degenerate_loop(raw) is False
    assert lg.is_degenerate_loop_strict(raw) is True


def test_wide_grid_survives_the_strict_predicate_too():
    """The strict arm must not resurrect the false positive: a MAR sheet whose rows
    repeat at the same width is supported by its own geometry."""
    rows = "\\n".join(_mar_row(24) for _ in range(6))
    raw = _raw(rows)
    assert lg.supported_grid_width(raw) >= 25
    assert lg.is_degenerate_loop_strict(raw) is False


def test_blank_rows_beyond_the_pages_own_body_is_strict_only():
    raw = _raw(_grid(n_cols=4, n_body=2, n_blank_rows=6))
    assert lg.content_row_count(raw) == 3      # header + 2 body (separator is not content)
    assert lg.max_blank_row_run(raw) == 6
    assert lg.is_degenerate_loop(raw) is False
    assert lg.is_degenerate_loop_strict(raw) is True


# --------------------------------------------------------------------------- #
# degenerate inputs / invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["", "   ", "plain prose with no table at all", "[]"])
def test_non_table_input_never_fires(raw):
    assert lg.is_degenerate_loop(raw) is False
    assert lg.is_degenerate_loop_strict(raw) is False


def test_empty_read_is_not_called_a_loop():
    """An empty read is ``qwen_empty``'s business; mislabelling it here would send it
    down the wrong recovery path."""
    assert lg.is_unterminated("") is False
    assert lg.is_degenerate_loop("") is False


def test_separator_row_is_not_a_blank_row():
    raw = _raw(_grid(n_cols=4, n_body=1))
    assert lg.max_blank_row_run(raw) == 0


def test_nbsp_between_pipes_is_still_a_blank_cell():
    """The character class stays ``[^\\S\\r\\n]`` (not ``[ \\t]``): NBSP / U+2000-200A /
    U+3000 between pipes were a measured blind spot in plan-114's r4 review."""
    assert lg.max_blank_cell_run("| a | |　| |") == 3


def test_strict_is_a_superset_of_serving():
    """Invariant: the restore arm may only ever fire MORE. Checked on shapes spanning
    both classes so a future edit cannot silently invert them."""
    cases = [
        _raw(_mar_row(30)),
        _raw(_grid(6, 3, 17)),
        _raw(_grid(6, 3, 30)),
        _raw("| a |", terminated=False),
        _raw(_grid(5, 3) + "\\n| x |" + "  |" * 99),
        "",
        "prose",
    ]
    for raw in cases:
        if lg.is_degenerate_loop(raw):
            assert lg.is_degenerate_loop_strict(raw), raw[:60]


def test_row_seam_handling_is_identical_raw_and_decoded():
    """The same function must be right on a raw completion (``\\n`` escape) and on a
    decoded ``text_content`` (real newline) — the two rulers diverging on that seam is
    exactly the blind spot loop_crosscheck.py documented."""
    decoded = _grid(n_cols=6, n_body=2, n_blank_rows=20)
    escaped = decoded.replace("\n", "\\n")
    assert lg.max_blank_row_run(decoded) == lg.max_blank_row_run(escaped) == 20
    assert lg.supported_grid_width(decoded) == lg.supported_grid_width(escaped) == 6


# --------------------------------------------------------------------------- #
# wiring: the two operating points reach the two callers
# --------------------------------------------------------------------------- #
def test_routing_is_v2():
    """The flip: routing reads the v2 pair, so the dot-leader page is served."""
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    leader = "Chapter 1 " + "." * 45 + " 12"
    pr = OCRPageResult(blocks=[OCRBlock(text=leader, bbox=[0, 0, 1, 1])])
    v1w, v1c, v2w, v2c = pdf._suspicious_limbs(pr)
    assert (v1w, v2w) == (True, False)          # the measured false positive
    assert pdf._ocr_page_result_is_suspicious(pr) is False   # v2 keeps it
    assert pdf._ocr_unusable_reason(pr) is None


# --------------------------------------------------------------------------- #
# the word limb, and the routing consequence of its old separator rule
# --------------------------------------------------------------------------- #
def test_a_twelve_month_constant_column_is_not_degeneration():
    """The production false positive that moved the token-run bound from 12 to 18.

    A ledger column carrying the same amount for twelve months, then its total, is
    twelve identical content tokens in a row — and it is correct. Measured on
    request e55d72e8.../page-0004 (65 distinct chunks, EOS-clean at 61% of cap).
    """
    col = "\n".join(["250"] * 12 + ["3,000"])
    assert lg.has_repeated_content(col) is False
    assert pdf._has_repeated_ngram(col) is True     # v1 discarded the whole page for it


@pytest.mark.parametrize("furniture", [
    "=" * 12,                                   # rule line
    "+-" * 6,                                   # ASCII table border
    "Chapter 1 " + "." * 40 + " 12",            # table-of-contents dot leader
    "-" * 40,                                   # dash rule
    "_" * 24,                                   # fill-in line
    "| Metoprolol |" + "  |" * 24,              # wide grid row
    "·" * 40,                              # middot leader (no whitelist covers it)
])
def test_typographic_furniture_is_not_degeneration(furniture):
    """Every one of these is document furniture, read correctly. The predecessor
    fires on all but the underscores — an inconsistency that is itself the argument:
    ``_`` was on a hard-coded whitelist and ``=`` was not."""
    assert lg.has_repeated_content(furniture) is False


@pytest.mark.parametrize("loop", [
    " ".join(["total"] * 18),                   # one word, repeated
    " ".join(["Rx 2"] * 30),                    # a phrase, repeated
    " ".join(["1234"] * 18),                    # a number, repeated
])
def test_repeated_content_is_degeneration(loop):
    assert lg.has_repeated_content(loop) is True


def test_dot_leader_page_is_the_measured_v1_only_class():
    """The production failure, end to end: a page of correctly-read leader lines.

    It is the ``v1_only`` column: the class the flip recovers, and the population the
    counters now confirm on the PHI lane, where no output is ever retained.
    """
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    blocks = [OCRBlock(text=f"Section {i} " + "." * 45 + f" {i * 3}", bbox=[0, 0, 1, 1])
              for i in range(1, 12)]
    pr = OCRPageResult(blocks=blocks)
    pr.raw = json.dumps([{"bbox_2d": [0, 0, 1, 1], "text_content": b.text} for b in blocks])
    v1w, v1c, v2w, v2c = pdf._suspicious_limbs(pr)
    assert (v1w, v1c) == (True, False)        # v1 discarded it, on the word limb
    assert (v2w, v2c) == (False, False)       # v2 keeps it
    assert pdf._ocr_unusable_reason(pr) is None


def test_the_75_blocks_exhibit_is_fixed_at_its_cause_not_by_restore():
    """The measured data-loss page: 75 correctly-read blocks destroyed because the
    separator rule flagged it and the flag is not restorable.

    Admitting ``qwen_repeated_ngram`` to ``_RESTORABLE_GUARD_REASONS`` was drafted and
    REJECTED: ``_restore_is_admissible`` re-runs the same classifier on the same bytes,
    so a read it just condemned can never pass — the permission would be unreachable
    and only a hand-constructed reason/result pair could ever "prove" it works. The
    honest fix is that the candidate never flags the page at all.
    """
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    blocks = [OCRBlock(text=f"{i}. Statement of Operations " + "." * 42 + f" {i}",
                       bbox=[0, 0, 1, 1]) for i in range(75)]
    pr = OCRPageResult(blocks=blocks)
    pr.raw = json.dumps([{"bbox_2d": [0, 0, 1, 1], "text_content": b.text} for b in blocks])
    assert any(pdf._has_repeated_ngram(b.text) for b in blocks) is True   # v1 destroys it
    assert lg.has_repeated_content(blocks[0].text) is False               # v2 never flags it
    assert "qwen_repeated_ngram" not in pdf._RESTORABLE_GUARD_REASONS


def test_a_genuine_loop_is_still_not_restored():
    """The invariant no restore change may ever break."""
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    looping = OCRPageResult(blocks=[OCRBlock(text="Total " * 200, bbox=[0, 0, 1, 1])])
    assert pdf._restore_is_admissible("qwen_repeated_ngram", looping) is False


def test_both_limb_halves_reproduce_their_predicates_exactly():
    """The counters must describe exactly what each predicate did — the v1 half the
    retired one (so the recovered class is countable) and the v2 half the live one."""
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    for text, raw in ((("Total " * 200), None),
                      ("Chapter 1 " + "." * 40 + " 12", None),
                      ("=" * 12, None),
                      ("ordinary clinic note text", None),
                      ("ok", _raw(_mar_row(30))),
                      ("ok", _raw("| a |", terminated=False))):
        pr = OCRPageResult(blocks=[OCRBlock(text=text, bbox=[0, 0, 1, 1])])
        pr.raw = raw
        v1w, v1c, v2w, v2c = pdf._suspicious_limbs(pr)
        texts = [b.text for b in pr.blocks if b.text]
        raw = pdf._extract_raw_text(pr.raw) if pr.raw is not None else ""
        # the retired predicate, reproduced bit-for-bit by the v1 half
        v1_expected = (any(pdf._has_repeated_ngram(t) for t in texts)
                       or pdf._has_degenerate_empty_cell_loop(raw))
        assert (v1w or v1c) is v1_expected
        # the LIVE predicate is the v2 half
        v2_expected = (any(lg.has_repeated_content(t) for t in texts)
                       or lg.is_degenerate_loop(raw))
        assert (v2w or v2c) is v2_expected
        assert pdf._ocr_page_result_is_suspicious(pr) is v2_expected


def _v2_is_degenerate_loop(raw: str) -> bool:
    """The SHIPPED v2 predicate, transcribed verbatim from the pre-4.14 source.

    Kept here (not imported) so the byte-identity test below compares against the old
    contract itself rather than against the new code's own opinion of it.
    """
    if not raw:
        return False
    if lg.is_unterminated(raw) and lg.ends_in_blank_cells(raw):
        return True
    return lg.max_blank_row_run(raw) >= lg.BLANK_ROW_RUN_MAX


def test_v3_is_byte_identical_to_v2_off_the_blank_grid_class():
    """THE EXEMPTION IS AN EXEMPTION, not a re-tune.

    v3 may differ from v2 on exactly one class: a CLOSED decode carrying >= 18
    consecutive blank rows — the shape codex 4.14 just made legal. On every other
    shape the two predicates must agree bit for bit, or this stopped being a scoped
    ruling and became an unmeasured change to a fail-closed serving gate.
    """
    shapes = [
        "", "   ", "prose with no table", "[]", "null",
        _raw(_mar_row(0)), _raw(_mar_row(9)), _raw(_mar_row(24)), _raw(_mar_row(400)),
        _raw(_grid(2, 1)), _raw(_grid(6, 3)), _raw(_grid(12, 40)),
        _raw(_grid(6, 3, 1)), _raw(_grid(6, 3, 5)), _raw(_grid(6, 3, 17)),
        _raw(_grid(5, 3) + "\\n| x |" + "  |" * 99),
        _raw("| a | b |", terminated=False),
        _raw(_grid(6, 3, 17), terminated=False),
        _raw(_grid(6, 3, 18), terminated=False),
        _raw(_grid(6, 3, 40), terminated=False),
        PREFIX + "| x |" + "  |" * 7448,
        PREFIX + "| a | b |" + (ESC_NL + "|  |  |") * 410,
        '[{"bbox_2d": [0, 0, 9, 9], "text_content": "| a | b |  |  |',
    ]
    exempt_class = []
    for raw in shapes:
        old, new = _v2_is_degenerate_loop(raw), lg.is_degenerate_loop(raw)
        if old == new:
            continue
        # The ONLY licensed disagreement.
        assert old is True and new is False, raw[:80]
        assert not lg.is_unterminated(raw), raw[:80]
        assert lg.max_blank_row_run(raw) >= lg.BLANK_ROW_RUN_MAX, raw[:80]
        exempt_class.append(raw)
    assert exempt_class == [], (
        "no shape in this list should be a CLOSED deep blank grid; the 4.14 class has "
        "its own tests above")


@pytest.mark.parametrize("n_blank_rows", [18, 24, 40])
def test_the_licensed_disagreement_is_exactly_the_414_class(n_blank_rows):
    """The other half of the byte-identity claim: on the 4.14 class the two predicates
    MUST differ, so the exemption is proven live and not silently a no-op."""
    raw = _raw(_grid(n_cols=6, n_body=1, n_blank_rows=n_blank_rows))
    assert _v2_is_degenerate_loop(raw) is True
    assert lg.is_degenerate_loop(raw) is False


def test_predicate_version_is_pinned():
    """Off-line contracts that pin serving parity key on this number. Bumping the
    predicate without bumping it would silently re-weight every pinned rollout."""
    assert lg.LOOP_PREDICATE_VERSION == 3
    assert lg.BLANK_ROW_RUN_MAX == 18
    assert lg.REPEATED_TOKEN_RUN_MAX == 18
    assert lg.REPEATED_WINDOW3_MAX == 8
    assert lg.REPEATED_WINDOW2_MAX == 30


def test_deleted_constants_are_gone_from_the_live_path():
    """The thin-air numbers the owner directive named. They survive only on the
    frozen v1 callable; nothing in ``loop_geometry`` may reintroduce them."""
    import ast

    tree = ast.parse(open(lg.__file__).read())        # code only — comments excluded
    code = ast.unparse(tree)
    assert "DOTS_" not in code, "the separator threshold family is back"
    assert '"-", "_", "."' not in code, "the separator whitelist is back"
    # No punctuation-specific repeat threshold may exist at all.
    assert "isalnum" in code, "the content test is the only classifier of a repeat unit"



def test_the_75_blocks_exhibit_is_served_end_to_end():
    """Coordinator bar 5, on the exact production shape.

    Measured on request e55d72e8.../page-0004: 75 chunks, 65 distinct, EOS-clean at
    4,986 of 8,192 tokens. Two chunks are a twelve-month constant-amount ledger column
    plus its total. v1 flagged the page on those two, Gemini returned zero chunks, and
    because ``qwen_repeated_ngram`` is not restorable the customer got an EMPTY page.
    Under v2 the page is never flagged, so it is simply served, all 75 blocks intact.
    """
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    blocks = [OCRBlock(text=f"Line {i} detail text", bbox=[0, 0, 1, 1]) for i in range(73)]
    blocks.append(OCRBlock(text=NL.join(["250"] * 12 + ["3,000"]), bbox=[0, 0, 1, 1]))
    blocks.append(OCRBlock(text=NL.join(["410"] * 12 + ["4,920"]), bbox=[0, 0, 1, 1]))
    pr = OCRPageResult(blocks=blocks)
    pr.raw = json.dumps([{"bbox_2d": [0, 0, 1, 1], "text_content": b.text} for b in blocks])
    assert any(pdf._has_repeated_ngram(b.text) for b in blocks) is True
    assert pdf._ocr_page_result_is_suspicious(pr) is False
    assert pdf._ocr_unusable_reason(pr) is None
    assert len(pr.blocks) == 75


def test_genuine_production_loops_still_route():
    """The 5 genuine degenerations in the retained production window: four single runs
    of 7,448-8,158 pipes (unterminated) and one page of 410 near-identical blank rows."""
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    mega = PREFIX + "| x |" + "  |" * 7448
    rows = PREFIX + "| a | b |" + (ESC_NL + "|  |  |") * 410
    for raw in (mega, rows):
        pr = OCRPageResult(blocks=[OCRBlock(text="partial content", bbox=[0, 0, 1, 1])])
        pr.raw = raw
        assert lg.is_degenerate_loop(raw) is True
        assert pdf._ocr_page_result_is_suspicious(pr) is True


def test_canary_risk_arm_is_untouched_by_the_flip():
    """The F1 consistency canary admits pages by RAW PIPE RUN
    (``EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN``), read off ``_max_empty_cell_run`` — a
    ruler this change does not touch. Eligibility must be bit-identical."""
    from extract.config import settings
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    raw = _raw(_mar_row(9))
    pr = OCRPageResult(blocks=[OCRBlock(text="x", bbox=[0, 0, 1, 1])])
    pr.raw = raw
    assert pdf._max_empty_cell_run(raw) >= settings.EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN
    assert pdf._consistency_risk_reason(pr, served_from_retry=False) == "empty_run"
    pr2 = OCRPageResult(blocks=[OCRBlock(text="prose", bbox=[0, 0, 1, 1])])
    pr2.raw = json.dumps([{"bbox_2d": [0, 0, 1, 1], "text_content": "prose"}])
    assert pdf._consistency_risk_reason(pr2, served_from_retry=False) is None


def test_retry_tier_still_triggers_on_the_reason():
    """The retry tier keys on the REASON, not the predicate, so a page v2 condemns
    retries exactly as before."""
    from extract.core.ocr.base import OCRBlock, OCRPageResult

    pr = OCRPageResult(blocks=[OCRBlock(text="real content", bbox=[0, 0, 1, 1])])
    pr.raw = PREFIX + "| x |" + "  |" * 200
    assert pdf._ocr_unusable_reason(pr) == "qwen_repeated_ngram"
    assert pdf._ocr_page_result_is_unusable(pr) is True
