"""Unit tests for the serving-side numeric self-consistency helpers
(extract.core.numeric_agreement) — the dots.ocr consistency gate's pure logic.
Token extraction must exclude word-embedded digits, agreement must be order-free,
and the majority vote must be order-stable and refuse when no two reads agree."""

from __future__ import annotations

from extract.core.numeric_agreement import (
    choose_majority_read,
    numeric_agreement,
    numeric_tokens,
)

A = ["120/80", "98.6", "5.7", "217", "1,234.56"]
B_REORDERED = ["217", "1,234.56", "120/80", "5.7", "98.6"]
C_DIFFERENT = ["999", "111.1", "42"]


def test_numeric_tokens_extraction():
    toks = numeric_tokens("BP 120/80 temp 98.6 HbA1c 5.7 chol 217 paid $1,234.56 at 9:08")
    assert "120/80" in toks and "98.6" in toks and "1,234.56" in toks and "9:08" in toks
    # the 1 in HbA1c is a label, not a value
    assert "1" not in toks
    assert numeric_tokens("no numbers here") == []
    assert numeric_tokens("ends with 42") == ["42"]
    assert numeric_tokens("-3.2 delta") == ["-3.2"]


def test_agreement_order_free_and_bounds():
    assert numeric_agreement(A, B_REORDERED) == 1.0  # same multiset, any order
    assert numeric_agreement(A, C_DIFFERENT) == 0.0  # disjoint
    assert numeric_agreement([], []) is None  # nothing to disagree about
    assert numeric_agreement(A, []) == 0.0  # one side dropped everything
    partial = numeric_agreement(A, A[:3])
    assert partial is not None and 0.0 < partial < 1.0


def test_majority_vote_serves_earliest_agreeing_read():
    # volatile signature from a production silent-failure audit: good / bad / good
    assert choose_majority_read([A, C_DIFFERENT, B_REORDERED], 0.9) == 0
    # bad first: the agreeing pair is (1, 2) -> serve read 1
    assert choose_majority_read([C_DIFFERENT, A, B_REORDERED], 0.9) == 1
    # two reads that agree -> first
    assert choose_majority_read([A, B_REORDERED], 0.9) == 0


def test_majority_vote_refuses_when_no_two_reads_agree():
    assert choose_majority_read([A, C_DIFFERENT, ["7", "8"]], 0.9) is None
    assert choose_majority_read([A, C_DIFFERENT], 0.9) is None


def test_majority_vote_number_free_reads_count_as_agreeing():
    # two reads with no numbers at all agree that there is nothing numeric
    assert choose_majority_read([[], []], 0.9) == 0


def test_majority_vote_empty_pair_cannot_outvote_numeric_read():
    # F2: if ANY read found numbers, an empty-empty pair must NOT count as the
    # majority — otherwise two degenerate reads outvote the one that worked and
    # the gate vanishes the numbers it exists to protect.
    assert choose_majority_read([A, [], []], 0.9) is None
    assert choose_majority_read([[], A, []], 0.9) is None
