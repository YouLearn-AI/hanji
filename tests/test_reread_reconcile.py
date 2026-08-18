"""Reconciliation logic for the tier-2 low-confidence re-read.

Policy (default): Gemini is the second-pass reader; agreement confirms, any
disagreement is FLAGGED for human review with the re-read surfaced as a
suggestion (never auto-overwrite). Cases mirror the real sub-80% member_ids on
the production demo docs, with r61615's human-confirmed truth pinning why
auto-recovery is unsafe.
"""
from extract.core.ocr.reread import reconcile


def test_above_gate_trusts_parse_without_reading():
    r = reconcile("90278471E", 1.0, [], gate=0.90)
    assert r.action == "trusted"
    assert r.value == "90278471E"
    assert r.needs_review is False


def test_reader_agrees_with_parse_confirms():
    # r54751: parse 78% but Gemini reproduces it -> false alarm, no review
    r = reconcile("2P17XX8XK68", 0.78, ["2P17XX8XK68"], gate=0.90)
    assert r.action == "confirmed"
    assert r.value == "2P17XX8XK68"
    assert r.needs_review is False


def test_reader_disagrees_flags_with_suggestion():
    # r29833: parse 28%, Gemini reads a different (better) value -> flag + suggest,
    # keep the parse value as emitted, human confirms. Label text is stripped.
    r = reconcile("1EG4TE5MK72", 0.285, ["Policy #: 1EG4-TE5-MK73"], gate=0.90)
    assert r.action == "flagged"
    assert r.needs_review is True
    assert r.value == "1EG4TE5MK72"            # never auto-overwritten
    assert r.suggested_value == "1EG4-TE5-MK73"  # cleaned Gemini read shown to reviewer


def test_illegible_flags_even_though_reader_is_confident():
    # r61615: Gemini stably reads XBS851038680, but the human truth is
    # XBS861036680 -> a stable-but-wrong re-read must still only FLAG, not recover.
    r = reconcile("XBS851038580", 0.508, ["XBS851038680"], gate=0.90)
    assert r.action == "flagged"
    assert r.needs_review is True
    assert r.value == "XBS851038580"           # parse value kept, human decides


def test_two_readers_split_offers_no_suggestion():
    # If a second opinion is added and the two disagree, there's nothing safe to suggest.
    r = reconcile("XBS851038580", 0.508, ["XBS851038680", "XBS861038680"], gate=0.90)
    assert r.action == "flagged"
    assert r.suggested_value is None


def test_auto_recover_opt_in_replaces_on_unanimous_disagreement():
    r = reconcile("1EG4TE5MK72", 0.285, ["1EG4-TE5-MK73"], gate=0.90, auto_recover=True)
    assert r.action == "recovered"
    assert r.value == "1EG4-TE5-MK73"


def test_gate_tripped_but_no_reader_flags():
    r = reconcile("H75668047", 0.5, [None, ""], gate=0.90)
    assert r.action == "flagged"
    assert r.needs_review is True
