"""Order-free numeric agreement between two OCR readings of the SAME page.

Serving-side support for the dots.ocr self-consistency gate: a production silent-failure audit
showed ~4.5% of pages are
VOLATILE — identical submissions return different numbers, with catastrophic flips
(numeric f1 0.26↔0.98) that a single-run guard can never see. Two runs of the same
model on the same image expose it trivially: their numeric content should match.

This is deliberately a MINIMAL sibling of ``evals2/core/metrics/numeric.py`` (the
full eval metric with canonicalization for cross-provider comparison). Here both
readings come from the SAME model on the SAME image, so formatting conventions are
identical and no canonicalization is needed — only digit-bearing token extraction
and a multiset F1. Kept inside ``extract.core`` because the serving containers do
not ship the ``evals2`` package; the eval-side metric remains the source of truth
for cross-provider scoring.

No regex: a left-to-right character scan (every rule explicit). A token is a
maximal run starting at a digit (or sign immediately before a digit) at a
non-alphanumeric boundary, spanning digits and the in-number separators
``. , : / -`` and ``%``; alphanumeric words are skipped whole so the ``1`` in
``HbA1c`` is never a value.
"""

from __future__ import annotations

_DIGITS = frozenset("0123456789")
_SEP = frozenset(".,:/-%")
_SIGN = frozenset("+-")


def numeric_tokens(text: str) -> list[str]:
    """Digit-bearing value tokens in ``text``, in order. Word-embedded digits are
    excluded (the whole alphanumeric word is skipped)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        boundary = i == 0 or not text[i - 1].isalnum()
        if c.isalpha():
            while i < n and text[i].isalnum():
                i += 1
            continue
        starts = boundary and (
            c in _DIGITS or (c in _SIGN and i + 1 < n and text[i + 1] in _DIGITS)
        )
        if starts:
            start = i
            if c in _SIGN:
                i += 1
            while i < n and (text[i] in _DIGITS or text[i] in _SEP):
                i += 1
            while i > start and text[i - 1] in _SEP and text[i - 1] != "%":
                i -= 1  # a value ends on a digit or %
            tok = text[start:i]
            if any(ch in _DIGITS for ch in tok):
                out.append(tok)
            if i <= start:
                i = start + 1
            continue
        i += 1
    return out


def page_numeric_tokens(page_result) -> list[str]:
    """Numeric tokens across an ``OCRPageResult``'s blocks + table cells."""
    if page_result is None:
        return []
    toks: list[str] = []
    for b in page_result.blocks:
        if b.text:
            toks.extend(numeric_tokens(b.text))
    for t in page_result.tables:
        for cell in t.cells:
            if cell.text:
                toks.extend(numeric_tokens(cell.text))
    return toks


def numeric_agreement(tokens_a: list[str], tokens_b: list[str]) -> float | None:
    """Multiset F1 between two token lists. ``None`` when neither carries numbers
    (nothing to disagree about); 0.0 when exactly one side is empty."""
    if not tokens_a and not tokens_b:
        return None
    if not tokens_a or not tokens_b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(tokens_a), Counter(tokens_b)
    overlap = sum((ca & cb).values())
    prec, rec = overlap / len(tokens_b), overlap / len(tokens_a)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def choose_majority_read(token_lists: list[list[str]], min_agreement: float) -> int | None:
    """Index of the read to SERVE among 2-3 reads of the same page, by majority
    numeric agreement — or ``None`` when no two reads agree (the page is
    inconsistent and must not be served silently).

    Pure and order-stable: the best agreeing pair wins and its EARLIEST read is
    served (biases toward the first attempt, matching prod's existing
    first-read-wins behavior when reads agree).

    Empty-pair rule (adversarial-review F2): two number-free reads "agree" ONLY
    when every read is number-free. If ANY read extracted numbers, an empty-empty
    pair scores 0.0 — otherwise two degenerate reads that both dropped the numbers
    would outvote the one read that found them, and the gate built to stop
    vanishing numbers would vanish them itself.
    """
    any_tokens = any(token_lists)
    best: tuple[float, int] | None = None
    for i in range(len(token_lists)):
        for j in range(i + 1, len(token_lists)):
            a = numeric_agreement(token_lists[i], token_lists[j])
            score = (0.0 if any_tokens else 1.0) if a is None else a
            if score >= min_agreement and (best is None or score > best[0]):
                best = (score, i)
    return None if best is None else best[1]
