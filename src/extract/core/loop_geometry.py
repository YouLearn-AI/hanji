"""Degenerate empty-cell loop detection, decided on table GEOMETRY.

WHY THIS REPLACED THE WIDTH THRESHOLDS
--------------------------------------
The predecessor (``pdf._has_degenerate_empty_cell_loop``, 2026-06-16) fired on any
single-line run of >= 16 pipes, or >= 12 pipes inside the final ~1% of the output. It
was calibrated on 96 banked production scans holding exactly ONE degenerate exemplar
(p0186, a 22-pipe terminal run) against an 11-pipe legitimate maximum, and the "~1%"
terminal window was never fitted at all.

That calibration set contained no wide-mostly-empty grids, and a run bounded by row
seams is ONE TABLE ROW: a 24-empty-cell medication-administration row is 25 pipes on one
line and is *correct*. Measured over 97,297 correct pages (experiment 130), the old
predicate discards 235 of them -- including 80 of the 83 plan-114 pages whose own gold
trips it, which that campaign's exclusion receipt already names "STANDING PRODUCTION
FALSE POSITIVE"s -- while catching only 60.16% of genuine loops.

THE SIGNALS, AND WHERE EACH NUMBER COMES FROM
---------------------------------------------
No width threshold survives here, because no width threshold can: a run's width is a
property of the page's layout, not of the decode's health. What separates the classes is
STRUCTURE.

1. ``ends_unterminated_in_blank_cells`` -- the output is not parseable JSON and the last
   thing the decode was doing was emitting blank cells. NO CONSTANT. Measured: 0 of
   97,297 correct pages are structurally unterminated; 1,250 of 1,250 genuine loops are.

2. ``BLANK_ROW_RUN_MAX`` -- consecutive fully-blank table rows, **and only on a decode
   that never closed**. Derived, not chosen: the maximum over the 97,297-page correct
   corpus is 17, so the trip is 18, one above the observed maximum. Rule-of-three 95%
   upper bound on P(fire | correct) is 3.1e-5. Receipt:
   the internal measurement records.

   THE CLOSURE CONJUNCT (added 2026-08-06, owner GT ruling 4.14). ``BLANK_ROW_RUN_MAX``
   was derived over a corpus that TRUNCATES blank grids, and codex rule 4.14 has just
   reversed that convention: a genuinely blank printed grid is now emitted as the FULL
   table at its printed row count, cells empty, and then the record ends. Under that
   contract "more blank rows than any correct page has produced" stops being evidence
   of anything -- correct pages are about to produce them ON PURPOSE, which is the
   whole point of the demonstration lane (plan-128 B1: masking the image out of
   attention BROKE 3/3 banked loop pages, so the model is faithfully transcribing real
   emptiness with no learned stopping rule).

   What still separates the classes is CLOSURE, and it separates them perfectly on the
   same corpus the bound was derived from. RE-MEASURED 2026-08-06 by rebuilding
   experiment 130's corpus (``build_corpus.py`` over the 23-surface evals2 gold cache
   and 325 decode banks; 71,079 rows -- 9,861 gold + 53,937 clean decodes + 924 loops):

     * LOOP rows that are TERMINATED: **0 of 924**.
     * CORRECT rows with a blank-row run >= 18: **0 of 63,798** (max observed 13).
     * old predicate vs new, disagreeing rows: **0 of 71,079**.
     * TPR reproduces the committed receipt: 781/924 = 84.52% (receipt: 84.48%), and
       the blank-row limb is load-bearing -- 97 loops are caught by it ALONE, and all
       97 are unterminated, so all 97 survive the conjunct.

   So the conjunct gives up nothing measurable and buys the 4.14 class. What it does
   NOT cover: a terminated runaway, a shape no page in this corpus exhibits. If one
   ever appears it is caught on the restore path, which is truncated by construction.

   PARTIAL RE-DERIVATION, DISCLOSED. The rebuild recovers 63,798 of the receipt's
   97,297 correct pages; the plan-114 substrate and the 83-row standing-FP class live
   in a scratch dir that no longer exists. The receipt's ``correct_max_blank_row_run
   = 17`` -- the closest approach to the bound, and what makes 18 derived rather than
   chosen -- is in that unrecoverable remainder. The conjunct can only REDUCE firing,
   so no false positive can be introduced there; the unverified half is TPR, and the
   0-of-924 termination result makes a terminated loop there very unlikely but not
   measured.

   Census of the shape this protects, taken before the change over the registered
   eval-gold surfaces (9,861 pages, from the 21 of 23 that carry a page
   serialization): worst blank-ROW run **5**, worst blank-CELL run **25**
   (``max_blank_cell_run``; the synth guard's cross-seam ruler reads 27 on the same
   corpus -- different rulers, both non-binding), pages at or over the bound **0**.
   Today's gold never approaches either gate -- it truncates blank grids instead,
   which is exactly the shape 4.14 forbids and B1 tied to the loops.

3. ``exceeds_supported_grid`` / ``exceeds_own_blank_budget`` -- SELF-RELATIVE, no
   constant. A same-row blank run wider than any row width the page demonstrates TWICE
   is off the page's own grid (a GFM table of N columns necessarily emits N-wide rows
   more than once; a runaway is a singleton). A blank-row run longer than the page's own
   content-row count outruns its own body.

TWO OPERATING POINTS, EACH WITH ITS COST STATED
-----------------------------------------------
``is_degenerate_loop`` (SERVING path, ``_ocr_unusable_reason``): signals 1+2 only.
    A false positive here throws away a correct first-tier answer and re-bills the page
    to Gemini -- real money, every time. A false negative costs almost nothing: every
    genuine loop in the corpus also ran to the token cap, and ``_ocr_unusable_reason``
    checks ``page_result.truncated`` BEFORE suspicion, so the page routes to the
    fallback either way and the miss only changes the attributed reason code.
    Measured: FP 0/97,297 (was 235), TPR 84.48% (was 60.16%). All 194 misses are cap-hit.

``is_degenerate_loop_strict`` (RESTORE path, ``_restore_is_admissible``): adds signal 3.
    There the page is truncated by construction, so signal 1 carries no information, and
    the cost asymmetry INVERTS: a miss ships a looping page to the customer, while a
    false positive only ships the empty fallback page instead of a partial one. So the
    strict form buys recall with the self-relative width rules.
    Measured on 1,125 grid-bearing truncated pages: catch 353/418 (was 272), deny 33/154
    (was 29).

UNITS. Everything here counts CELLS. The predecessor counted PIPES while naming the
variable ``cells`` (N blank cells = N+1 pipes); that off-by-one leaked into
``config.EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN``. Nothing in this module inherits it.
"""

from __future__ import annotations

import json
import re

#: Bump when any predicate here changes semantics. Consumers that pin serving parity
#: (plan-114's reward floor imports the production callable on purpose) must re-pin
#: their contract hash against this number, never against a line range.
#:
#: v3 (2026-08-06): the blank-row-run trip is conjoined with ``is_unterminated`` so a
#: correct blank grid transcribed WHOLE (codex GT rule 4.14) is not a loop. Measured
#: cost on experiment 130's corpus: zero -- the arms disagree on no row.
LOOP_PREDICATE_VERSION = 3

#: Consecutive fully-blank table rows that no correct page has ever produced.
#: DERIVED: max = 17 over 97,297 correct pages (experiment 130). Do not tune by hand --
#: re-measure with the internal measurement records.
#:
#: NOT A STANDALONE TRIP since v3. Codex rule 4.14 makes a deep blank grid a LEGAL GT
#: shape, so this count is only read on a decode that never closed -- see the module
#: docstring, "THE CLOSURE CONJUNCT". Raising this number is NOT the way to admit a
#: deeper printed grid; the closure conjunct already admits it at any depth.
BLANK_ROW_RUN_MAX = 18

#: A blank markdown cell is a ``|`` followed only by horizontal whitespace; a chain of
#: them is a blank-cell run. Character class matches the predecessor's deliberately
#: (``[^\S\r\n]``, not ``[ \t]``) so NBSP / U+2000-200A / U+3000 between pipes are still
#: seen -- narrowing it was rejected as a live blind spot in plan-114's r4 review.
_BLANK_CELL_RUN_RE = re.compile(r"\|(?:[^\S\r\n]*\|)+")

#: A row seam in the RAW serving string is the two-character JSON escape ``\n`` (the
#: model's markdown newline, escaped because it sits inside a JSON string), or a real
#: newline once the string has been decoded. Splitting on both makes every function here
#: correct on a raw completion AND on a decoded ``text_content``.
_ROW_SEAM_RE = re.compile(r"\\r\\n|\r\n|\\n|\n")

#: A GFM separator row (``|---|---|``) is structure, not a blank row.
_SEP_ROW_RE = re.compile(r"^\s*\|(?:\s*:?-{2,}:?\s*\|)+\s*$")

#: Trailing punctuation a well-formed decode may close its JSON string/array with.
_JSON_TAIL_RE = re.compile(r"[\"'\]\}\s,]+$")

_CODE_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z]*\s*")
_CODE_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def _rows(raw: str) -> list[str]:
    """``raw`` split at row seams. Never parses JSON, so it behaves identically on a
    complete decode and on one the model never finished."""
    return _ROW_SEAM_RE.split(raw or "")


def _cell_count(row: str) -> int:
    """Inner ``|``-delimited cells in a markdown row (k pipes bound k-1 cells); 0 when
    the line is prose rather than grid."""
    p = row.count("|")
    return p - 1 if p >= 2 else 0


def _row_has_content(row: str) -> bool:
    """Does this row carry at least one non-blank cell?

    A GFM separator row (``|---|---|``) is table scaffolding, not content — counting
    it would inflate every page's own blank-row budget by one.
    """
    if row.count("|") < 2 or _SEP_ROW_RE.match(row.strip()):
        return False
    inner = row[row.index("|") + 1: row.rindex("|")]
    return any(c.strip() for c in inner.split("|"))


def _is_blank_row(row: str) -> bool:
    """A fully-blank grid row: >= 2 pipes, not a separator, no cell carrying anything."""
    if row.count("|") < 2 or _SEP_ROW_RE.match(row.strip()):
        return False
    inner = row[row.index("|") + 1: row.rindex("|")]
    return not any(c.strip() for c in inner.split("|"))


def max_blank_cell_run(raw: str) -> int:
    """Longest run of consecutive blank cells WITHIN ONE ROW, in cells.

    Bounded by row seams on purpose: a run that stops at a seam is one table row, which
    is legitimate at any width. This is a diagnostic ruler, not a trip -- no predicate
    here compares it to a constant.
    """
    best = 0
    for row in _rows(raw or ""):
        for m in _BLANK_CELL_RUN_RE.finditer(row):
            best = max(best, m.group(0).count("|") - 1)
    return best


def max_blank_row_run(raw: str) -> int:
    """Longest run of consecutive fully-blank grid rows."""
    best = cur = 0
    for row in _rows(raw or ""):
        if _is_blank_row(row):
            cur += 1
            best = max(best, cur)
        elif row.strip():
            cur = 0
    return best


def content_row_count(raw: str) -> int:
    """How many grid rows on this page carry content -- the page's own body size."""
    return sum(1 for row in _rows(raw or "") if _row_has_content(row))


def supported_grid_width(raw: str) -> int:
    """Widest row width this page demonstrates MORE THAN ONCE, in cells.

    A GFM table of N columns emits N-wide rows repeatedly (header, separator, body), so
    a width seen twice is a real grid. A width seen exactly once is a singleton -- and a
    runaway is always a singleton, because the decode never got to emit a second row of
    that width. Asking the page how wide its own tables are is what makes the width
    rules here constant-free.
    """
    counts: dict[int, int] = {}
    for row in _rows(raw or ""):
        w = _cell_count(row)
        if w > 0:
            counts[w] = counts.get(w, 0) + 1
    supported = [w for w, c in counts.items() if c >= 2]
    return max(supported) if supported else 0


def _strip_fence(s: str) -> str:
    if s.startswith("```"):
        s = _CODE_FENCE_CLOSE_RE.sub("", _CODE_FENCE_OPEN_RE.sub("", s)).strip()
    return s


def is_unterminated(raw: str) -> bool:
    """True when ``raw`` is not a complete JSON document.

    Fail-SAFE: an empty/whitespace raw is not called unterminated (an empty read is
    ``qwen_empty``'s business, and claiming a loop there would mis-attribute it).
    """
    s = _strip_fence((raw or "").strip())
    if not s:
        return False
    try:
        json.loads(s)
    except Exception:  # noqa: BLE001 — any parse failure means the decode did not close
        return True
    return False


def ends_in_blank_cells(raw: str) -> bool:
    """Does the output END inside a blank-cell run?

    Exact: only whitespace and the JSON punctuation a decode would close with may follow.
    There is no percentage window here -- the predecessor's ``max(3, n // 100)`` was
    asserted, never fitted, and a window is not needed once "ends in" is tested directly.
    """
    s = (raw or "").rstrip()
    if not s:
        return False
    s = _JSON_TAIL_RE.sub("", s)
    if not s.endswith("|"):
        return False
    last = _rows(s)[-1]
    runs = list(_BLANK_CELL_RUN_RE.finditer(last))
    return bool(runs) and runs[-1].end() == len(last)


def is_degenerate_loop(raw: str) -> bool:
    """SERVING-path predicate: the decode is stuck emitting blank table cells.

    Two structural signals, no width rule. Both require the decode to have NEVER
    CLOSED, which is the only property that separates the classes once codex rule
    4.14 makes a deep blank grid legal output:

      * the decode never closed its JSON and its tail is blank cells;
      * the decode never closed its JSON and it stacked more consecutive fully-blank
        rows than any correct page has produced.

    A COMPLETE transcription is never a loop here, however deep its blank grid runs.
    That is rule 4.14's whole content -- "the FULL table at its PRINTED row count,
    cells empty, and then the record ENDS" -- and it is why a page that stops because
    it finished the grid must not be re-billed to the fallback.

    Measured over experiment 130's corpus: 0 false positives on 97,297 correct pages
    (the predecessor: 235), 84.48% of 1,250 genuine loops (the predecessor: 60.16%).
    The v3 closure conjunct changes neither number -- every one of those 1,250 loops
    is unterminated, so nothing it removes was ever caught.
    """
    if not raw:
        return False
    if not is_unterminated(raw):
        # Closed JSON. The decode chose to stop, so whatever it emitted -- including a
        # 40-row printed blank grid -- is a finished answer, not a runaway.
        return False
    return ends_in_blank_cells(raw) or max_blank_row_run(raw) >= BLANK_ROW_RUN_MAX


def is_degenerate_loop_strict(raw: str) -> bool:
    """RESTORE-path predicate: :func:`is_degenerate_loop` plus the self-relative width
    rules.

    Used only where truncation is forgiven (``_restore_is_admissible``), because there
    the page is truncated by construction — so the serving predicate's structural signal
    carries no information — and a miss ships a loop to the customer while a false
    positive only ships the empty fallback page instead of a partial one.

    RULE 4.14, DELIBERATELY NOT APPLIED HERE. A terminated 4.14 blank grid still trips
    this predicate's self-relative limb (``max_blank_row_run > content_row_count``: a
    blank grid's only content row is its header). That is left alone because this
    callable is unreachable for a terminated page — ``pdf._RESTORABLE_GUARD_REASONS``
    is ``{"qwen_truncated"}``, so the page is truncated, hence unterminated, by
    construction. Narrowing an unreachable branch would be an unmeasured change to the
    one predicate whose cost asymmetry is inverted. If that reason set ever widens,
    re-derive this limb against 4.14 FIRST.
    """
    if not raw:
        return False
    if is_degenerate_loop(raw):
        return True
    # FAIL-OPEN when the page offers no evidence. A self-relative rule needs the page
    # to have demonstrated something; a page holding a single wide row has no repeated
    # width and no content-row count to judge against, and condemning it on absent
    # evidence is exactly the false positive this redesign exists to remove. (Caught
    # by test_wide_grid_row_is_not_a_loop: one legitimate 24-blank-cell MAR row.)
    support = supported_grid_width(raw)
    if support and max_blank_cell_run(raw) > support:
        return True
    content = content_row_count(raw)
    return bool(content) and max_blank_row_run(raw) > content


# ===========================================================================
# Repeated-unit degeneration (the WORD limb).
#
# THE MEASURED PROBLEM. The predecessor (``pdf._has_repeated_ngram``) gave
# punctuation the LOOSEST thresholds and words the tightest, on the theory that
# separators repeat legitimately and words do not. It then let a 2-gram window rule
# fire at 6 repeats with a whitelist that only exempts windows made ENTIRELY of
# ``{-, _, .}``. The result is that a 12-character ``============`` rule line, an
# ASCII table border ``+-+-+-+-+-+-``, a table-of-contents dot leader and a wide grid
# row all trip it, while a 24-underscore fill-in line does not — arbitrary in both
# directions. Production's own 14-day read
# (the internal measurement records
# §6) puts 243 of 255 token-run fires on runs of ``.`` at the 40 threshold, on
# correctly-read leader lines, and attributes 40.7–45.5% of the entire Gemini
# fallback bill to this rule. Over experiment 130's corpus it fires on 3.77% of
# CORRECT pages and 2.93% of looping ones — worse than chance.
#
# THE FIX. The discriminator is WHAT repeats, not how many times. A repeated
# punctuation glyph is typographic furniture — a leader, a rule, a border, an empty
# cell — and every document is full of them, so a count threshold on punctuation is
# measuring layout, not health. Content-bearing repetition is different in kind: it
# still happens legitimately (a twelve-month ledger column; a Yes/No option grid) but
# it is BOUNDED, and the bound is measurable. So the rule is "content-bearing units
# only", and the counts that survive are set from that measured bound.
#
# Restricting all three shapes to content-bearing units takes the false-positive rate
# over 43,618 correct pages from 866 to 0, while RAISING the catch on 2,861 cap-hit
# pages from 624 to 792. Every deleted constant governed punctuation.
#
# DELETED: DOTS_SEPARATOR_REPEAT_THRESHOLD (40), the single-punctuation threshold
# (24), the ``{-, _, .}`` whitelist, and the separator-window override. Every one
# governed punctuation only, and punctuation is no longer evidence.
# ===========================================================================

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

# THE HEADROOM RULE. Each bound below is **>= 1.5x the measured maximum over the
# correct corpus, rounded up** — one stated rule applied uniformly to all three axes,
# not a number picked per axis. It follows the repo's existing precedent for exactly
# this class of bound: ``synth.empty_run_guard.CELL_RUN_CAP`` is "1.28x the measured
# real-gold maximum", owner-accepted. An earlier draft used "observed max + 1", which
# review correctly called a fit with zero margin rather than a derivation — and the
# eval-surface golds then proved the point by moving which axis was tight.
#
# Measured maxima over 1,139,602 correct chunk texts / 45,815 correct pages, spanning
# eval-surface gold, plan-114 wide-grid pool gold, and EOS-clean decodes at
# token_f1 >= 0.90. Receipt:
# the internal measurement records
#
#: One content-bearing token repeated. Correct max 12 — a REAL PRODUCTION PAGE, and
#: the direct counter-example to "no document repeats a token twelve times"
#: (request e55d72e8.../page-0004, the "75 blocks" exhibit): two of its 75 chunks are
#: a column of the same 3-digit number 12 times followed by a 4-digit comma total,
#: i.e. a twelve-month constant-amount column and its sum. The page is 65 distinct
#: chunks, terminated at 4,986 of 8,192 tokens, with zero blank-cell/row runs — it is
#: correct, and 12 was firing on it. 12 * 1.5 = 18.
#: (The offline corpus max was 8; production carried the real tail. This is why the
#: bound is 1.5x a measured maximum and not the maximum + 1.)
REPEATED_TOKEN_RUN_MAX = 18
#: A content-bearing 3-token window repeated. Correct max 5
#: (a gold-labeled production page). 5 * 1.5 = 7.5 -> 8.
REPEATED_WINDOW3_MAX = 8
#: A content-bearing 2-token window repeated. Correct max 20
#: (table_stress v054_000311_wide_sparse_borderless, token_f1 0.913 — a wide sparse
#: borderless grid, i.e. exactly the answer-grid/Yes-No-column class that makes a
#: repeated content 2-gram legitimate). 20 * 1.5 = 30.
#: Production used 6 here, a level ordinary documents reach.
REPEATED_WINDOW2_MAX = 30


def _carries_content(unit: tuple[str, ...]) -> bool:
    """Does this repeated unit carry alphanumeric content?

    The whole discriminator, in one line. Deliberately not a character whitelist: a
    middot leader, a box-drawing rule and a bullet border are the same object as a dot
    leader, and a whitelist would have to grow forever to cover them.
    """
    return any(any(c.isalnum() for c in t) for t in unit)


def has_repeated_content(text: str) -> bool:
    """True iff ``text`` repeats a CONTENT-BEARING unit beyond the measured envelope.

    Three shapes, matching the predecessor's so the change is a restriction and not a
    new trigger: a single token repeated, a 2-token window repeated, a 3-token window
    repeated. Punctuation-only units are furniture and are never degeneration here —
    a runaway made of pipes is the blank-cell geometry's business
    (:func:`is_degenerate_loop`), which asks the right question about it.
    """
    if not text:
        return False
    tokens = _TOKEN_RE.findall(text.casefold())
    if not tokens:
        return False

    run_token, run_len = tokens[0], 1
    for token in tokens[1:]:
        if token == run_token:
            run_len += 1
            if run_len >= REPEATED_TOKEN_RUN_MAX and _carries_content((token,)):
                return True
        else:
            run_token, run_len = token, 1

    for n, bound in ((3, REPEATED_WINDOW3_MAX), (2, REPEATED_WINDOW2_MAX)):
        if len(tokens) < n * bound:
            continue
        previous = tuple(tokens[0:n])
        repeats = 1
        for start in range(n, len(tokens) - n + 1, n):
            current = tuple(tokens[start: start + n])
            if current == previous:
                repeats += 1
                if repeats >= bound and _carries_content(current):
                    return True
            else:
                previous, repeats = current, 1
    return False
