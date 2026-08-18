"""Entry-major regrouping of concordance tables (owner ruling 2026-08-05).

Input: a GFM concordance table whose cells are printed-line fragments in
column order (row-per-line or per-column-ordinal — both preserve each
column's top-to-bottom fragment sequence). Output: same columns, one cell
per ENTRY (term + [count] + all cites, fragments space-joined); row i =
i-th entry of each column; empty header. Letter/section bands are their own
single entries. Fail-closed: unclassifiable fragments -> None (caller keeps
the input form and flags).
Token conservation is exact by construction (pure regrouping); the caller
asserts it anyway.

W12 extension (2026-08-05) — three print styles, one classifier:

* **count-marker heads** — `term [9]` (bracket) OR `term (9)` (paren). Both
  reporter houses are in the corpus (Min-U-Script prints `(9)`, the
  `[9]` form is the other), and matching only `[9]` left every `(n)` sheet
  line-major: the head was unrecognisable, so every fragment fell through
  to the cite/continuation branch.
* **bare-term heads** — `term 7:9 8:25` with no count marker at all
  (First Coast / Wilson deposition indexes). Text alone cannot separate a
  numeric-section head (`1 7:8,12 9:5`) from a wrapped cite run
  (`79:23 102:10,11`) — both are all-digit token runs — so the caller may
  pass a per-fragment geometry `hint` (True = printed on the column's left
  margin ⇒ head; False = indented ⇒ continuation). The hint never overrides
  a band or an explicit count marker.
* **real cite tokens** — a cite is `page:line` with optional appended line
  lists and `;`-chained refs (`45:12,17,25;48:14;`), not just `\\d+:\\d+`.
  The old token pattern rejected every comma-list cite, so continuation
  lines were misread as new heads.
"""

import re
from collections import Counter

# `[9]` or `(9)` — both printed count-marker conventions in the corpus
CNT = re.compile(r"[\[(]\d+[\])]")
# a citation run: starts with a digit, then only digits and cite punctuation
# (`:` page/line, `/` alt page/line, `,` line list, `;` ref chain, `-` range)
CITE_TOK = re.compile(r"^(?:\d[\d:;,./–—-]*|-{1,2}|\.{2,}|\$?[\d.,]+)[,;.]?$")
# A letter/section band owns its own cell. Bands print centred (so their
# raster indent looks like a deep continuation) and some carry labeler
# heading/rule decoration (`### D`, `---`); a punctuation band (`'`) heads the
# apostrophe section of a speaker index. None of these is ever a wrapped term,
# so they must never be swallowed by the wrapped-term join below.
BAND = re.compile(
    r"^(?:###\s*|-{3,}\s*)?"
    r"(?:[A-Za-z]|\$|\d|#|['\"&]|[A-Z]\s*[-–]\s*[A-Z])$"
    r"|^-{3,}$|^###$"
)

# OWNER RULE 2026-08-05 (BAND RESTRICTION), final.
#
#   "A standalone dash is a section band ONLY when raster-confirmed centred in
#    its column on its own line (band-style separation, like letter bands); a
#    dash trailing text on a line is never a band."
#
# W13 had put a lone dash inside `BAND` outright, on the strength of one page
# (flmd.436144 p0126 prints a centred `-` section band above its hyphen-section
# entries, in the same centred style as the `0` band two lines below).  That
# over-reaches: a lone `-` is far more often the trailing dash of a printed
# `term (n) -` line whose cites wrapped, and classifying THAT as a band cuts an
# entry in half (tned.112290 p0023 col 4: `directly (2)` / `-`).
#
# So the dash is pulled back OUT of `BAND` and gated: it is a band only when
# the caller supplies `band_ok=True` for that fragment, which only the RASTER
# path can do — `align.hints_for` returns it from the printed line the fragment
# aligns to (centred in its column, on its own line).  With no raster
# confirmation available the dash falls through to the cite-continuation branch,
# which is what a trailing dash is.
DASH_BAND = re.compile(r"^[-–—]$")

SEP_ROW = re.compile(r":?-{3,}:?")


def _classify(f, hint=None, band_ok=None):
    t = f.strip()
    if not t:
        return "empty"
    # OWNER RULE (BAND RESTRICTION): a standalone dash needs raster confirmation
    if DASH_BAND.match(t):
        return "band" if band_ok is True else "cont"
    # a lone band glyph is a band whatever the geometry says (bands print
    # centred, so their indent looks like a deep continuation)
    if BAND.match(t):
        return "band"
    m = CNT.search(t)
    if m:
        return "head" if t[: m.start()].strip() else "count_attach"
    if hint is True:
        return "pending_head"
    if hint is False:
        return "cont"
    toks = t.split()
    if toks and all(CITE_TOK.match(w) for w in toks):
        return "cont"
    # alpha text with no count marker: either a bare-term head, or a term
    # line whose count wrapped to the next line
    return "pending_head"


def regroup_columns(col_frags, hints=None, column_bounded=True, band_ok=None):
    """col_frags: [[fragment, ...] per column]. Returns [[entry, ...]] or None.

    `band_ok` mirrors `hints`: [[True/False/None per fragment] per column].
    It carries the RASTER confirmation required by the 2026-08-05 BAND
    RESTRICTION rule before a standalone dash may be read as a section band.

    One reading stream across columns. `column_bounded` (owner ruling
    2026-08-05, w13): a cell NEVER spans a printed column boundary — an entry
    whose cites overflow into the next column is CLOSED at the boundary and the
    overflow opens a CONTINUATION CELL at the next column's top (the overflow
    fragments only, no term repeated), attributed by reading order to the
    column it printed in.  The leading cross-PAGE orphan is unchanged: a page
    may still open mid-entry, and those headless cites are still that page's
    first cell.  With `column_bounded=False` the pre-w13 behaviour returns (the
    entry belongs to the column where its HEAD printed, cites and all).
    """
    ncol = len(col_frags)
    stream = [(c, i, f) for c in range(ncol) for i, f in enumerate(col_frags[c]) if f.strip()]
    cols_entries = [[] for _ in range(ncol)]
    cur, cur_col = None, None
    opened_any = False
    for col, idx, f in stream:
        hint = None
        if hints is not None:
            hint = hints[col][idx] if idx < len(hints[col]) else None
        bok = None
        if band_ok is not None:
            bok = band_ok[col][idx] if idx < len(band_ok[col]) else None
        k = _classify(f, hint, bok)
        if k == "empty":
            continue
        # COLUMN BOUNDARY: close whatever is open — it printed in `cur_col`
        # and cannot reach into `col`.
        carried = False
        if column_bounded and cur is not None and col != cur_col:
            cols_entries[cur_col].append(cur)
            cur, cur_col = None, None
            carried = True
        # a page can OPEN mid-entry (head printed on the previous page): the
        # leading headless cites become an orphan entry — geometrically what
        # the page shows.  `carried` is the same situation one column over:
        # the headless cites at a column's top are that column's continuation
        # cell.  Mid-stream orphans still fail.
        if k in ("cont", "count_attach") and cur is None and (not opened_any or carried):
            cur, cur_col, opened_any = f, col, True
            continue
        if k in ("head", "band"):
            # a citeless pending_head is a WRAPPED TERM (an entry in this
            # print style always carries [n]/(n)); join it to the head that
            # completes it
            if (
                k == "head"
                and cur is not None
                and not CNT.search(cur)
                and not any(CITE_TOK.match(w) for w in cur.split())
            ):
                cur = cur + " " + f
                continue
            if cur is not None:
                cols_entries[cur_col].append(cur)
            cur, cur_col = f, col
            opened_any = True
            if k == "band":
                cols_entries[cur_col].append(cur)
                cur, cur_col = None, None
        elif k == "pending_head":
            if cur is not None:
                cols_entries[cur_col].append(cur)
            cur, cur_col = f, col
            opened_any = True
        elif k in ("count_attach", "cont"):
            if cur is None:
                return None
            cur = cur + " " + f
    if cur is not None:
        cols_entries[cur_col].append(cur)
    return cols_entries


def render(cols_entries):
    ncol = len(cols_entries)
    nrows = max((len(e) for e in cols_entries), default=0)
    if nrows == 0:
        return None
    out = ["|" + " |" * ncol, "|" + "---|" * ncol]
    for i in range(nrows):
        cells = [cols_entries[c][i] if i < len(cols_entries[c]) else "" for c in range(ncol)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def bag(s):
    """Token bag ignoring cell/row separators — `<br>` is a separator too
    (rule 11 bans it as a line-wrap join on these grids, so exploding it
    back to a space must read as token-neutral)."""
    s = s.replace("<br>", " ").replace("<hr>", " ")
    return Counter(w for w in re.sub(r"[|\n]", " ", s).split() if not SEP_ROW.fullmatch(w))


def table_columns(text, split_br=True):
    """Per-column printed-line fragment lists from a GFM table body."""
    lines = text.split("\n")
    if len(lines) < 2 or not lines[0].startswith("|"):
        return None
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in lines
        if ln.strip().startswith("|")
    ]
    rows = [r for r in rows if not (any(r) and all(SEP_ROW.fullmatch(c) for c in r if c))]
    if not rows:
        return None
    header, body = rows[0], rows[1:]
    ncol = len(header)
    if any(header):  # populated header = a printed first line; keep its
        body = [header] + body  # content as data (output header is always empty)
    if any(len(r) != ncol for r in body):
        return None
    cols = [[] for _ in range(ncol)]
    for r in body:
        for c in range(ncol):
            cell = r[c]
            if not cell.strip():
                continue
            parts = [p for p in re.split(r"<br>|<hr>", cell)] if split_br else [cell]
            cols[c].extend(p.strip() for p in parts if p.strip())
    return cols


def regroup_table(text, hints=None, split_br=True, column_bounded=True, band_ok=None):
    cols = table_columns(text, split_br=split_br)
    if cols is None:
        return None
    ce = regroup_columns(cols, hints, column_bounded=column_bounded, band_ok=band_ok)
    if ce is None:
        return None
    new = render(ce)
    if new is None:
        return None
    if bag(new) != bag(text):
        return None
    return new
