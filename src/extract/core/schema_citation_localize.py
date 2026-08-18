"""Image citation localization for schema extraction (plan 057).

The second pass of the two-pass citation experiment. Given values that another
pass already extracted, this module turns each ``(field path, value, page)`` into
an independent localization task, validates the model's returned box against the
value, and merges the localized boxes back into the ``evidence`` dict the
a production customer scorer reads — never touching the values themselves.

The module is deliberately pure: it knows about :class:`Chunk` / :class:`FieldEvidence`
(both ``extract.core.models``) and the box-normalization helper from
``schema_extract``, but nothing about Vertex, FastAPI, storage, or evals2. The
actual Gemini image call is injected as a :class:`SchemaCitationLocalizer`
(implemented in ``extract.clients.gemini_citations``); tests pass a stub.

Coordinate contract: a localized ``bbox`` is ``[x0, y0, x1, y1]`` on a 0–1000
page-relative grid with a top-left origin — identical to the normalized
``FieldEvidence.bbox`` the parse-citation path emits, so a localized box and a
parse-chunk box score through the same scorer code with no special-casing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from extract.core.models import Chunk, FieldEvidence
from extract.core.schema_extract import normalize_bbox

#: Coordinate grid the localizer works on (matches schema_extract.COORD_SCALE).
COORD_GRID = 1000.0
#: Reject a "localized" box that covers more than this fraction of the page — a
#: page-sized rectangle is not a tight citation, it is the model giving up.
MAX_BOX_AREA_FRACTION = 0.9
#: Cap candidate pages for one value so an ambiguous value (e.g. a state code that
#: recurs on every page) cannot explode into hundreds of localization calls. When
#: hit, the target is flagged ``candidate_pages_truncated`` for telemetry.
MAX_CANDIDATE_PAGES = 8
#: A value at least this many digits long may be grounded by digit-subsequence
#: match (ids/dates/amounts whose punctuation differs from the visible text).
_MIN_DIGIT_SUPPORT = 3


# --- Data carriers ----------------------------------------------------------


@dataclass
class CitationTarget:
    """One value to cite, with every page it might live on (routing hints)."""

    path: str
    value: Any
    description: str = ""
    candidate_pages: list[int] = field(default_factory=list)
    first_pass_quote: str | None = None
    first_pass_bbox: list[float] | None = None
    candidate_pages_truncated: bool = False


@dataclass
class CitationPageTarget:
    """One independent localization task: this value, on exactly this page."""

    path: str
    value: Any
    page: int
    description: str = ""
    first_pass_quote: str | None = None
    first_pass_bbox: list[float] | None = None


@dataclass
class CitationLocalizationResult:
    """Raw localizer output for one ``(path, page)`` task (model coordinates)."""

    path: str
    page: int
    found: bool
    bbox: list[float] | None = None
    source_text: str = ""
    raw_confidence: float | None = None


@dataclass
class LocalizationOutcome:
    """A validated localization: the page-relative evidence to keep, or why not."""

    path: str
    page: int
    valid: bool
    reason: str
    evidence: FieldEvidence | None = None


class SchemaCitationLocalizer(Protocol):
    """The injected image-localization model (Gemini-on-Vertex in production).

    Given one rendered page image and the fields routed to that page, return one
    :class:`CitationLocalizationResult` per requested field. Implementations must
    not invent or revise values — they only locate already-extracted values.
    """

    async def localize_page(
        self, image_png: bytes, page: int, targets: list[CitationPageTarget]
    ) -> list[CitationLocalizationResult]: ...


# --- Normalization (kept aligned with the downstream benchmark scorer) ---------------


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower() if s not in (None, "") else ""


def value_supported_by_text(value: Any, text: Any) -> bool:
    """Does ``text`` (the model's verbatim source span) support ``value``?

    Primary check mirrors the scorer's ``_supported_by_evidence`` (normalized
    value is a substring of the normalized source text). Adds the reverse
    containment (the span may be tighter than the printed value) and a
    digit-subsequence check so an id/date/amount whose punctuation differs from
    the visible text (``$124.00`` vs ``124 00``) is still accepted.
    """
    nv, nt = _norm(value), _norm(text)
    if not nv or not nt:
        return False
    if nv in nt or nt in nv:
        return True
    dv, dt = re.sub(r"\D", "", str(value)), re.sub(r"\D", "", str(text))
    return len(dv) >= _MIN_DIGIT_SUPPORT and dv in dt


# --- Leaf flattening (path convention matches schema_extract._flatten) -------


def default_field_selector(path: str, value: Any) -> bool:
    """Benchmark v1: cite every non-null leaf — the benchmark decides which
    fields have citation gold, so over-selecting here costs calls, not accuracy."""
    return value not in (None, "")


def iter_leaf_values(values: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Walk a value tree → ``(path, scalar)`` for every leaf, ``a.b`` / ``a[0]``
    keyed exactly like the evidence dict ``schema_extract`` produces."""
    if isinstance(values, dict):
        for k, v in values.items():
            yield from iter_leaf_values(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(values, list):
        for i, v in enumerate(values):
            yield from iter_leaf_values(v, f"{prefix}[{i}]")
    elif prefix:
        yield prefix, values


# --- Indexed chunks (for candidate-page + fallback occurrence search) --------


@dataclass
class _PageChunk:
    text: str
    norm_text: str
    page: int
    bbox: list[float] | None


def _index_chunks(
    chunks: list[Chunk], page_sizes: list[tuple[float, float]]
) -> list[_PageChunk]:
    out: list[_PageChunk] = []
    for c in chunks or []:
        text = re.sub(r"\s+", " ", (c.page_content or "")).strip()
        if not text:
            continue
        page = c.page_no or 0
        size = page_sizes[page - 1] if 0 < page <= len(page_sizes) else None
        out.append(_PageChunk(text=text, norm_text=text.lower(), page=page,
                              bbox=normalize_bbox(c.bbox, size)))
    return out


def _occurrence_pages(value: Any, indexed: list[_PageChunk]) -> tuple[list[int], bool]:
    """Pages whose chunk text visibly contains ``value`` (deterministic fallback
    page routing when the first pass gives no usable page). Returns ``(pages,
    truncated)``."""
    nv = _norm(value)
    if not nv:
        return [], False
    pages: list[int] = []
    seen: set[int] = set()
    for c in indexed:
        if c.page in seen:
            continue
        if nv in c.norm_text:
            seen.add(c.page)
            pages.append(c.page)
    pages.sort()
    if len(pages) > MAX_CANDIDATE_PAGES:
        return pages[:MAX_CANDIDATE_PAGES], True
    return pages, False


def _occurrence_evidence(
    value: Any, indexed: list[_PageChunk], pages: set[int] | None = None
) -> list[FieldEvidence]:
    """Parse-derived evidence (chunk box + text) for every chunk that contains
    ``value``. The deterministic fallback when image localization is rejected and
    there is no first-pass evidence (the ``{value,pages}`` arm)."""
    nv = _norm(value)
    if not nv:
        return []
    ev: list[FieldEvidence] = []
    for c in indexed:
        if pages is not None and c.page not in pages:
            continue
        if nv in c.norm_text and c.bbox is not None:
            ev.append(FieldEvidence(page=c.page, bbox=c.bbox, text=c.text[:300]))
    return ev


# --- M1 public surface -------------------------------------------------------


def build_citation_targets(
    values: dict[str, Any],
    evidence: dict[str, list[FieldEvidence]],
    chunks: list[Chunk],
    page_sizes: list[tuple[float, float]],
    *,
    selector=default_field_selector,
    page_hints: dict[str, list[int]] | None = None,
) -> list[CitationTarget]:
    """Flatten non-null leaves into :class:`CitationTarget`s with candidate pages.

    Candidate pages come from (in order): the first-pass ``page_hints`` for that
    path (the ``{value,pages}`` arm), else the pages of the first-pass
    ``evidence`` (current-schema-pass arms), else a deterministic value-occurrence
    search over the parse chunks (the fallback). An invalid/empty/over-broad page
    list falls through to occurrence search; an over-long list is capped and
    flagged ``candidate_pages_truncated``.
    """
    indexed = _index_chunks(chunks, page_sizes)
    n_pages = len(page_sizes)
    targets: list[CitationTarget] = []
    for path, value in iter_leaf_values(values):
        if not selector(path, value):
            continue
        ev = (evidence or {}).get(path) or []
        quote = ev[0].text if ev else None
        fp_bbox = ev[0].bbox if ev else None

        pages: list[int] = []
        truncated = False
        if page_hints is not None:
            hinted = [p for p in (page_hints.get(path) or [])
                      if isinstance(p, int) and (n_pages == 0 or 1 <= p <= n_pages)]
            pages = sorted(dict.fromkeys(hinted))
            if len(pages) > MAX_CANDIDATE_PAGES:
                pages, truncated = pages[:MAX_CANDIDATE_PAGES], True
        else:
            pages = sorted({e.page for e in ev
                            if isinstance(e.page, int) and (n_pages == 0 or 1 <= e.page <= n_pages)})

        if not pages:  # deterministic fallback routing
            pages, truncated = _occurrence_pages(value, indexed)

        if not pages:
            continue  # nothing to localize against — leave to fallback evidence only
        targets.append(CitationTarget(
            path=path, value=value, candidate_pages=pages,
            first_pass_quote=quote, first_pass_bbox=fp_bbox,
            candidate_pages_truncated=truncated,
        ))
    return targets


def expand_page_targets(targets: list[CitationTarget]) -> list[CitationPageTarget]:
    """One value on N candidate pages → N independent ``(path, value, page)``
    localization tasks."""
    out: list[CitationPageTarget] = []
    for t in targets:
        for page in t.candidate_pages:
            out.append(CitationPageTarget(
                path=t.path, value=t.value, page=page, description=t.description,
                first_pass_quote=t.first_pass_quote, first_pass_bbox=t.first_pass_bbox,
            ))
    return out


def partition_citation_targets(
    page_targets: list[CitationPageTarget], fields_per_call: int
) -> list[list[CitationPageTarget]]:
    """Group tasks by page first, then chunk each page's fields into calls of
    ``fields_per_call``. ``fields_per_call <= 0`` (or huge) → one call per page
    with all that page's fields (the monolithic control)."""
    by_page: dict[int, list[CitationPageTarget]] = {}
    for t in page_targets:
        by_page.setdefault(t.page, []).append(t)
    calls: list[list[CitationPageTarget]] = []
    for page in sorted(by_page):
        fields = by_page[page]
        step = fields_per_call if fields_per_call and fields_per_call > 0 else len(fields)
        step = max(1, step)
        for i in range(0, len(fields), step):
            calls.append(fields[i:i + step])
    return calls


def validate_localized_evidence(
    target: CitationPageTarget, result: CitationLocalizationResult
) -> LocalizationOutcome:
    """Validate one localizer result → keep it or say why not (MINIMAL policy,
    2026-07-01 review + audit).

    Rejects only what is structurally unusable: ``found=false``; a malformed /
    non-numeric box; a box (mostly) off the 0–1000 grid; a zero-area box. A
    surviving box is clipped to the grid and returned as page-relative
    :class:`FieldEvidence` (page taken from the target, never the model).

    Deliberately NOT checked (audit of 1,240 localizations, 2026-07-01):
    - ``too_broad`` (> MAX_BOX_AREA_FRACTION): never fired once — Gemini does not
      emit page-sized boxes.
    - ``unsupported_text`` (value must appear in ``source_text``): 152/152 sampled
      rejections were false positives — normalized values vs printed forms
      ('1957-06-03' vs boxed '06-03-1957', 'False' vs boxed 'No') — which silently
      stripped citations from exactly the date/boolean/enum fields that need them.
      A future text check must normalize value forms first; do not reintroduce the
      raw substring test.
    """
    if not result.found:
        return LocalizationOutcome(target.path, target.page, False, "not_found")
    b = result.bbox
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return LocalizationOutcome(target.path, target.page, False, "bad_bbox_shape")
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return LocalizationOutcome(target.path, target.page, False, "bad_bbox_type")
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    # Reject a box that lies (mostly) off the page; a small overflow is clipped.
    if x1 <= 0 or y1 <= 0 or x0 >= COORD_GRID or y0 >= COORD_GRID:
        return LocalizationOutcome(target.path, target.page, False, "out_of_range")
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(COORD_GRID, x1), min(COORD_GRID, y1)
    if x1 <= x0 or y1 <= y0:
        return LocalizationOutcome(target.path, target.page, False, "degenerate")
    ev = FieldEvidence(page=target.page, bbox=[x0, y0, x1, y1],
                       text=(result.source_text or "")[:300])
    return LocalizationOutcome(target.path, target.page, True, "ok", evidence=ev)


def _render_char_w(ch: str) -> float:
    """Approximate proportional render width (same font model as production's
    ``_interp_value_box``): digits/capitals wide, spaces narrow, punctuation
    narrowest. A font model, not a document heuristic — deliberately kept."""
    if ch.isspace():
        return 0.5
    if ch.isdigit() or ch.isupper():
        return 1.1
    if ch.islower():
        return 1.0
    return 0.6


def value_alnum_variants(value: Any) -> list[str]:
    """Alphanumeric-flattened PRINTED forms of a value, most-specific first.

    The plain alnum flatten is order-sensitive: ISO ``1957-06-03`` flattens to
    ``19570603`` but the page prints ``06-03-1957`` → ``06031957`` — no match.
    Variants cover reordered dates and boolean words so normalized values can be
    located in printed text (2026-07-02; same printed-form family as
    ``_value_text_variants`` in the scorer and the SOM candidate matching)."""
    s = str(value).strip()
    out = [re.sub(r"[^a-z0-9]", "", s.lower())]
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = m.groups()
        out += [f"{mo}{d}{y}", f"{int(mo)}{int(d)}{y}", f"{mo}{d}{y[2:]}", f"{int(mo)}{int(d)}{y[2:]}"]
    if isinstance(value, bool) or s.lower() in ("true", "false"):
        truthy = (value is True) or s.lower() == "true"
        out = ["yes"] if truthy else ["no"]
    return [v for v in dict.fromkeys(out) if v]


def doc_typography(chunks: list[Chunk], page_sizes: list[tuple[float, float]]) -> tuple[float, float]:
    """(median line height, median per-weighted-char width) of the document's
    chunks in the 0–1000 frame — the document-adaptive scales that replace the
    absolute-constant guards (24/1000 height, 500/1000 width) of production's
    tightener. Derived from the parse itself, so they track the actual
    typography (fax dpi, font size) instead of assuming letter-page geometry."""
    heights: list[float] = []
    widths: list[float] = []
    for c in _index_chunks(chunks, page_sizes):
        if not c.bbox:
            continue
        h = c.bbox[3] - c.bbox[1]
        if h > 0:
            heights.append(h)
        wsum = sum(_render_char_w(ch) for ch in c.text)
        if wsum > 0 and c.bbox[2] > c.bbox[0]:
            widths.append((c.bbox[2] - c.bbox[0]) / wsum)
    def med(xs: list[float]) -> float:
        return sorted(xs)[len(xs) // 2] if xs else 0.0

    return med(heights), med(widths)


def tighten_value_box(
    value: Any,
    text: str,
    bbox: list[float] | None,
    *,
    med_line_h: float,
    med_char_w: float,
) -> list[float] | None:
    """Generalized char-span box tightener (2026-07-02 review request).

    Same idea as production's ``_interp_value_box`` — locate the value's char
    span in the chunk's text and interpolate its x-slice across the box with
    proportional font widths — with two generalizations:

    1. **No absolute page-fraction constants.** Single-line detection uses the
       DOCUMENT's median line height (≤ 1.8×); the wide-form-row bail is
       replaced by a hidden-gap DENSITY check: if the box implies a per-char
       width far above the document median (≥ 2.5×), the line has large
       collapsed gaps (multi-field row) and interpolation would misplace —
       this also correctly tightens wide-but-dense lines the absolute 500-width
       rule refused, and refuses narrow-but-sparse rows it wrongly accepted.
    2. **Printed-form span matching** via :func:`value_alnum_variants`, so
       ISO dates and booleans tighten to their printed forms.

    Bails to ``None`` (keep the coarse box) whenever assumptions fail — never
    worse than the untightened box. The unique-occurrence rule is kept: if the
    value's variants match more than one distinct span, don't guess.
    """
    if bbox is None or not text:
        return None
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    if med_line_h > 0 and (y1 - y0) > 1.8 * med_line_h:
        return None  # multi-line: 1-D char offset doesn't map to a 2-D position
    wsum = sum(_render_char_w(ch) for ch in text)
    if wsum <= 0:
        return None
    if med_char_w > 0 and (x1 - x0) / wsum > 2.5 * med_char_w:
        return None  # hidden collapsed gaps (multi-field row): interpolation misplaces
    keep = [(c.lower(), i) for i, c in enumerate(text) if c.isalnum()]
    if not keep:
        return None
    flat = "".join(c for c, _ in keep)
    spans: set[tuple[int, int]] = set()
    for v in value_alnum_variants(value):
        start = 0
        while True:
            pos = flat.find(v, start)
            if pos < 0:
                break
            spans.add((keep[pos][1], keep[pos + len(v) - 1][1] + 1))
            start = pos + 1
    if len(spans) != 1:
        return None  # absent or ambiguous — don't guess
    s, e = next(iter(spans))
    cum = [0.0]
    for ch in text:
        cum.append(cum[-1] + _render_char_w(ch))
    total = cum[-1]
    if total <= 0:
        return None
    w = x1 - x0
    vx0 = max(x0, min(x0 + (cum[s] / total) * w, x1))
    vx1 = max(x0, min(x0 + (cum[e] / total) * w, x1))
    if vx1 - vx0 < 1.0:
        return None
    return [vx0, y0, vx1, y1]


def parse_occurrence_evidence(
    values: dict[str, Any],
    chunks: list[Chunk],
    page_sizes: list[tuple[float, float]],
    *,
    selector=default_field_selector,
) -> dict[str, list[FieldEvidence]]:
    """Deterministic parse-derived evidence for every selected non-null leaf, by
    value-occurrence search over chunks. Used as the fallback ``old_evidence`` for
    the ``{value,pages}`` arm, which has no first-pass quote/box evidence."""
    indexed = _index_chunks(chunks, page_sizes)
    out: dict[str, list[FieldEvidence]] = {}
    for path, value in iter_leaf_values(values):
        if not selector(path, value):
            continue
        ev = _occurrence_evidence(value, indexed)
        if ev:
            out[path] = ev[:16]
    return out


def merge_localized_evidence(
    values: dict[str, Any],
    old_evidence: dict[str, list[FieldEvidence]],
    localized: list[LocalizationOutcome],
    fallback_policy: str = "parse",
) -> dict[str, list[FieldEvidence]]:
    """Merge validated localizations into the final evidence dict (values are
    never changed).

    For every path that was attempted: keep its valid localized boxes; if none
    survived, fall back to ``old_evidence[path]`` when ``fallback_policy='parse'``,
    or drop it when ``'localized-only'``. Paths that were never attempted (not
    selected for citation) keep their existing evidence unchanged.
    """
    attempted: set[str] = {o.path for o in localized}
    valid_by_path: dict[str, list[FieldEvidence]] = {}
    for o in localized:
        if o.valid and o.evidence is not None:
            valid_by_path.setdefault(o.path, []).append(o.evidence)

    out: dict[str, list[FieldEvidence]] = {}
    for path in attempted:
        if path in valid_by_path:
            out[path] = valid_by_path[path]
        elif fallback_policy == "parse" and (old_evidence or {}).get(path):
            out[path] = old_evidence[path]
        # localized-only with no valid box → no evidence for this path
    for path, ev in (old_evidence or {}).items():
        if path not in attempted and ev:
            out[path] = ev
    return out
