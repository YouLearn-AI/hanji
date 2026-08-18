"""Tier-2 low-confidence reconciler (plan 028 workstream B, gate consumer).

When a short high-stakes value (member IDs) comes back below the confidence
gate (``ChunkConfidence.min_confidence``; see ``chunk_confidence``), the field
is re-read from an isolated, upscaled crop of its cited bbox by one or more
INDEPENDENT readers, and reconciled against the original parse by AGREEMENT —
never by blindly trusting a single re-read:

  - parse confidence at/above the gate         -> trust parse (no re-read)
  - all readers agree WITH the parse value     -> confirm (harmless trigger)
  - all readers agree with EACH OTHER, ≠ parse -> recover to that value
  - readers disagree                           -> flag for human, keep parse

The last branch is the safety guard: a genuinely illegible field (e.g. a mushy
fax digit) makes independent readers diverge, and we must NOT let one of them
overwrite the value with its own wrong guess. Detecting "parse is wrong" and
routing to review still beats silently shipping a wrong digit (the audited customer's
0.1%-wrong-digit bar). ``reconcile`` is pure (it takes reader outputs, not a
network client) so the decision logic is deterministically testable; the crop
+ reader IO lives in ``crop_region`` / the reader callables below.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["trusted", "confirmed", "recovered", "flagged"]


def _extract_id(value: str | None) -> str:
    """Pull the ID-shaped token out of a reader's raw output, dropping any label
    text a crop read carries (e.g. ``"Medicare Beneficiary ID | 2P17XX8XK68"`` ->
    ``"2P17XX8XK68"``). The ID token is the longest digit-bearing run of
    alphanumerics + internal dashes/spaces; separators (incl. ``|``, newlines,
    ``:``) bound tokens. Formatting (dashes) is preserved on the winner."""
    if not value:
        return ""
    best = ""
    best_len = 0
    for m in re.finditer(r"[A-Za-z0-9](?:[A-Za-z0-9\- ]*[A-Za-z0-9])?", value):
        tok = m.group().strip()
        flat = re.sub(r"[^A-Za-z0-9]", "", tok)
        if any(c.isdigit() for c in flat) and len(flat) > best_len:
            best, best_len = tok, len(flat)
    return best or value.strip()


def _norm(value: str | None) -> str:
    """Compare IDs on alphanumerics only, case-folded, after extracting the ID
    token — labels/separators/spacing never decide agreement
    (``2123-21002`` == ``212321002``, ``"Policy #: 1EG4-TE5"`` == ``1EG4TE5``)."""
    return re.sub(r"[^A-Za-z0-9]", "", _extract_id(value)).upper()


@dataclass(frozen=True)
class Reconciliation:
    action: Action
    value: str | None            # the value to EMIT (always the parse value unless auto_recover)
    needs_review: bool
    parse_value: str | None
    reader_values: tuple[str, ...] = field(default_factory=tuple)
    suggested_value: str | None = None  # the re-reader's cleaned candidate, shown to the reviewer


def reconcile(
    parse_value: str | None,
    parse_confidence: float | None,
    reader_values: Sequence[str | None],
    *,
    gate: float = 0.90,
    auto_recover: bool = False,
) -> Reconciliation:
    """Reconcile a parsed value against the second-pass re-read(s). See module docstring.

    Default policy (``auto_recover=False``): the re-read is used to *detect* a bad
    parse, not to silently replace it. A disagreement is FLAGGED for human review
    with the re-reader's value surfaced as ``suggested_value`` — because a
    confidently-stable re-read can still be wrong on an illegible field (r61615),
    so we never auto-overwrite. Set ``auto_recover=True`` only when the readers'
    unanimous value can be trusted outright.

    ``reader_values`` are raw re-read outputs; ``_norm``/``_extract_id`` strip any
    label text and separators before comparison.
    """
    # Gate not tripped: the parse read is confident enough to trust as-is.
    if parse_confidence is None or parse_confidence >= gate:
        return Reconciliation("trusted", parse_value, False, parse_value)

    reads = [r for r in reader_values if r and _norm(r)]
    if not reads:
        # gate tripped but no usable re-read -> can't confirm, don't guess
        return Reconciliation("flagged", parse_value, True, parse_value, tuple())

    norms = {_norm(r) for r in reads}
    pn = _norm(parse_value)
    reads_t = tuple(reads)

    if norms == {pn}:
        # every reader reproduced the parse value -> low-conf was a false alarm
        return Reconciliation("confirmed", parse_value, False, parse_value, reads_t)

    # Disagreement with parse. A suggestion is offered only when the readers
    # concur (single reader trivially concurs); if they split, there is nothing
    # trustworthy to suggest.
    unanimous = len(norms) == 1
    suggestion = _extract_id(reads[0]) if unanimous else None

    if auto_recover and unanimous:
        return Reconciliation("recovered", suggestion, False, parse_value, reads_t, suggestion)
    # Default: keep the parse value, flag for human review, surface the suggestion.
    return Reconciliation("flagged", parse_value, True, parse_value, reads_t, suggestion)


_REREAD_PROMPT = ("Transcribe ONLY the insurance/member/policy ID value in this image, "
                  "exactly as written (letters, digits, dashes). Output just the value.")

#: Cached Vertex genai client (thread-safe for calls; built once, reused across
#: the concurrent per-field re-reads).
_VERTEX_CLIENT = None


def _vertex_client():
    """GenAI client (Vertex AI or Gemini API key — see genai_client.py), built
    once and reused. Returns None if neither transport is configured."""
    global _VERTEX_CLIENT
    if _VERTEX_CLIENT is not None:
        return _VERTEX_CLIENT
    from extract.genai_client import genai_configured, make_genai_client

    if not genai_configured():
        return None
    _VERTEX_CLIENT = make_genai_client()
    return _VERTEX_CLIENT


def gemini_reader(crop_bytes: bytes, *, model: str = "gemini-2.5-flash-lite") -> str:
    """Second-pass re-read of an isolated crop via Gemini on **Vertex** (PHI-safe).
    ``temperature=0`` is MANDATORY: unpinned, Gemini samples and flips ambiguous
    glyphs run-to-run (a ``D``/``0`` correctness landmine we hit and pinned).
    Thinking off + tiny output budget (the value is short) for latency. Returns the
    raw text; the caller normalizes via ``_extract_id``."""
    from google.genai import types as gt

    client = _vertex_client()
    if client is None:
        raise RuntimeError("Vertex not configured (GOOGLE_VERTEX_PROJECT unset)")
    resp = client.models.generate_content(
        model=model,
        contents=[
            gt.Part.from_bytes(data=crop_bytes, mime_type="image/png"),
            gt.Part.from_text(text=_REREAD_PROMPT),
        ],
        config=gt.GenerateContentConfig(
            temperature=0, max_output_tokens=64,
            thinking_config=gt.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (resp.text or "").strip()


def field_confidence(evidence: Sequence[object]) -> float | None:
    """A field's confidence = the MIN over its citations (weakest cited chunk).
    None when no citation carried a confidence."""
    confs = [e.confidence for e in evidence if getattr(e, "confidence", None) is not None]
    return min(confs) if confs else None


def is_high_stakes(path: str, suffixes: Sequence[str]) -> bool:
    """A field path is high-stakes if it ends with any configured suffix
    (e.g. ``".member_id"`` matches ``discoveredInsurances[0].member_id``)."""
    return any(path.endswith(s) for s in suffixes)


@dataclass
class _ReadField:
    path: str
    value: object
    evidence: list


def apply_reread(
    fields: Sequence[_ReadField],
    page_image_for,
    *,
    reader=None,
    gate: float = 0.80,
    high_stakes: Sequence[str] = (".member_id",),
    auto_recover: bool = False,
    max_workers: int = 8,
) -> int:
    """Tier-2 pass over extracted fields (EXTRACTION side). Latency-minimizing:

    1. Build the work-list from IN-MEMORY confidence only — if no high-stakes
       field is below ``gate``, return immediately (``page_image_for`` is never
       called, so the caller never opens/renders the PDF: ~0 added latency on the
       common case).
    2. Render each needed page at most once (sequential, in the caller's thread —
       pymupdf isn't thread-safe).
    3. Run the independent per-field re-reads CONCURRENTLY (network-bound).

    Mutates each affected citation's ``needs_review`` / ``suggested_value`` in
    place; returns the count of fields that tripped the gate.
    """
    if reader is None:
        if _vertex_client() is None:  # Vertex not configured -> inert, no-op
            return 0
        from extract.config import settings
        reader = lambda crop: gemini_reader(crop, model=settings.SCHEMA_REREAD_MODEL)  # noqa: E731

    # 1. work-list — cheap, no rendering
    work = []
    for f in fields:
        if not is_high_stakes(f.path, high_stakes):
            continue
        conf = field_confidence(f.evidence)
        if conf is None or conf >= gate:
            continue
        suspect = min(f.evidence, key=lambda e: e.confidence if e.confidence is not None else 1.0)
        if suspect.bbox is not None:
            # Snapshot bbox + page at work-list time so this pass is race-free even when
            # the box-tightening pass mutates the same evidence bboxes concurrently: the
            # crop uses the snapshot, never the (possibly shrinking) live bbox.
            work.append((f, suspect, conf, tuple(suspect.bbox), suspect.page))
    if not work:
        return 0

    # 2. render only the pages we need, once each, in this thread
    for pg in {pg for *_, pg in work}:
        page_image_for(pg)

    # 3. concurrent crop + re-read (crop/read touch only per-field snapshot bytes)
    def _read(item):
        _f, _suspect, _conf, bbox, page = item
        img = page_image_for(page)  # cached from step 2
        if img is None:
            return item, None
        frac = tuple(v / 1000.0 for v in bbox)
        try:
            return item, reader(crop_region(img, frac))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — a re-read failure must never break extraction
            return item, None

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(max_workers, len(work))) as ex:
        results = list(ex.map(_read, work))

    for (f, _suspect, conf, _bbox, _pg), read in results:
        if read is None:
            continue
        rec = reconcile(str(f.value) if f.value is not None else None, conf, [read],
                        gate=gate, auto_recover=auto_recover)
        if rec.action in ("flagged", "recovered"):
            for e in f.evidence:
                e.needs_review = rec.needs_review
                e.suggested_value = rec.suggested_value
    return len(work)


def reread_evidence(
    evidence: dict[str, list],
    values: object,
    doc_bytes: bytes | None,
    *,
    reader=None,
    gate: float = 0.80,
    high_stakes: Sequence[str] = (".member_id",),
    render_scale: int = 2,
) -> int:
    """Pipeline seam mirroring ``vision_tighten.tighten_evidence`` — the "Pass 4"
    low-confidence member_id re-read. Flattens ``values`` to per-path parse
    values, scopes to high-stakes paths, and lazily renders pages from
    ``doc_bytes`` only when the gate actually trips. The default reader is Gemini
    on Vertex (PHI-safe); returns 0 if Vertex isn't configured. Returns fields
    re-read."""
    if not evidence or not doc_bytes:
        return 0
    from extract.core.vision_tighten import _flatten_values

    flat = _flatten_values(values)
    fields = [_ReadField(p, flat.get(p), evs)
              for p, evs in evidence.items() if is_high_stakes(p, high_stakes)]
    if not fields:
        return 0

    state: dict = {"doc": None, "cache": {}}

    def page_image_for(pg: int) -> bytes | None:
        cache = state["cache"]
        if pg in cache:
            return cache[pg]
        import pymupdf

        if state["doc"] is None:  # opened lazily — never if the gate doesn't trip
            state["doc"] = pymupdf.open(stream=doc_bytes, filetype="pdf")
        try:
            pix = state["doc"][pg - 1].get_pixmap(
                matrix=pymupdf.Matrix(render_scale, render_scale), alpha=False)
            cache[pg] = pix.tobytes("png")
        except Exception:  # noqa: BLE001
            cache[pg] = None
        return cache[pg]

    return apply_reread(fields, page_image_for, reader=reader,
                        gate=gate, high_stakes=high_stakes)


def crop_region(image_bytes: bytes, bbox: tuple[float, float, float, float],
                *, pad: float = 0.012, scale: int = 5) -> bytes:
    """Isolated, upscaled PNG crop of a page-normalized ``bbox`` (x0,y0,x1,y1).

    Isolation + upscale is the entire lift: the same model that misreads a
    handwritten value in full-page context reads it correctly on the crop."""
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes))
    w, h = im.size
    x0, y0, x1, y1 = bbox
    c = im.crop((max(0, int((x0 - pad) * w)), max(0, int((y0 - pad) * h)),
                 min(w, int((x1 + pad) * w)), min(h, int((y1 + pad) * h))))
    c = c.resize((max(1, c.width * scale), max(1, c.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    c.save(buf, "PNG")
    return buf.getvalue()
