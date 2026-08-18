"""Cloud Vision citation-box tightening — an optional pass 3 of schema extraction.

Pass 1 (the model) answers WHAT each field's value is; the parse-citation path
attaches a COARSE box (the whole cited chunk — a row, paragraph, or table band).
This pass shrinks that box to just the printed value: it renders the evidence
page, crops the coarse region, runs Google Cloud Vision ``document_text_detection``
on the crop (per-word boxes), finds the run of consecutive words whose
alphanumeric concat equals the value, and maps that span back to page coordinates.

Guarantees (same contract as the parse-citation path):
- **Value-preserving** — only a citation's ``bbox`` can change, never a value,
  never its page.
- **Shrink-only** — an accepted box must be strictly smaller than the coarse one.
- **Fail-open** — a field that doesn't match, a Vision error, missing creds, or
  any pass-level exception keeps the coarse box; the pass returns the count it
  tightened and never raises into extraction.

This is the third geometry option alongside ``value_locate`` (deterministic word
geometry) and ``schema_citation_localize`` (Gemini image localizer). It is gated
by :data:`~extract.config.Settings.EXTRACT_VISION_TIGHTEN_ENABLED` and wired into
the schema-extract route; when the flag is off it is never constructed or called.
"""
from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from extract.config import settings
from extract.core.models import FieldEvidence

#: Coordinate grid of ``FieldEvidence.bbox`` (0–1000, page-relative, top-left origin).
_COORD = 1000.0
#: Page render scale for the crop (2× is enough for Vision on face-sheet text).
_RENDER = 2.0
#: A crop smaller than this (px) can't hold a value — skip.
_MIN_CROP_PX = 4


def _norm(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s or "")).upper()


def _date_key(s: Any) -> tuple[int, ...] | None:
    nums = re.findall(r"\d+", str(s or ""))
    if len(nums) < 3:
        return None
    parts = [(int(n) + 1900 if len(n) == 2 and int(n) > 31 else int(n)) for n in nums]
    return tuple(sorted(parts))


def _span_box(words: list[tuple[str, int, int, int, int]]) -> tuple[int, int, int, int]:
    xs = [w[1] for w in words] + [w[1] + w[3] for w in words]
    ys = [w[2] for w in words] + [w[2] + w[4] for w in words]
    return (min(xs), min(ys), max(xs), max(ys))


def _find_span(value: Any, words: list[tuple[str, int, int, int, int]]):
    """words = [(text, left, top, w, h)] in crop px. Return the box of the shortest
    run of consecutive words whose alnum-concat == value, else a 3-word date match."""
    target = _norm(value)
    if not target:
        return None
    for i in range(len(words)):
        if not _norm(words[i][0]):
            continue
        acc, run = "", []
        for j in range(i, len(words)):
            if not _norm(words[j][0]):
                continue
            acc += _norm(words[j][0])
            run.append(words[j])
            if acc == target:
                return _span_box(run)
            if not target.startswith(acc):
                break
    dk = _date_key(value)
    if dk is not None:
        for i in range(len(words)):
            for j in range(i, min(i + 3, len(words))):
                win = words[i : j + 1]
                if _date_key(" ".join(w[0] for w in win)) == dk:
                    return _span_box(win)
    return None


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class CloudVisionTightener:
    """Wraps a Google Cloud Vision client. Constructed only when the flag is on.

    Uses Application Default Credentials / ``GOOGLE_APPLICATION_CREDENTIALS`` — the
    same GCP service account as Vertex, with the Vision API enabled on the project.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _c(self):
        if self._client is None:
            from google.cloud import vision

            # Match the Gemini/Vertex client's auth: prod injects the service-account
            # key as GOOGLE_VERTEX_SA_JSON (Secrets Manager), not a GOOGLE_APPLICATION_
            # CREDENTIALS file — without this the client would find no ADC in ECS and the
            # whole pass would silently fail-open. Falls back to ADC locally.
            sa_json = settings.GOOGLE_VERTEX_SA_JSON
            if sa_json:
                import json

                from google.oauth2.service_account import Credentials

                creds = Credentials.from_service_account_info(
                    json.loads(sa_json),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self._client = vision.ImageAnnotatorClient(credentials=creds)
            else:
                self._client = vision.ImageAnnotatorClient()
        return self._client

    def _words(self, crop_png: bytes) -> list[tuple[str, int, int, int, int]]:
        from google.cloud import vision

        r = self._c().document_text_detection(image=vision.Image(content=crop_png))
        out: list[tuple[str, int, int, int, int]] = []
        for p in r.full_text_annotation.pages:
            for b in p.blocks:
                for par in b.paragraphs:
                    for w in par.words:
                        t = "".join(s.text for s in w.symbols)
                        xs = [v.x for v in w.bounding_box.vertices]
                        ys = [v.y for v in w.bounding_box.vertices]
                        out.append((t, min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
        return out

    def tighten_box(self, page_img, value, bbox) -> list[float] | None:
        """page_img: full-page PIL.Image; bbox: 0–1000. Return a shrunk 0–1000 box
        or None (keep coarse). Never raises."""
        try:
            W, H = page_img.size
            cpx = (
                max(0, int(bbox[0] / _COORD * W)),
                max(0, int(bbox[1] / _COORD * H)),
                int(bbox[2] / _COORD * W),
                int(bbox[3] / _COORD * H),
            )
            if _norm(value) == "" or cpx[2] - cpx[0] < _MIN_CROP_PX or cpx[3] - cpx[1] < _MIN_CROP_PX:
                return None
            buf = io.BytesIO()
            page_img.crop(cpx).save(buf, "PNG")
            span = _find_span(value, self._words(buf.getvalue()))
            if span is None:
                return None
            px = [cpx[0] + span[0], cpx[1] + span[1], cpx[0] + span[2], cpx[1] + span[3]]
            new = [px[0] / W * _COORD, px[1] / H * _COORD, px[2] / W * _COORD, px[3] / H * _COORD]
            return new if _area(new) < _area(bbox) else None
        except Exception:  # noqa: BLE001 — fail-open, keep the coarse box
            return None


def _flatten_values(values: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(values, dict):
        for k, v in values.items():
            out.update(_flatten_values(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(values, list):
        for i, v in enumerate(values):
            out.update(_flatten_values(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = values
    return out


def tighten_evidence(
    evidence: dict[str, list[FieldEvidence]] | None,
    values: Any,
    doc_bytes: bytes,
    *,
    tightener: CloudVisionTightener | None = None,
    concurrency: int | None = None,
) -> int:
    """Shrink each field's citation box to its value via Cloud Vision, IN PLACE.

    Renders each evidence page once, then tightens every field on that page in a
    thread pool. Returns the number of boxes tightened. Fail-open: any error keeps
    everything and returns the count so far (or 0)."""
    if not evidence or not doc_bytes:
        return 0
    try:
        import pymupdf
        from PIL import Image

        pymupdf.TOOLS.mupdf_display_errors(False)
    except Exception:  # noqa: BLE001
        return 0

    tightener = tightener or CloudVisionTightener()
    flat = _flatten_values(values)
    # collect (path, value, bbox) grouped by page
    by_page: dict[int, list[tuple[str, Any, list[float]]]] = {}
    for path, evs in evidence.items():
        if not evs:
            continue
        e = evs[0]
        pg, box = getattr(e, "page", None), getattr(e, "bbox", None)
        val = flat.get(path)
        if pg and box and val not in (None, ""):
            by_page.setdefault(int(pg), []).append((path, val, box))
    if not by_page:
        return 0

    n = 0
    try:
        doc = pymupdf.open(stream=doc_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001
        return 0
    workers = concurrency or settings.EXTRACT_VISION_TIGHTEN_CONCURRENCY
    try:
        for pg, fields in by_page.items():
            if not (1 <= pg <= doc.page_count):
                continue
            pix = doc[pg - 1].get_pixmap(matrix=pymupdf.Matrix(_RENDER, _RENDER), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(tightener.tighten_box, img, v, b): p for p, v, b in fields}
                for fu in futs:
                    p = futs[fu]
                    nb = fu.result()
                    if nb:
                        evidence[p][0].bbox = nb
                        n += 1
    finally:
        doc.close()
    return n
