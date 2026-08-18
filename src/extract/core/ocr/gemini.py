"""Gemini OCR provider — the fallback for :class:`QwenLoraProvider`.

This mirrors the shared first-pass parse-GT labeler prompt from
``extract.parse_prompts``: semantic layout blocks, tight bboxes, signatures as
images, and Gemini's native YXYX coordinate order. The fallback keeps that same
labeling policy because it is the page-level backstop when Qwen output is
unusable (blank / looping). The model is ``gemini-3.5-flash`` — GA, because
Pre-GA models sit outside the Cloud BAA and this path carries PHI.

Output is normalized to the same ``OCRPageResult`` contract as
``QwenLoraProvider`` so the extractor cannot tell which backend produced a page.

Transport: Vertex AI (``aiplatform.googleapis.com``) only — the Google surface
covered by the Cloud BAA, required because customer pages can contain PHI.
There is deliberately no Developer-API fallback: a missing ``GOOGLE_VERTEX_*``
configuration fails loudly instead of silently routing pages to a non-BAA
endpoint.

Gemini emits ``bbox_2d = [y_min, x_min, y_max, x_max]`` normalized 0-1000 (its
native object-detection order); we transpose to ``[x0, y0, x1, y1]`` and scale to
page points here. The ``type`` field ("text" | "table" | "image") is explicit, so
unlike the Qwen path we don't infer element kind from ``text_content`` — though
the conventions match (tables = GFM markdown, images = ``"<image>"``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from extract.config import settings
from extract.core.ocr.base import (
    OCRBlock,
    OCRFigure,
    OCRKeyValue,
    OCRPageResult,
    OCRTable,
)
from extract.core.ocr.qwen_lora import (
    _HANDWRITING_MARKER,
    _IMAGE_SENTINEL,
    _looks_like_markdown_table,
    _table_from_markdown,
)
from extract.logger import get_logger
from extract.parse_prompts import GEMINI_PARSE_GT_NATIVE_YXYX_PROMPT

logger = get_logger()

_MISSING = object()

# Back-compat name retained for tests and diagnostic scripts. This is now the
# shared parse-GT prompt, not the older one-record-per-line Gemini prompt.
GEMINI_NATIVE_PROMPT = GEMINI_PARSE_GT_NATIVE_YXYX_PROMPT

# Content judgment for champion-empty pages (mirrors the near_blank gold
# convention the champion was trained/gated on: isolated artifacts are not
# content). One word out; parsed strictly.
_JUDGE_PAGE_CONTENT_PROMPT = (
    "You are inspecting one scanned page. Does this page contain CLEARLY "
    "LEGIBLE document content to transcribe - sentences, paragraphs, form "
    "fields with values, tables with data, or informative headers that a "
    "careful human could read with confidence directly from these pixels? "
    "Be strict about legibility: smudges, bleed-through, ghosted or broken "
    "text fragments that you would have to GUESS at do NOT count, even if "
    "you can imagine what they might say. Isolated page numbers, fax edge "
    "artifacts, scanner noise, stray marks, a lone stamp or watermark "
    "(including a degraded diagonal stamp), or blank ruled/signature lines "
    "do NOT count as content. If you are not certain the text is legible, "
    "answer no. Answer with exactly one word: yes or no."
)

# Gemini structured-output schema (mirrors scripts/pilot_label_3way.py::call_gemini).
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "bbox_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
            "type": {"type": "STRING", "enum": ["text", "table", "image", "kv"]},
            "text_content": {"type": "STRING"},
        },
        "required": ["bbox_2d", "type", "text_content"],
    },
}

_COORD_SCALE = 1000.0


class GeminiOcrProvider:
    """OCR via Gemini with the shared parse-GT prompt. Used as the Qwen fallback."""

    name = "gemini"

    def __init__(
        self,
        *,
        model: str | None = None,
        thinking_budget: int | None | object = _MISSING,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._model = model or settings.GEMINI_OCR_MODEL
        self._thinking_budget = (
            settings.GEMINI_OCR_THINKING_BUDGET if thinking_budget is _MISSING else thinking_budget
        )
        self._max_output_tokens = max_output_tokens or settings.GEMINI_OCR_MAX_OUTPUT_TOKENS
        self._timeout_s = timeout_s or settings.GEMINI_OCR_TIMEOUT_SECONDS
        # Built once and reused: the Vertex client holds OAuth credentials whose
        # token is cached until expiry; rebuilding per page would add a token
        # round-trip to every fallback call.
        self._client: Any = None

    def _make_client(self, genai: Any) -> Any:
        """Build the GenAI client — Vertex AI or Gemini API key (genai_client.py)."""
        from extract.genai_client import make_genai_client

        return make_genai_client(genai)

    async def judge_page_has_content(self, image_bytes: bytes) -> bool | None:
        """One-word content judgment for champion-empty pages (plan 087 follow-up).

        Used when the champion read a page as empty but a retry/fallback produced
        a nonempty override: the override may only ship if this judge affirms the
        page has transcribable content. Asks a strict yes/no (temperature 0, no
        thinking, tiny output cap) on the SAME BAA Vertex transport as the OCR
        fallback. Returns ``True``/``False`` on a clean verdict, ``None`` on any
        error or ambiguous reply — callers treat ``None`` as fail-open to the
        legacy behavior (ship the override) so a judge outage can never suppress
        real content.
        """
        try:
            from google import genai
            from google.genai import types as gt

            if self._client is None:
                self._client = self._make_client(genai)
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=[
                        gt.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        gt.Part.from_text(text=_JUDGE_PAGE_CONTENT_PROMPT),
                    ],
                    config=gt.GenerateContentConfig(
                        max_output_tokens=8,
                        temperature=0,
                        thinking_config=gt.ThinkingConfig(thinking_budget=0),
                    ),
                ),
                timeout=min(self._timeout_s, 45.0),
            )
            word = (response.text or "").strip().lower()
            if word.startswith("yes"):
                return True
            if word.startswith("no"):
                return False
            return None
        except Exception:  # noqa: BLE001 — judge failure = fail-open to legacy
            logger.debug("gemini content judge failed", exc_info=True)
            return None

    async def ocr_page(
        self,
        image_bytes: bytes,
        *,
        page_width: float,
        page_height: float,
    ) -> OCRPageResult:
        from google import genai
        from google.genai import types as gt

        if self._client is None:
            self._client = self._make_client(genai)
        client = self._client
        cfg_kwargs: dict[str, Any] = dict(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=0,
        )
        if self._thinking_budget is not None:
            cfg_kwargs["thinking_config"] = gt.ThinkingConfig(thinking_budget=self._thinking_budget)

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=self._model,
                contents=[
                    gt.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    gt.Part.from_text(text=GEMINI_NATIVE_PROMPT),
                ],
                config=gt.GenerateContentConfig(**cfg_kwargs),
            ),
            timeout=self._timeout_s,
        )
        raw = response.text or ""
        result = parse_gemini_response(raw, page_width=page_width, page_height=page_height)
        # Keep the raw model text for internal diagnostics only (the
        # Gemini-fallback review bundle). Never serialized into the customer
        # response or any log/metric — see ``OCRPageResult.raw``.
        result.raw = raw
        return result


def parse_gemini_response(
    raw: str,
    *,
    page_width: float,
    page_height: float,
) -> OCRPageResult:
    """Parse Gemini's native-YXYX JSON array into an ``OCRPageResult``.

    Coordinates arrive as ``[y_min, x_min, y_max, x_max]`` in 0-1000; we transpose
    to ``[x0, y0, x1, y1]`` and scale to page points. Element kind comes from the
    explicit ``type`` field, with the same table/image conventions as Qwen.
    """
    records = _parse_records(raw)
    # 2026-07-30 reading-order fix (7ccdaa202, extended to this provider):
    # records keep Gemini's own EMISSION order and carry ``seq``, exactly as the
    # Qwen path does. Gemini reads in reading order too, so the old bbox-center
    # (y, x) sort here interleaved the columns of every 2-column page AND, with
    # ``seq`` unset, dropped the page into the legacy type-bucketed assembly
    # (``pdf._chunks_from_ocr_page_result``) — both halves of the −0.137 mean
    # text-accuracy bug, on the default fallback path.

    blocks: list[OCRBlock] = []
    tables: list[OCRTable] = []
    figures: list[OCRFigure] = []
    key_values: list[OCRKeyValue] = []
    for i, (x0, y0, x1, y1, text, kind) in enumerate(records):
        bbox = _yxyx_to_points(x0, y0, x1, y1, page_width=page_width, page_height=page_height)
        text = (text or "").strip()
        # confidence=None: Gemini exposes no token-level signal here; a placeholder
        # 0.0 would be silently dropped by the old ``or None`` mapping and would
        # shadow real confidences now that the field is meaningful (plan 028 B3).
        # The explicit KV discriminator wins over ALL content inference (parity
        # with the Qwen path): a record tagged ``type:"kv"`` is a KV region
        # regardless of its text.
        if kind == "kv":
            if text:
                key_values.append(OCRKeyValue(text=text, bbox=bbox, confidence=None, seq=i))
            continue
        if kind == "image" or text == _IMAGE_SENTINEL:
            figures.append(OCRFigure(bbox=bbox, confidence=None, seq=i))
            continue
        if kind == "table" or _looks_like_markdown_table(text):
            table = _table_from_markdown(text=text, bbox=bbox, confidence=None)
            table.seq = i
            tables.append(table)
            continue
        if text.startswith(_HANDWRITING_MARKER):
            # Strip the handwriting sentinel; the read value stays a clean text block.
            text = text[len(_HANDWRITING_MARKER):].strip()
        if text:
            blocks.append(OCRBlock(text=text, bbox=bbox, confidence=None, seq=i))

    return OCRPageResult(
        blocks=blocks, tables=tables, figures=figures, key_values=key_values
    )


def _parse_records(raw: str) -> list[tuple[int, int, int, int, str, str]]:
    """Return ``(x0, y0, x1, y1, text, kind)`` tuples in transposed XYXY 0-1000.

    Gemini emits ``bbox_2d = [y_min, x_min, y_max, x_max]``; we transpose to XYXY
    here so every downstream consumer sees the same box convention as the Qwen
    path. Records are returned in EMISSION order — the caller stamps ``seq``
    from it and must not re-sort (2026-07-30 reading-order fix).
    """
    s = raw.strip()
    end = s.rfind("]")
    candidate = s[: end + 1] if end != -1 else s
    try:
        parsed: Any = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Gemini OCR: unparseable JSON response (%d chars)", len(raw))
        return []
    if not isinstance(parsed, list):
        return []

    out: list[tuple[int, int, int, int, str, str]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        bb = entry.get("bbox_2d") or entry.get("bbox")
        text = entry.get("text_content")
        if text is None:
            text = entry.get("text")
        kind = str(entry.get("type") or "text").strip().lower()
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        if not isinstance(text, str):
            continue
        try:
            y_min, x_min, y_max, x_max = (int(round(float(v))) for v in bb)
        except (TypeError, ValueError):
            continue
        out.append((x_min, y_min, x_max, y_max, text.strip(), kind))
    return out


def _yxyx_to_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    page_width: float,
    page_height: float,
) -> list[float]:
    """Transposed-XYXY 0-1000 → page points (top-left origin), clipped & ordered."""
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [
        _clip(x0 / _COORD_SCALE * page_width, page_width),
        _clip(y0 / _COORD_SCALE * page_height, page_height),
        _clip(x1 / _COORD_SCALE * page_width, page_width),
        _clip(y1 / _COORD_SCALE * page_height, page_height),
    ]


def _clip(value: float, hi: float) -> float:
    return max(0.0, min(hi, value))
