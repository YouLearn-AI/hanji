"""OCR provider for the fine-tuned Qwen3-VL parse model.

Talks to any endpoint implementing the /predict contract in serving/server.py
(the published model: https://huggingface.co/hanji-dev/hanji-parse-4b). Sends
one rendered page per call with the production prompt; parses the model's
bbox_2d_json output (normalized 0-1000 coordinates, tables as GFM markdown,
"<image>" sentinel for figures) into OCR blocks/tables."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
from typing import Any

import httpx

from extract.config import settings
from extract.core.ocr.base import (
    OCRBlock,
    OCRFigure,
    OCRKeyValue,
    OCRPageResult,
    OCRTable,
    OCRTableCell,
)
from extract.core.ocr.chunk_confidence import align_chunk_confidences
from extract.logger import get_logger
from extract.parse_prompts import (
    PRODUCTION_BBOX_2D_JSON_PROMPT,
    PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE,
)

logger = get_logger()

_MISSING = object()

# The production bbox_2d_json prompt (shared parse policy in
# ``extract.parse_prompts``): the body the model was trained + served on,
# including the leading ``<image>\n`` prefix.
PROMPT_BBOX_2D_JSON = PRODUCTION_BBOX_2D_JSON_PROMPT
PROMPT_BBOX_2D_JSON_WITH_IMAGE = PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE

# The figure sentinel the model emits for image regions (AGENTS.md §4).
_IMAGE_SENTINEL = "<image>"

# Handwriting marker some checkpoint lineages were trained to emit as a leading
# ``text_content`` prefix (selection-state addendum era). The selection element
# is gone, but the marker is a serving sentinel — never customer text — so it is
# stripped here (like ``<image>``); the handwritten value itself stays a normal
# text block. Checkbox marks the model transcribes (``[x]``/``[ ]``) are plain
# glyphs and pass through untouched.
_HANDWRITING_MARKER = "<hw>"


# A1 (plan 028): the escalated sampling tier the pipeline sends on the ONE Qwen
# retry of an unusable page (repeated-ngram / truncated / empty) when
# ``settings.OCR_RETRY_ESCALATION`` is on. This is the eval-proven tier-1 ladder
# step (``evals2/candidates/model.py`` ``_RETRY_TIERS[0]``) — olmOCR's
# escalate-on-retry pattern; the identical-greedy retry it replaces re-produced
# the same loop almost every time (S0.6: 584 fallback pages/14d).
ESCALATED_RETRY_SAMPLING: dict[str, float | int] = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 50,
    "min_p": 0.02,
    "repetition_penalty": 1.05,
}

# The served model's context window (for reference / sanity bounds).
QWEN_SERVING_CONTEXT_TOKENS = 16384

# A2 retry-only recovery tier (plan 028; re-scoped in plan 035 cycle-4 now that
# the first pass owns a generous budget). GREEDY stays greedy — the A1
# escalation gate FAILED (recovery 0/52, grounded −0.044 regression;
# the internal measurement records). The recovery override is
# the loop-killer ALONE and deliberately does NOT set max_new_tokens, so the
# retry inherits the first-pass budget (now 8192):
#   - no_repeat_ngram=100: the measured loop-killer (degenerate 9->2 on the
#     flagged subset, accuracy ties, provably inert on non-looping pages). This
#     makes the retry genuinely different from the first greedy pass without a
#     cap change — it turns a slow degenerate loop into a fast
#     qwen_repeated_ngram → fallback rather than running to the cap.
#   - NO cap escalation: cycle-4 raised the FIRST pass 4096->8192, which
#     subsumes the recovery's old cap-raising role (it was 6144, sized to the
#     4096 first pass). A direct e2e on the always-truncating docs showed
#     escalating the retry above 8192 served NONE of them — genuinely-too-big
#     pages truncate again and fall back regardless — so the escalation was pure
#     latency on the worst pages with no quality yield. Inheriting 8192 lets
#     those pages fall back fast.
# The served default json_schema (C2) applies to the retry as well, so a
# malformed page also gets one structurally-valid second chance for free.
RECOVERY_RETRY_SAMPLING: dict[str, float | int] = {
    "no_repeat_ngram": 100,
}

# C1 (plan 028): the chunk-contract envelope as a JSON schema for xgrammar
# constrained decoding — the ENVELOPE only. GFM table bodies stay free text
# inside ``text_content`` (standing GFM rule, agent-rules §4); there is no
# ``category`` field. Used by the constrained-decoding spike (C2) and, if C4
# lands on constrained-retry-only, by the A1 retry attempt.
CHUNK_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "text_content": {"type": "string"},
        },
        "required": ["bbox_2d", "text_content"],
        "additionalProperties": False,
    },
}

# Coordinate space the model emits in.
_COORD_SCALE = 1000.0

# Cold-start / transient HTTP statuses worth retrying (min_replica=0 means the
# first request during scale-up is a 503). 4xx are deterministic — never retry.
_RETRY_STATUS = frozenset({502, 503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 2.0

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_+\-]*)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)


class QwenLoraProvider:
    name = "qwen_lora"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        timeout_s: float | None = None,
        max_new_tokens: int | None = None,
        name: str | None = None,
        prompt: str | None = None,
        prompt_with_image: str | None = None,
    ) -> None:
        # Per-instance overrides so a second registration can serve a different
        # endpoint with different prompt text than the primary.
        if name is not None:
            self.name = name
        self._prompt = prompt
        self._prompt_with_image = prompt_with_image
        self._api_key = api_key or settings.PARSE_MODEL_API_KEY
        self._url = str(url or settings.PARSE_MODEL_URL or "").strip()
        self._timeout_s = timeout_s or settings.PARSE_MODEL_TIMEOUT_SECONDS
        self._max_new_tokens = max_new_tokens or settings.PARSE_MODEL_MAX_NEW_TOKENS

    @property
    def endpoint_url(self) -> str:
        if not self._url:
            raise RuntimeError(
                "The parse model endpoint is not configured. Set PARSE_MODEL_URL to a "
                "/predict-compatible serving endpoint (see serving/SELF_HOSTING.md)."
            )
        return self._url

    async def ocr_page(
        self,
        image_bytes: bytes,
        *,
        page_width: float,
        page_height: float,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> OCRPageResult:
        # The production serving prompt: the new-canon body WITH the leading
        # "<image>\n" prefix the model was trained on (one literal text prefix;
        # the single real image still rides image_b64 + the server chat
        # template). The model is prompt-coupled — serve exactly this string.
        prompt = self._prompt_with_image or PROMPT_BBOX_2D_JSON_WITH_IMAGE
        payload = {
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "prompt": prompt,
            "max_new_tokens": self._max_new_tokens,
            # Greedy decode for reproducible OCR (matches the eval adapter's
            # do_sample=False). A fine-tuned bbox model has a single right
            # answer; sampling only adds variance.
            "temperature": 0.0,
        }
        if sampling_overrides:
            # A1 (plan 028): the escalated retry tier. The serving shim defaults
            # every absent field to the frozen greedy values, so overrides only
            # ever widen the decode on the explicit retry call.
            payload.update(sampling_overrides)
        if settings.OCR_CHUNK_CONFIDENCE:
            # B (plan 028, dark flag): per-token logprobs for chunk confidence.
            payload["return_logprob"] = True
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else {}
        data = await self._post_with_retry(payload, headers)
        result = parse_qwen_lora_response(
            data,
            page_width=page_width,
            page_height=page_height,
        )
        # Keep the raw serving envelope for internal diagnostics only (the
        # Gemini-fallback review bundle). Never serialized into the customer
        # response or any log/metric — see ``OCRPageResult.raw``.
        result.raw = data
        # Flag a decode that ran to the token cap — the page is cut off mid-output,
        # so the pipeline treats it as unusable and falls back rather than ship a
        # partial extraction. The A1 shim returns the engine's real finish_reason
        # ("length" = hit the cap); pre-A1 envelopes carry none, so fall back to
        # inferring the cap from the token count, mirroring _extract_raw_text's
        # tolerance of the genuinely-polymorphic envelope shapes.
        finish_reason = _extract_finish_reason(data)
        if finish_reason is not None:
            result.truncated = finish_reason == "length"
        else:
            # Compare against the EFFECTIVE cap for this request, not the
            # instance default — a retry tier overrides max_new_tokens upward
            # (recovery escalates above the first-pass budget), so a retry that
            # finished naturally below its higher cap must not be mis-flagged as
            # truncated when the envelope carries no finish_reason.
            effective_cap = int(
                (sampling_overrides or {}).get("max_new_tokens", self._max_new_tokens)
            )
            n_out = _extract_output_tokens(data)
            if n_out is not None and n_out >= effective_cap:
                result.truncated = True
        return result

    async def _post_with_retry(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """POST to /predict, retrying only cold-start / transient errors.

        The extraction hot path (``_ocr_pages`` in ``core/pdf.py``) already
        handles empty/suspicious results and Textract fallback. The one thing it
        cannot do well is ride out a ``503`` while a scaled-to-zero replica spins
        up — that is what this loop is for. Deterministic 4xx are surfaced
        immediately so the page is dropped rather than retried into the same
        error.
        """
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            for attempt in range(_MAX_ATTEMPTS):
                try:
                    response = await client.post(self.endpoint_url, headers=headers, json=payload)
                    if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
                        await self._backoff(attempt, reason=f"HTTP {response.status_code}")
                        continue
                    response.raise_for_status()
                    return response.json()
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                    last_exc = exc
                    if attempt < _MAX_ATTEMPTS - 1:
                        await self._backoff(attempt, reason=type(exc).__name__)
                        continue
                    raise
        # Only reachable if the final iteration was a retryable status that we
        # then re-issued and it raised; re-raise the captured cause.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Qwen LoRA request failed without a response")

    async def _backoff(self, attempt: int, *, reason: str) -> None:
        delay = _BACKOFF_BASE_S * (2**attempt)
        logger.warning(
            "Qwen LoRA request retry %d/%d after %s; sleeping %.1fs",
            attempt + 1,
            _MAX_ATTEMPTS,
            reason,
            delay,
        )
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Response parsing: bbox_2d_json → OCRPageResult
# ---------------------------------------------------------------------------


def parse_qwen_lora_response(
    response: Any,
    *,
    page_width: float,
    page_height: float,
) -> OCRPageResult:
    """Normalize a Qwen LoRA ``/predict`` response into an ``OCRPageResult``.

    Accepts either the serving envelope (``{"raw": "[...]", ...}``) or a bare
    ``raw`` string / pre-parsed list, so the Truss can evolve without forcing
    a change here.
    """
    raw = _extract_raw_text(response)
    records = parse_bbox_2d_records(raw)
    # B2 (plan 028): per-record confidence from served token logprobs, aligned in
    # GENERATION order. ``None`` per record when the envelope carries no logprobs
    # or that record could not be aligned — never a fake 0.0.
    confidences = _record_confidences(response, raw, records)
    # 2026-07-30 reading-order fix: records stay in GENERATION order and carry
    # ``seq`` so page assembly can interleave blocks/tables/figures/KVs the way
    # the model emitted them. The old bbox-center (y, x) sort interleaved the
    # columns of every 2-column page and cost the extract-bench text-accuracy
    # headline −0.137 mean (receipts:
    # the internal measurement records — 141/400
    # served preds were exactly center-sorted; raw emission scored 0.643 vs
    # 0.316 served on the owner-flagged pages).

    blocks: list[OCRBlock] = []
    tables: list[OCRTable] = []
    figures: list[OCRFigure] = []
    key_values: list[OCRKeyValue] = []
    for i in range(len(records)):
        x0, y0, x1, y1, text, kind = records[i]
        conf = confidences[i]
        text = text.strip()
        bbox = _scale_bbox(x0, y0, x1, y1, page_width=page_width, page_height=page_height)

        # The explicit KV-region discriminator wins over ALL content inference
        # (image sentinel, markdown table): a record tagged ``type:"kv"`` is a KV
        # region regardless of its text — a region's ``Key: Value`` text must never
        # be misread. See .claude/skills/data-curation/references/kv-region-contract.md.
        if kind == "kv":
            if text:
                key_values.append(OCRKeyValue(text=text, bbox=bbox, confidence=conf, seq=i))
            continue
        if text == _IMAGE_SENTINEL:
            figures.append(OCRFigure(bbox=bbox, confidence=conf, seq=i))
            continue
        if _looks_like_markdown_table(text):
            table = _table_from_markdown(text=text, bbox=bbox, confidence=conf)
            table.seq = i
            tables.append(table)
            continue
        if text.startswith(_HANDWRITING_MARKER):
            # Strip the handwriting sentinel; the READ value stays a clean text
            # block (a hand-written name/date filled into a form is body text).
            text = text[len(_HANDWRITING_MARKER):].strip()
        if text:
            blocks.append(OCRBlock(text=text, bbox=bbox, confidence=conf, seq=i))

    return OCRPageResult(
        blocks=blocks, tables=tables, figures=figures, key_values=key_values
    )


def _record_confidences(
    response: Any,
    raw: str,
    records: list[tuple[int, int, int, int, str, str | None]],
) -> list[float | None]:
    """Uncalibrated per-record confidence (exp MIN value-token logprob), or
    ``None`` per record. Weakest-link, not mean: a single misread character in a
    chunk (e.g. one wrong digit in a member ID) must drag the score down rather
    than average out. Only the dict serving envelope can carry
    ``output_token_logprobs`` (B1 shim); every other tolerated shape → all None."""
    if not isinstance(response, dict) or not records:
        return [None] * len(records)
    token_logprobs = response.get("output_token_logprobs")
    if not token_logprobs:
        return [None] * len(records)
    stats = align_chunk_confidences(raw, token_logprobs, [r[4] for r in records])
    return [s.min_confidence if s is not None else None for s in stats]


def _extract_finish_reason(response: Any) -> str | None:
    """The engine's finish reason ("stop" / "length" / "abort") from the dict
    serving envelope, or ``None`` for pre-A1 envelopes / other tolerated shapes."""
    if isinstance(response, dict):
        fr = response.get("finish_reason")
        if isinstance(fr, str) and fr:
            return fr
    return None


def _extract_output_tokens(response: Any) -> int | None:
    """The decode's output-token count from the dict serving envelope, or ``None``
    when the response is one of the other shapes the parser tolerates (bare string
    / pre-parsed list) that carry no count. Mirrors :func:`_extract_raw_text`'s
    handling of the same genuinely-polymorphic envelope, so the truncation read
    never assumes a shape the parser itself does not."""
    if isinstance(response, dict):
        n = response.get("n_output_tokens")
        if isinstance(n, int) and not isinstance(n, bool):  # bool is an int subclass; not a count
            return n
    return None


def _extract_raw_text(response: Any) -> str:
    """Pull the raw model string out of the serving envelope (or pass through)."""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        # Already-parsed array — re-serialize so the single parse path handles it.
        # dict records pass through; parser-native (x0,y0,x1,y1,text) tuples are
        # reconstituted to records so the list path round-trips its own output.
        norm: list[dict[str, Any]] = []
        for item in response:
            if isinstance(item, dict):
                norm.append(item)
            elif isinstance(item, (list, tuple)) and len(item) in (5, 6):
                # 5-tuple (x0,y0,x1,y1,text) or 6-tuple (…,kind) — carry the
                # optional discriminator so parse_bbox_2d_records output round-trips.
                rec: dict[str, Any] = {"bbox_2d": list(item[:4]), "text_content": item[4]}
                if len(item) == 6 and item[5]:
                    rec["type"] = item[5]
                norm.append(rec)
        return json.dumps(norm)
    if isinstance(response, dict):
        for key in ("raw", "output", "text", "generated_text"):
            value = response.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return json.dumps(value)
    return ""


def _scale_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    page_width: float,
    page_height: float,
) -> list[float]:
    """0-1000 page-normalized (top-left origin) → page points, clipped."""
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


def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    m = _CODE_FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def _normalize_record(entry: Any) -> tuple[int, int, int, int, str, str | None] | None:
    """One bbox_2d record (object) → ``(x0, y0, x1, y1, text, kind)``, or ``None`` if
    invalid. Accepts the ``bbox_2d``/``bbox`` and ``text_content``/``text`` key
    aliases and any key order; rounds float coords; skips records with non-finite
    or non-numeric coords or non-string text. ``kind`` is the optional element
    discriminator (``"kv"`` for a key-value region; ``None`` for the inferred
    text/table/figure records — the champion grammar). This is the SINGLE
    normalization used by both the whole-array parse and the per-object fallback,
    so the fallback can never accept fewer fields than the valid-array path."""
    if not isinstance(entry, dict):
        return None
    bb = entry.get("bbox_2d")
    if bb is None:
        bb = entry.get("bbox")
    text = entry.get("text_content")
    if text is None:
        text = entry.get("text")
    raw_kind = entry.get("type")
    kind = raw_kind.strip().lower() if isinstance(raw_kind, str) and raw_kind.strip() else None
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    if not isinstance(text, str):
        return None
    # Validate each coord: it must be a real number (not bool — a Python int
    # subclass json would otherwise let through as 1.0/0.0 — and not a string/null),
    # finite (json accepts Infinity/NaN), and convertible (a huge int overflows
    # float()). Any failure skips the whole record.
    vals: list[float] = []
    for v in bb:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        try:
            f = float(v)
        except OverflowError:
            return None
        if not math.isfinite(f):
            return None
        vals.append(f)
    try:
        x0, y0, x1, y1 = (int(round(f)) for f in vals)
    except (ValueError, OverflowError):
        return None
    return (x0, y0, x1, y1, text.strip(), kind)


def _iter_json_objects(s: str):
    """Yield each complete top-level ``{...}`` object substring in ``s``, string-
    and escape-aware. An incomplete trailing object (truncated array) is skipped;
    braces and quotes inside string values do not affect nesting. Robust where a
    record-level regex is brittle (escapes, nested brackets, unicode, key order)."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield s[start : i + 1]
                start = -1


def _repair_json_string_escapes(chunk: str) -> str:
    """Repair the JSON-escaping violations generative models commonly emit inside
    string values, so a complete record is recovered instead of dropped: a stray
    backslash (LaTeX ``$\\alpha$``, a Windows path) or a literal control char (an
    embedded newline/tab). A string-aware char scanner (not a regex): inside a
    string, a backslash that does not begin a valid JSON escape is doubled, and
    raw control characters are escaped."""
    out: list[str] = []
    in_str = False
    i = 0
    n = len(chunk)
    while i < n:
        ch = chunk[i]
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            i += 1
        elif ch == '"':
            in_str = False
            out.append(ch)
            i += 1
        elif ch == "\\":
            nxt = chunk[i + 1] if i + 1 < n else ""
            if nxt == "u":
                # \uXXXX is valid ONLY with 4 hex digits; otherwise it's a stray
                # backslash (e.g. LaTeX \underline) and doubling it recovers the text.
                hexd = chunk[i + 2 : i + 6]
                if len(hexd) == 4 and all(c in "0123456789abcdefABCDEF" for c in hexd):
                    out.append(chunk[i : i + 6])
                    i += 6
                else:
                    out.append("\\\\")
                    i += 1
            elif nxt in '"\\/bfnrt':
                out.append(ch + nxt)  # already a valid escape — keep the pair
                i += 2
            else:
                out.append("\\\\")  # stray backslash → escape it
                i += 1
        elif ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            i += 1
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _loads_object(chunk: str) -> Any:
    """``json.loads`` one scanned object; if the strict parse fails, retry once
    after repairing common model escape violations. ``None`` if still unparseable."""
    try:
        return json.loads(chunk)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(_repair_json_string_escapes(chunk))
    except (json.JSONDecodeError, ValueError):
        return None


#: ``"text_content": "`` opener, used to locate the unterminated trailing object's
#: string value during single-table-blob salvage.
_TEXT_CONTENT_OPEN_RE = re.compile(r'"text_content"\s*:\s*"')
#: ``"bbox_2d": [ ... ]`` — capture the 4 ints of the unterminated object's box.
_BBOX_2D_INNER_RE = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]*)\]')


def _string_is_unterminated(body: str) -> bool:
    """True iff ``body`` (the chars after a JSON string's opening quote) never
    reaches an unescaped closing quote — i.e. the string was cut off mid-stream by
    the token cap. A terminated string means this is NOT a mid-string truncation."""
    esc = False
    for ch in body:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            return False
    return True


def _salvage_truncated_table_object(s: str) -> dict[str, Any] | None:
    """Recover the COMPLETE rows of a single pipe-delimited table chunk whose
    ``text_content`` string was truncated mid-stream at the token cap.

    When the model packs a whole dense page into ONE table record and the decode
    hits the cap inside that string, the trailing object never closes — so
    :func:`_iter_json_objects` (and the eval ``repair_truncated_json``) drop it
    WHOLE, losing every amount in it (the p0186 class: 26/26 amounts gone). The
    amounts, however, sit in COMPLETE pipe cells emitted before the cut. This
    salvages them: it rebuilds the object with its ``text_content`` truncated back
    to the last complete cell boundary (the last ``|`` or row-delimiting ``\\n``),
    so the standard JSON path decodes it as a normal table chunk.

    Keeping only content up to a cell boundary is what makes the salvage
    phantom-free: a value cut mid-digits has no trailing ``|``/``\\n`` so it is
    dropped, and every recovered token is verbatim from the raw. Returns ``None``
    unless the trailing object is genuinely an unterminated pipe-table string (so a
    truncated bbox, a closed string, or a truncated prose string is left alone)."""
    # The unterminated trailing object opens at the last '{' that comes AFTER the
    # last top-level object closed cleanly (string-aware brace walk).
    depth = 0
    in_str = False
    esc = False
    last_complete_end = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                last_complete_end = i
    open_idx = s.rfind("{")
    if open_idx <= last_complete_end:
        return None  # no unterminated trailing object to salvage
    frag = s[open_idx:]
    bbm = _BBOX_2D_INNER_RE.search(frag)
    tcm = _TEXT_CONTENT_OPEN_RE.search(frag)
    if not bbm or not tcm:
        return None  # truncated before its bbox/text_content — nothing to rebuild
    body = frag[tcm.end():]
    if not _string_is_unterminated(body):
        return None  # the string closed; the truncation is elsewhere, leave it
    # Require a genuine pipe table (>=2 cell pipes, or a row-delimiting '\n') so a
    # truncated prose string is never mangled into cells.
    if body.count("|") < 2 and "\\n" not in body:
        return None
    pipe = body.rfind("|")
    nl = body.rfind("\\n")  # the literal 2-char JSON row escape (string not decoded)
    cut = max(pipe + 1 if pipe != -1 else -1, nl + 2 if nl != -1 else -1)
    if cut <= 0:
        return None
    # Drop any dangling backslash at the cut so the rebuilt JSON string is valid.
    salvaged = body[:cut].rstrip("\\")
    rebuilt = '{"bbox_2d":[' + bbm.group(1) + '], "text_content":"' + salvaged + '"}'
    return _loads_object(rebuilt)


def parse_bbox_2d_json(raw: str) -> list[tuple[int, int, int, int, str]]:
    """Parse a ``bbox_2d_json`` model output into ``(x0, y0, x1, y1, text)`` tuples.

    The stable 5-tuple contract used by the tests and the ``evals/adapters/qwen_lora.py``
    parity check. It drops the optional element discriminator; the production
    parser (:func:`parse_qwen_lora_response`) uses :func:`parse_bbox_2d_records`
    when it needs the ``kind`` (KV regions). Mirrors ``evals/adapters/qwen_lora.py``.
    """
    return [(x0, y0, x1, y1, text) for (x0, y0, x1, y1, text, _kind) in parse_bbox_2d_records(raw)]


def parse_bbox_2d_records(raw: str) -> list[tuple[int, int, int, int, str, str | None]]:
    """Parse a ``bbox_2d_json`` model output into ``(x0, y0, x1, y1, text, kind)``
    tuples, where ``kind`` is the optional element discriminator (``"kv"`` for a
    key-value region; ``None`` for the inferred text/table/figure champion grammar).

    Robust to: surrounding markdown code fences, leading/trailing prose, stray
    special tokens (``<|im_end|>``), and TRUNCATED/malformed arrays. A valid array
    is decoded whole; otherwise each complete ``{...}`` object is decoded
    individually (an incomplete trailing record is dropped, every complete one is
    kept). Both paths share :func:`_normalize_record`, so the fallback handles the
    same aliases/key-order/float coords as the whole-array path and never silently
    drops a record the array path would keep.

    The one trailing-object drop we DO recover from: a single dense-table chunk
    whose ``text_content`` string was cut off mid-stream at the token cap. The
    object never closes, so the per-object scan drops it whole and every amount in
    it is lost (the p0186 class). :func:`_salvage_truncated_table_object` rebuilds
    it from the complete pipe cells emitted before the cut.

    Coordinates are returned as-is (0-1000 page-normalized); scaling to page points
    happens in :func:`_scale_bbox`.
    """
    s = _strip_code_fence(raw)
    out: list[tuple[int, int, int, int, str, str | None]] = []

    # Fast path: the whole string is a valid JSON array of records.
    try:
        parsed: Any = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        for entry in parsed:
            rec = _normalize_record(entry)
            if rec is not None:
                out.append(rec)
        return out

    # Fallback: decode each complete top-level object. Handles truncated arrays,
    # trailing prose/brackets, and stray tokens without a brittle record regex.
    for chunk in _iter_json_objects(s):
        obj = _loads_object(chunk)
        if obj is None:
            continue
        rec = _normalize_record(obj)
        if rec is not None:
            out.append(rec)

    # Salvage a single table chunk whose text_content was truncated mid-stream — the
    # one object _iter_json_objects drops whole. No-op unless the trailing object is
    # an unterminated pipe-table string, so it never touches a normal truncation.
    salvaged = _salvage_truncated_table_object(s)
    if salvaged is not None:
        rec = _normalize_record(salvaged)
        if rec is not None:
            out.append(rec)
    return out


def _reading_order(
    items: list[tuple[int, int, int, int, str]],
) -> list[tuple[int, int, int, int, str]]:
    """Top-to-bottom, left-to-right by bbox center."""
    return sorted(items, key=lambda b: ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2))


# ---------------------------------------------------------------------------
# GFM markdown tables → OCRTable
# ---------------------------------------------------------------------------


# How far down to scan for the GFM delimiter row. The model occasionally wraps a
# header cell across a literal newline (``| INDUSTRIAL\nLASER BAR CODE SCANNERS |
# … |``), which pushes the ``|---|`` delimiter below line 2; locating it within
# the first lines (rather than requiring it at exactly ``lines[1]``) recovers the
# whole table instead of demoting it to a prose block. Measured worth ~+8.6 pts of
# table_similarity on RD-TableBench champion-only (decision-logs/2026-06-17).
_TABLE_DELIM_SCAN_LINES = 8


def _locate_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Return ``(header_cells, body_rows)`` if ``text`` is a GFM table, else ``None``.

    Delimiter-position-agnostic: locate the first delimiter row within the first
    lines, treat everything before it as a (possibly newline-wrapped) header joined
    into one logical row, and merge body lines carrying no pipe into the previous
    row (a wrapped-cell continuation). The header/delimiter width-match guard
    (>=2 cols, equal width) keeps prose and captions out. This is the SINGLE source
    of truth for both detection (:func:`_looks_like_markdown_table`) and
    construction (:func:`_table_from_markdown`) so the two cannot drift.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    d = next(
        (i for i in range(min(_TABLE_DELIM_SCAN_LINES, len(lines))) if _is_delimiter_row(lines[i])),
        None,
    )
    if not d:  # None (no delimiter found) or 0 (delimiter with no header above it)
        return None
    header = _split_markdown_row(" ".join(lines[:d]))
    delim = _split_markdown_row(lines[d])
    if len(header) < 2 or len(delim) != len(header):
        return None
    body: list[str] = []
    for ln in lines[d + 1:]:
        if "|" in ln or not body:
            body.append(ln)
        else:
            body[-1] = body[-1] + " " + ln
    # A header + delimiter followed by PURE prose (no body line has a pipe) is not a
    # table — keep the long-standing rejection.
    if body and not any("|" in ln for ln in body):
        return None
    return header, [_split_markdown_row(ln) for ln in body]


def _looks_like_markdown_table(text: str) -> bool:
    """True iff ``text`` is a GFM markdown table (delimiter may be newline-wrapped).

    Requires a pipe-delimited header AND a delimiter row (``|---|---|``) of matching
    width — the reliable, prose-proof table signal. The delimiter need not sit
    physically on line 2: a header cell the model wrapped across a newline pushes it
    down, and :func:`_locate_table` finds it. Body rows may be RAGGED; only a
    header+delimiter followed by pure prose is rejected.
    """
    return _locate_table(text) is not None


def _is_delimiter_row(line: str) -> bool:
    cells = _split_markdown_row(line)
    if not cells:
        return False
    # EVERY cell must be a GFM delimiter token (``:?-+:?``); an empty internal cell
    # (e.g. ``|  | --- |``) means this is not a real delimiter row, so the text must
    # stay prose rather than be mis-parsed into a table that drops the line.
    return all(re.fullmatch(r":?-{1,}:?", cell.strip()) for cell in cells)


def _split_markdown_row(line: str) -> list[str]:
    """Split a GFM table row into cells, dropping the outer borders.

    A char scanner (not a regex) because GFM treats a ``|`` as a column separator
    only when it is neither backslash-escaped (``\\|``) nor inside a backtick code
    span (`` `a|b` ``) — both are literal cell content.
    """
    s = line.strip()
    parts: list[str] = []
    buf: list[str] = []
    fence = 0  # length of the backtick run that opened the current code span (0 = outside)
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            buf.append(s[i : i + 2])  # keep the escape pair intact (e.g. "\|")
            i += 2
            continue
        if ch == "`":
            # GFM code spans are delimited by RUNS of backticks; a span opened by
            # N backticks closes only on another run of exactly N.
            j = i
            while j < n and s[j] == "`":
                j += 1
            run = j - i
            buf.append(s[i:j])
            if fence == 0:
                fence = run
            elif run == fence:
                fence = 0
            i = j
            continue
        if ch == "|" and fence == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    # A bordered row "| a | b |" yields a leading/trailing empty fragment — drop them.
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.replace("\\|", "|").strip() for p in parts]


def _table_from_markdown(*, text: str, bbox: list[float], confidence: float | None) -> OCRTable:
    """Parse a GFM markdown table into an ``OCRTable`` via :func:`_locate_table`.

    The model gives one bbox per table (not per cell), so every cell shares the
    table bbox — the same simplification the eval adapter makes for HTML tables.
    GFM cannot express row/col spans, so spans are always 1. Delimiter location and
    header/body splitting are delegated to :func:`_locate_table` (shared with
    detection, so they can't drift). If the text does not locate as a table, fall
    back to a single cell carrying the raw markdown so text is never lost.
    """
    located = _locate_table(text)
    if located is None:
        return OCRTable(
            cells=[OCRTableCell(text=text, row=0, col=0, bbox=bbox, confidence=confidence)],
            bbox=bbox,
            n_rows=1,
            n_cols=1,
            confidence=confidence,
        )
    header, body_rows = located
    cells: list[OCRTableCell] = []
    out_row = 0
    n_cols = 0
    for row_cells in (header, *body_rows):
        if not row_cells:
            continue
        n_cols = max(n_cols, len(row_cells))
        for col, cell_text in enumerate(row_cells):
            cells.append(
                OCRTableCell(text=cell_text, row=out_row, col=col, bbox=bbox, confidence=confidence)
            )
        out_row += 1

    if not cells:
        return OCRTable(
            cells=[OCRTableCell(text=text, row=0, col=0, bbox=bbox, confidence=confidence)],
            bbox=bbox,
            n_rows=1,
            n_cols=1,
            confidence=confidence,
        )
    return OCRTable(cells=cells, bbox=bbox, n_rows=out_row, n_cols=n_cols, confidence=confidence)
