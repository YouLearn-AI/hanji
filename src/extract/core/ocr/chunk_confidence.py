"""Token→chunk logprob alignment — per-chunk confidence from served logprobs.

Plan 028 workstream B (B2): the served SGLang shim can return
``output_token_logprobs`` (``[[logprob, token_id, token_text], ...]``,
generation order, token_text decoded per-id server-side). This module maps
those token logprobs onto the parsed ``bbox_2d_json`` records' ``text_content``
value spans and reduces them to MinerU-style ``ScoredOutput`` statistics
(mean / min / std logprob) plus an uncalibrated ``confidence = exp(mean)``.

Alignment is fail-closed at every step: any token-offset or value-span
mismatch yields ``None`` for the affected chunk (no confidence), never a wrong
number. Coverage (chunks-with-confidence / chunks) is therefore a quality
signal in its own right and is reported by the B5 gate.

Two consumers, one home (no copies):
  - the production provider (``extract.core.ocr.qwen_lora``) behind the dark
    ``OCR_CHUNK_CONFIDENCE`` flag, and
  - the eval harness's served-engine candidate, which
    lazy-imports this module only when a run requests ``return_logprob``.

Pure stdlib on purpose (no tokenizer, no torch): per-token *text* comes from
the serving side, so alignment here is plain string matching.

Known limits (documented, measured by the coverage metric rather than hidden):
  - Per-id decoding splits multi-byte UTF-8 across tokens into U+FFFD
    fragments (non-Latin scripts). Fragments are grouped and anchored to the
    next cleanly-matching token; logprobs inside the group still count toward
    any value span overlapping the group's char range.
  - Only records parsed from real JSON (strict / truncation-repaired tiers)
    should be passed in; regex-salvaged chunks carry no trustworthy value span
    (the caller passes ``[]`` or skips this call for those tiers).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfidence:
    """Logprob statistics for one parsed chunk's text_content value tokens."""

    mean_logprob: float
    min_logprob: float
    std_logprob: float
    n_tokens: int

    @property
    def confidence(self) -> float:
        """Uncalibrated parse-channel confidence in [0, 1]: exp(mean logprob)."""
        return math.exp(self.mean_logprob)

    @property
    def min_confidence(self) -> float:
        """Weakest-link confidence in [0, 1]: exp(min logprob).

        The mean masks a single misread character (one bad digit averaged
        against a run of confident ones reads high); the min surfaces it. This
        is the signal to gate short high-stakes values like member IDs on — a
        lone low-confidence glyph (e.g. a handwritten ``3`` read as ``2``) drags
        this down even when the mean stays near 1.0.
        """
        return math.exp(self.min_logprob)


def _token_offsets(
    raw: str, token_texts: Sequence[str | None]
) -> list[tuple[int, int]] | None:
    """Char span ``[start, end)`` in ``raw`` for each token, or ``None`` when the
    token stream cannot be aligned to ``raw`` at all.

    Tokens whose text concatenates cleanly advance an exact cursor. Tokens that
    do not (U+FFFD multi-byte fragments, missing text, detokenization drift,
    trailing special tokens absent from ``raw``) are pooled into a pending group
    that is closed by the next token anchoring exactly — every pending token is
    assigned the whole gap span, which keeps their logprobs available to any
    value span overlapping the gap without guessing per-token boundaries.
    Anchoring requires a >=2-char token to cut false anchors from single-char
    matches landing early inside the gap.
    """
    offsets: list[tuple[int, int]] = [(0, 0)] * len(token_texts)
    pos = 0
    pending: list[int] = []
    for k, text in enumerate(token_texts):
        clean = bool(text) and "�" not in text
        if clean and not pending and raw.startswith(text, pos):
            offsets[k] = (pos, pos + len(text))
            pos += len(text)
            continue
        if clean and pending and len(text) >= 2:
            found = raw.find(text, pos)
            if found != -1:
                for j in pending:
                    offsets[j] = (pos, found)
                pending = []
                offsets[k] = (found, found + len(text))
                pos = found + len(text)
                continue
        pending.append(k)
    for j in pending:
        offsets[j] = (pos, len(raw))
    # Fail-closed sanity: if nothing anchored (all tokens pending → all spans
    # collapse to the tail), the stream does not describe this raw string.
    if token_texts and all(start == pos for start, _ in offsets):
        return None
    return offsets


def _value_span(raw: str, text: str, cursor: int) -> tuple[int, int] | None:
    """Locate ``text`` (a parsed text_content value) in ``raw`` at/after ``cursor``.

    The value appears in ``raw`` JSON-escaped; try the model's usual form
    (unicode kept literal) first, then the fully-escaped form, then the raw
    text itself (covers values that needed no escaping).
    """
    seen: set[str] = set()
    for cand in (
        json.dumps(text, ensure_ascii=False)[1:-1],
        json.dumps(text)[1:-1],
        text,
    ):
        if not cand or cand in seen:
            continue
        seen.add(cand)
        i = raw.find(cand, cursor)
        if i != -1:
            return (i, i + len(cand))
    return None


def align_chunk_confidences(
    raw: str,
    output_token_logprobs: Sequence[Sequence],
    chunk_texts: Sequence[str],
) -> list[ChunkConfidence | None]:
    """Per-chunk logprob statistics, aligned with ``chunk_texts``.

    ``output_token_logprobs`` is the served envelope field
    (``[[logprob, token_id, token_text], ...]``); ``chunk_texts`` are the parsed
    records' ``text_content`` values in **generation order** (pre any
    reading-order sort). Returns one entry per chunk text — a
    :class:`ChunkConfidence` or ``None`` when that chunk could not be aligned.
    JSON syntax tokens are excluded naturally: only tokens overlapping the
    located value span contribute.
    """
    n = len(chunk_texts)
    if not raw or not output_token_logprobs or n == 0:
        return [None] * n

    logprobs: list[float] = []
    token_texts: list[str | None] = []
    for entry in output_token_logprobs:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            return [None] * n
        lp, _tid, text = entry[0], entry[1], entry[2]
        if not isinstance(lp, (int, float)):
            return [None] * n
        logprobs.append(float(lp))
        token_texts.append(text if isinstance(text, str) else None)

    offsets = _token_offsets(raw, token_texts)
    if offsets is None:
        return [None] * n

    out: list[ChunkConfidence | None] = []
    cursor = 0
    for text in chunk_texts:
        span = _value_span(raw, text, cursor)
        if span is None:
            out.append(None)
            continue
        s, e = span
        cursor = e
        vals = [
            lp
            for lp, (ts, te) in zip(logprobs, offsets, strict=False)
            if ts < e and te > s  # token overlaps the value span
        ]
        if not vals:
            out.append(None)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out.append(
            ChunkConfidence(
                mean_logprob=mean,
                min_logprob=min(vals),
                std_logprob=math.sqrt(var),
                n_tokens=len(vals),
            )
        )
    return out
