"""Gemini structured-output client for schema extraction (plan 050).

The :class:`~extract.core.schema_extract.SchemaModel` implementation: a text-in /
JSON-out call against ``response_schema``, distinct from the OCR provider in
``core/ocr/gemini.py`` (image-in / bbox-out). Same Vertex transport, for the
same reason: Vertex is the Google surface covered by the Cloud BAA, so this
endpoint can serve PHI where the Claude playground prototype (plan 026) 403s.

Transport is Vertex AI (``GOOGLE_VERTEX_PROJECT``) or the plain Gemini API key
(``GEMINI_API_KEY``) — see ``extract.genai_client``. The client is built once and
reused: the OAuth credentials' token is cached until expiry, so rebuilding per
call would add a token round-trip.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any

from extract.config import settings
from extract.core.errors import UpstreamTimeout

log = logging.getLogger(__name__)

#: Transient-error retry. Field-group splitting (wide schemas) fires many
#: concurrent Gemini calls per request, so a transient Vertex throttle/timeout on
#: one batch must not fail the whole extraction. We retry transient errors only —
#: never a 400/INVALID_ARGUMENT (a real schema error).
#: Diagnostic sink for per-call candidate dicts (paired replay experiments).
#: None in production; a harness sets it to a list.
CAND_SINK: list | None = None

_MAX_ATTEMPTS = 4
_BACKOFF_BASE_S = 1.0
_RETRYABLE_MARKERS = (
    "429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
    "DEADLINE", "Empty Gemini",
)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc)
    if "400" in msg or "INVALID_ARGUMENT" in msg:
        return False
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _leaf_value(node: Any) -> Any:
    """The scalar of a wrapped ``{value, chunks, quote}`` leaf, else None."""
    if isinstance(node, dict) and "value" in node and not isinstance(node.get("value"), (dict, list)):
        return node.get("value")
    return None


def _identity_keys(rec: Any) -> set[str]:
    """Normalized non-null scalar leaves of a record (one level deep) — the record's
    identity signature for cross-candidate alignment. Booleans and short strings are
    excluded (an ``is_x=false`` or a 2-char state code is not identity)."""
    out: set[str] = set()
    if isinstance(rec, dict):
        for v in rec.values():
            lv = _leaf_value(v)
            if lv in (None, "") or isinstance(lv, bool):
                continue
            s = re.sub(r"[^A-Za-z0-9]+", "", str(lv)).upper()
            if len(s) >= 3:
                out.add(s)
    return out


def _union_records(a: list, b: list) -> list:
    """Identity-aware union of two candidates' record arrays.

    Candidates frequently order array records differently (or one omits a record
    entirely). Index-aligned merging then stitches FIELDS OF DIFFERENT RECORDS
    together — e.g. candidate A's ``{name: "KP MEDICARE"}`` filled with candidate
    B's Medicare row ``member_id`` (a cross-candidate mispair a reviewer reads as
    a wrong money field). Align records by shared identity leaves instead:
    greedy 1:1 on the largest overlap of normalized non-null scalars; only the
    matched pair merges. Unmatched ``b`` records append (recovering omissions —
    the reason candcount exists); ``b`` records with no identity fall back to
    their own index if that ``a`` slot is unclaimed. ``a`` is always the floor.
    """
    a_keys = [_identity_keys(r) for r in a]
    b_keys = [_identity_keys(r) for r in b]
    scored = []
    for j, bk in enumerate(b_keys):
        for i, ak in enumerate(a_keys):
            ov = len(ak & bk)
            if ov:
                scored.append((ov, -abs(i - j), i, j))
    scored.sort(reverse=True)
    a_used: set[int] = set()
    b_used: set[int] = set()
    pair_for_a: dict[int, int] = {}
    for _ov, _, i, j in scored:
        if i in a_used or j in b_used:
            continue
        a_used.add(i)
        b_used.add(j)
        pair_for_a[i] = j
    # no-identity b records: index fallback into an unclaimed identity-less a slot
    for j, bk in enumerate(b_keys):
        if j in b_used or bk:
            continue
        if j < len(a) and j not in a_used and not a_keys[j]:
            a_used.add(j)
            b_used.add(j)
            pair_for_a[j] = j
    out = [
        _union_nonnull(rec, b[pair_for_a[i]]) if i in pair_for_a else rec
        for i, rec in enumerate(a)
    ]
    out.extend(b[j] for j in range(len(b)) if j not in b_used)
    return out


def _union_nonnull(a: Any, b: Any) -> Any:
    """Deep-merge two candcount candidates: keep ``a`` everywhere, but fill a leaf from
    ``b`` when ``a``'s value is null/missing. Leaves are ``{value, chunks, quote}`` wrappers;
    a leaf is "filled from b" only when a.value is null and b.value is not. Recurses through
    objects. Arrays of RECORD dicts align by record identity (see :func:`_union_records`);
    other arrays align by index. Pure structure — grounding happens later."""
    if isinstance(a, dict) and isinstance(b, dict):
        if "value" in a and "value" in b and not isinstance(a.get("value"), (dict, list)):
            return a if a.get("value") not in (None, "") else b
        out = dict(a)
        for k, v in b.items():
            out[k] = _union_nonnull(a[k], v) if k in a else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        if any(isinstance(x, dict) and "value" not in x for x in a + b):
            return _union_records(a, b)
        return [_union_nonnull(a[i], b[i]) if i < len(a) else b[i] for i in range(len(b))] or a
    return a


class GeminiSchemaExtractor:
    """Vertex structured-output model used by the schema-extraction endpoint."""

    name = "gemini_extract"

    def __init__(
        self,
        *,
        model: str | None = None,
        thinking_budget: int | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        temperature: float = 0.0,
        candidate_count: int | None = None,
    ) -> None:
        self._model = model or settings.GEMINI_EXTRACT_MODEL or settings.GEMINI_OCR_MODEL
        self._temperature = temperature
        self._thinking_budget = (
            settings.GEMINI_EXTRACT_THINKING_BUDGET if thinking_budget is None else thinking_budget
        )
        # candcount: >1 → request N candidates + grounding-gated union (see config.py).
        self._candidate_count = max(
            1, settings.GEMINI_EXTRACT_CANDIDATE_COUNT if candidate_count is None else candidate_count
        )
        # The multi-candidate request shape is a Vertex feature; the plain
        # Gemini developer API rejects candidateCount > 1 for these models, so
        # the API-key transport degrades to a single candidate.
        if candidate_count is None and not settings.GOOGLE_VERTEX_PROJECT:
            self._candidate_count = 1
        self._max_output_tokens = max_output_tokens or settings.GEMINI_EXTRACT_MAX_OUTPUT_TOKENS
        self._timeout_s = timeout_s or settings.GEMINI_EXTRACT_TIMEOUT_SECONDS
        self._total_deadline_s = settings.GEMINI_EXTRACT_TOTAL_DEADLINE_SECONDS
        self._max_timeout_attempts = max(1, settings.GEMINI_EXTRACT_MAX_TIMEOUT_ATTEMPTS)
        self._client: Any = None

    def _make_client(self, genai: Any) -> Any:
        """Build the GenAI client — Vertex AI (GOOGLE_VERTEX_PROJECT) or the
        plain Gemini API key transport (GEMINI_API_KEY). See genai_client.py."""
        from extract.genai_client import make_genai_client

        return make_genai_client(genai)

    async def generate_json(
        self, prompt: str, response_schema: dict[str, Any],
        images: list[bytes] | None = None,
        images_first: bool = False,
        service_tier: str | None = None,
        model: str | None = None,
        candidate_count: int | None = None,
    ) -> dict[str, Any]:
        """Structured JSON from Vertex, bounded by a total deadline.

        The budget covers every attempt AND its backoff, and each attempt is
        additionally clamped so it cannot overrun what remains — so the wall
        time of this call is the deadline, whatever the provider does. Retries
        that could not start before the deadline are not started at all.

        ``service_tier`` overrides ``EXTRACT_SERVICE_TIER`` for THIS call only
        (``None`` = use the deployment setting). It exists so a single lane can
        ride flex without the deployment putting every other customer's calls on
        a shed-able tier; the per-call flex→Standard fallback below is unchanged.

        ``model`` and ``candidate_count`` are the same idea for the two levers a
        measured stack is actually pinned to: which Vertex model id answers, and
        how many candidates it is asked for. ``None`` (every caller but the lane)
        keeps the instance's own values, which come from ``GEMINI_EXTRACT_MODEL``
        / ``GEMINI_EXTRACT_CANDIDATE_COUNT``. Per-request rather than per-
        deployment because those settings are global: the deployment runs
        candcount=2 for everyone, and a lane whose sealed stack is cc=1 on
        gemini-3.6-flash must not have to move everybody else to get it.
        """
        started = time.monotonic()
        deadline_s = self._total_deadline_s
        state: dict[str, Any] = {"attempts": 0, "timeouts": 0, "last_exc": None}
        try:
            # Structural bound, not arithmetic: the budget holds even if an
            # attempt overruns its own timeout. Checking the clock only BETWEEN
            # attempts would leave the exact hole this fix exists to close.
            async with asyncio.timeout(deadline_s):
                return await self._attempt_loop(
                    prompt, response_schema, images, started, state,
                    images_first=images_first,
                    service_tier=service_tier,
                    model=model,
                    candidate_count=candidate_count,
                )
        except TimeoutError as exc:  # asyncio.TimeoutError is an alias since 3.11
            raise UpstreamTimeout(
                f"Vertex did not respond within the {deadline_s:.0f}s budget "
                f"({state['attempts']} attempt(s), {state['timeouts']} timed out).",
                elapsed_s=time.monotonic() - started,
                attempts=int(state["attempts"]),
            ) from exc

    async def _attempt_loop(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        images: list[bytes] | None,
        started: float,
        state: dict[str, Any],
        images_first: bool = False,
        service_tier: str | None = None,
        model: str | None = None,
        candidate_count: int | None = None,
    ) -> dict[str, Any]:
        """Retry body. Runs under the caller's deadline; ``state`` is carried out
        so the timeout handler can report attempt counts."""
        deadline_s = self._total_deadline_s
        # Per-request pins, forwarded only when set: an unset override must leave
        # ``_call_once`` called with exactly the signature it had (the same
        # splat-guard the schema pass uses for its stub models).
        pins: dict[str, Any] = {}
        if model:
            pins["model"] = model
        if candidate_count is not None:
            pins["candidate_count"] = candidate_count
        last_exc: Exception | None = None
        while state["attempts"] < _MAX_ATTEMPTS:
            remaining = deadline_s - (time.monotonic() - started)
            if remaining <= 0:
                break
            # Counted BEFORE the call, not in the except block: when the outer
            # deadline fires it cancels this coroutine mid-attempt, so an
            # increment after the await never runs and the error under-reports
            # by one ("1 attempt" for a call that made two). The count exists to
            # explain the failure in logs, so it has to survive cancellation.
            state["attempts"] += 1
            # Flex tier (50% off, same synchronous call) until it stalls once —
            # then this REQUEST falls back to Standard per call, bounding the
            # latency a shed flex attempt can cost (strikethrough-lane pattern).
            # The caller's per-request tier wins over the deployment setting; the
            # fallback is identical either way.
            tier = service_tier or settings.EXTRACT_SERVICE_TIER
            use_flex = tier == "flex" and not state.get("flex_failed")
            attempt_timeout = min(
                self._timeout_s if not use_flex else settings.EXTRACT_FLEX_TIMEOUT_SECONDS,
                remaining,
            )
            try:
                # Clamp the attempt to what's left: a 90 s attempt must not be
                # started against a deadline with 20 s on it.
                return await self._call_once(
                    prompt, response_schema, images,
                    timeout_s=attempt_timeout,
                    images_first=images_first,
                    flex=use_flex,
                    **pins,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                state["last_exc"] = exc
                if isinstance(exc, TimeoutError):
                    state["timeouts"] += 1
                    if use_flex:
                        # A stalled flex attempt doesn't count against the
                        # repeated-timeout cap — the Standard retry is the point.
                        state["timeouts"] -= 1
                    # Repeated stalls burn budget without new information.
                    if state["timeouts"] >= self._max_timeout_attempts:
                        break
                # ANY retryable flex failure demotes this request, not only a
                # stall: a shed/preempted flex attempt comes back as a retryable
                # 503, and re-issuing it on the same shed tier is how a request
                # burns its whole attempt budget without ever reaching Standard.
                if use_flex and (isinstance(exc, TimeoutError) or _is_retryable(exc)):
                    state["flex_failed"] = True
                if not _is_retryable(exc):
                    raise
                if state["attempts"] >= _MAX_ATTEMPTS:
                    break
                backoff = _BACKOFF_BASE_S * (2 ** (state["attempts"] - 1)) + random.random()
                if (time.monotonic() - started) + backoff >= deadline_s:
                    break  # no room to sleep AND retry — stop, don't idle out the clock
                await asyncio.sleep(backoff)

        elapsed = time.monotonic() - started
        if state["timeouts"]:
            raise UpstreamTimeout(
                f"Vertex did not respond within the {deadline_s:.0f}s budget "
                f"({state['attempts']} attempt(s), {state['timeouts']} timed out).",
                elapsed_s=elapsed,
                attempts=int(state["attempts"]),
            ) from last_exc
        if last_exc is not None:
            raise last_exc
        raise UpstreamTimeout(  # budget spent before any attempt could start
            f"Extraction budget ({deadline_s:.0f}s) was exhausted before any attempt ran.",
            elapsed_s=elapsed,
            attempts=0,
        )

    async def _call_once(
        self, prompt: str, response_schema: dict[str, Any],
        images: list[bytes] | None = None,
        timeout_s: float | None = None,
        images_first: bool = False,
        flex: bool = False,
        model: str | None = None,
        candidate_count: int | None = None,
    ) -> dict[str, Any]:
        from google import genai
        from google.genai import types as gt

        if self._client is None:
            self._client = self._make_client(genai)
        client = self._client

        # Per-request pins beat the instance's settings-derived values; ``None``
        # (every caller but the receipt lane) keeps them exactly as configured.
        model_id = model or self._model
        cc = self._candidate_count if candidate_count is None else max(1, candidate_count)
        # candcount needs temp>0 so the candidates diverge; floor it when enabled.
        temperature = max(self._temperature, settings.GEMINI_EXTRACT_CANDCOUNT_TEMP) if cc > 1 else self._temperature
        cfg_kwargs: dict[str, Any] = dict(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            max_output_tokens=self._max_output_tokens,
            candidate_count=cc,
        )
        if self._thinking_budget is not None:
            cfg_kwargs["thinking_config"] = gt.ThinkingConfig(thinking_budget=self._thinking_budget)
        if settings.EXTRACT_MEDIA_RESOLUTION:
            # A typo in this deployment variable must not KeyError every extraction
            # call: normalize, and fall back to the SDK default with one warning.
            _mr = settings.EXTRACT_MEDIA_RESOLUTION.strip().lower()
            _res = {
                "low": gt.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": gt.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": gt.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }.get(_mr)
            if _res is None:
                log.warning(
                    "EXTRACT_MEDIA_RESOLUTION=%r is not one of low/medium/high — "
                    "using the SDK default", settings.EXTRACT_MEDIA_RESOLUTION
                )
            else:
                cfg_kwargs["media_resolution"] = _res
        if flex:
            # The ONLY working mechanism (strikethrough lane, measured): the
            # lowercase header. The SDK's service_tier config field is accepted
            # and silently billed at Standard; never use it.
            cfg_kwargs["http_options"] = gt.HttpOptions(
                headers={"X-Vertex-AI-LLM-Shared-Request-Type": "flex"}
            )

        if images_first:
            # Cache-friendly layout: images lead so the (identical-across-group-
            # calls) image+context prefix hits Vertex implicit caching.
            parts = [gt.Part.from_bytes(data=img, mime_type="image/png") for img in (images or [])]
            parts.append(gt.Part.from_text(text=prompt))
        else:
            parts = [gt.Part.from_text(text=prompt)]
            for img in (images or []):
                parts.append(gt.Part.from_bytes(data=img, mime_type="image/png"))
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_id,
                contents=parts,
                config=gt.GenerateContentConfig(**cfg_kwargs),
            ),
            timeout=self._timeout_s if timeout_s is None else timeout_s,
        )

        um = getattr(response, "usage_metadata", None)
        if um is not None:
            # Raw per-call token accounting (rate-card of record; see
            # the internal measurement records: a modeled cost was 2.4x off measured). DEBUG: it
            # fires several times per request, which is what a bench wants and
            # not what a production log is for — raise the level to bench it.
            log.debug(
                "gemini_extract usage: prompt=%s cached=%s output=%s thoughts=%s tier=%s",
                getattr(um, "prompt_token_count", None),
                getattr(um, "cached_content_token_count", None),
                getattr(um, "candidates_token_count", None),
                getattr(um, "thoughts_token_count", None),
                getattr(um, "traffic_type", None),
            )
        if cc == 1:
            # default path — byte-identical to before candcount existed.
            raw = response.text or ""
            if not raw:
                finish = response.candidates and response.candidates[0].finish_reason
                raise SchemaModelError(f"Empty Gemini response (finish: {finish})")
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                raise SchemaModelError("Gemini returned non-JSON output.") from e
            if not isinstance(parsed, dict):
                raise SchemaModelError("Gemini returned a non-object top-level result.")
            return parsed

        # candcount (cc > 1): grounding-gated union over candidates. Each candidate's leaf
        # carries {value, chunks, quote}; we keep the first non-null value per leaf and the
        # downstream quote-check (aextract_schema, strict mode) still grounds it — so the
        # union can recover an omission but cannot smuggle an ungrounded value through.
        dicts: list[dict[str, Any]] = []
        for cand in (response.candidates or []):
            try:
                d = json.loads(cand.content.parts[0].text)
            except (AttributeError, IndexError, TypeError, json.JSONDecodeError, ValueError):
                continue
            if isinstance(d, dict):
                dicts.append(d)
        if not dicts:
            finish = response.candidates and response.candidates[0].finish_reason
            raise SchemaModelError(f"Empty/invalid Gemini candcount response (finish: {finish})")
        if CAND_SINK is not None:
            CAND_SINK.append(dicts)  # diagnostic: per-call candidate dicts for paired replay
        merged = dicts[0]
        for d in dicts[1:]:
            merged = _union_nonnull(merged, d)
        return merged


class SchemaModelError(RuntimeError):
    """Model timeout / empty / non-JSON response → mapped to HTTP 502."""


def from_settings() -> GeminiSchemaExtractor:
    """Construct the extractor. Lazy: no Vertex round-trip until first call, so
    the app boots without Vertex creds (and tests inject a stub)."""
    return GeminiSchemaExtractor()
