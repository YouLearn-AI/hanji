"""SGLang /predict shim for the merged Qwen3-VL parse model.

Serves https://huggingface.co/hanji-dev/hanji-parse-4b (or any merged
Qwen3-VL checkpoint) behind the /predict contract expected by
``extract.core.ocr.qwen_lora.QwenLoraProvider``:

  request : {"image_b64": "<png>", "prompt": "<bbox_2d_json prompt>", "max_new_tokens": int}
  response: {"raw": "<model output string>", "wall_ms", "image_size", "n_input_tokens",
             "n_output_tokens", "finish_reason",
             "generation": {"requested": {...}, "effective": {...}}}

The client's ``parse_qwen_lora_response`` reads ``response["raw"]`` and parses
bbox_2d_json from it.

Optional request fields (every default reproduces the frozen greedy decode, so
a request without them is byte-compatible):
  "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"
      — sampling overrides for the pipeline's escalated retry tier. ``seed``
      maps to SGLang's ``sampling_seed`` and is echoed in ``generation``.
  "return_logprob": true
      — response gains "output_token_logprobs": [[logprob, token_id, token_text], ...]
      (feeds the pipeline's per-chunk confidence).
  "json_schema": "<json-schema string>" | {...}
      — xgrammar-constrained decoding for the chunk contract.
  "no_repeat_ngram": int (e.g. 100)
      — large-n no-repeat custom logit processor (bans only exact large-n-gram
      continuations, never the short repeated bbox_2d/text_content keys).
      Requires enable_custom_logit_processor=true in SGLANG_ENGINE_EXTRA.

Greedy decoding by default; the image is resized to the training pixel cap
(each dimension a multiple of 32) so the model's 0-1000 normalized bboxes stay
calibrated to what it saw in training. See SELF_HOSTING.md for how to run this.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from sglang import Engine
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from transformers import AutoProcessor

# A Hugging Face repo id (downloaded on first boot) or a local directory of
# merged weights. QWEN_WEIGHTS_PATH is honored as a legacy alias.
WEIGHTS = (
    os.environ.get("MODEL_PATH")
    or os.environ.get("QWEN_WEIGHTS_PATH")
    or "hanji-dev/hanji-parse-4b"
)
MAX_PIXELS = int(os.environ.get("QWEN_MAX_IMAGE_PIXELS", 2_000_000))
PATCH = 32
CONTEXT_LENGTH = 16384

engine: Engine | None = None
processor: Any | None = None


def _env_bool(name: str, default: str) -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = (MAX_PIXELS / max(1, w * h)) ** 0.5
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    w = max(PATCH, (w // PATCH) * PATCH)
    h = max(PATCH, (h // PATCH) * PATCH)
    return img.resize((w, h), Image.LANCZOS) if (w, h) != img.size else img


class LargeNoRepeatNgramProcessor(CustomLogitProcessor):
    """Large-n no-repeat-ngram ban over GENERATED ids (plan 028 A2; MinerU's
    ``no_repeat_ngram_size=100`` recipe ported to SGLang's CustomLogitProcessor).

    Bans any token that would complete an exact n-gram already present in the
    request's output ids. With n≈100 only a degenerate loop can ever re-produce
    the (n-1)-token context, so legitimate short repeats (the ``bbox_2d`` /
    ``text_content`` JSON keys that small-n bans corrupted) are never touched.

    Serialized per-request via ``to_str()`` (dill, by value) and dispatched with
    ``custom_params={"no_repeat_ngram_size": n}``; SGLang injects ``__req__``
    so ``req.output_ids`` (generated ids only — the prompt is never scanned) is
    available. The id scan is byte-packed so the hot loop is C-speed ``bytes.find``.
    """

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list:
            return logits
        import struct

        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            n = int(params.get("no_repeat_ngram_size") or 0)
            req = params.get("__req__")
            if n <= 1 or req is None:
                continue
            out = getattr(req, "output_ids", None)
            if not out or len(out) < n:
                continue
            ctx = out[-(n - 1) :]
            buf = struct.pack(f">{len(out)}I", *out)
            pat = struct.pack(f">{len(ctx)}I", *ctx)
            tail_byte = (len(out) - (n - 1)) * 4  # the context's own position
            banned = []
            start = 0
            while True:
                j = buf.find(pat, start)
                if j < 0 or j >= tail_byte:
                    break
                if j % 4 == 0:
                    banned.append(out[j // 4 + (n - 1)])
                start = j + 1
            if banned:
                logits[i, banned] = float("-inf")
        return logits


def _engine_kwargs() -> dict[str, Any]:
    # SGLang's Engine kwargs are ServerArgs fields. Keep this env-driven so
    # bbox/throughput sweeps can redeploy config-only variants.
    kwargs: dict[str, Any] = dict(
        model_path=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        context_length=CONTEXT_LENGTH,
        tp_size=1,
        enable_multimodal=True,
        disable_cuda_graph=_env_bool("SGLANG_DISABLE_CUDA_GRAPH", "0"),
        disable_radix_cache=_env_bool("SGLANG_DISABLE_RADIX_CACHE", "0"),
        attention_backend=os.environ.get("SGLANG_ATTENTION_BACKEND", "fa3"),
        mm_attention_backend=os.environ.get("SGLANG_MM_ATTENTION_BACKEND", "fa3"),
        mem_fraction_static=_env_float("SGLANG_MEM_FRACTION_STATIC", "0.85"),
        max_running_requests=_env_int("SGLANG_MAX_RUNNING_REQUESTS", "8"),
        chunked_prefill_size=_env_int("SGLANG_CHUNKED_PREFILL_SIZE", "8192"),
        log_level="warning",
    )
    extra = os.environ.get("SGLANG_ENGINE_EXTRA", "").strip()
    if extra:
        parsed = json.loads(extra)
        if not isinstance(parsed, dict):
            raise ValueError("SGLANG_ENGINE_EXTRA must be a JSON object")
        kwargs.update(parsed)
    return kwargs


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, processor
    processor = AutoProcessor.from_pretrained(WEIGHTS, trust_remote_code=True)
    engine = Engine(**_engine_kwargs())
    yield
    if engine is not None:
        engine.shutdown()
    engine = None
    processor = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": engine is not None}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    if engine is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    body = await request.json()
    image_b64 = body.get("image_b64")
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise HTTPException(status_code=422, detail="image_b64 is required")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt is required")
    try:
        max_new_tokens = int(body.get("max_new_tokens", 3072))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="max_new_tokens is not an integer") from exc
    requested_generation: dict[str, Any] = {}
    if body.get("max_new_tokens") is not None:
        requested_generation["max_new_tokens"] = max_new_tokens

    try:
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_b64, validate=False))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="image_b64 is not a valid image") from exc
    image = _resize(image)
    # SGLang's image_data accepts a file path, URL, or base64 string (PIL objects are not a
    # documented input); pass the resized image as a base64 PNG data URI so it round-trips
    # unambiguously through SGLang's image loader.
    _buf = io.BytesIO()
    image.save(_buf, format="PNG")
    image_uri = "data:image/png;base64," + base64.b64encode(_buf.getvalue()).decode()

    # Build the same Qwen3-VL image-then-text chat prompt as the vLLM deployment,
    # then hand SGLang the resized image (base64) as the single visual input.
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    sampling: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
    }
    # A1 (plan 028): optional per-request sampling overrides for the pipeline's
    # escalated retry tier. Each default equals the frozen greedy value, so a
    # request without these fields decodes exactly as before. (The pre-028 shim
    # ignored a client-sent "temperature" entirely; production clients have
    # always sent temperature=0.0, which is also the default — no behavior change.)
    for key, cast in (
        ("temperature", float),
        ("top_p", float),
        ("top_k", int),
        ("min_p", float),
        ("repetition_penalty", float),
    ):
        if body.get(key) is not None:
            try:
                sampling[key] = cast(body[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{key} is not a number") from exc
            requested_generation[key] = sampling[key]
    # SGLang's request-level RNG field is ``sampling_seed`` (not ``seed``).
    # Keep the public shim contract concise (``seed``), but translate explicitly
    # and return both requested/effective values so a client can prove the field
    # was not silently ignored. Omission leaves SamplingParams byte-for-byte as
    # before this feature.
    if "seed" in body:
        seed = body["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > (2**63 - 1):
            raise HTTPException(
                status_code=422,
                detail="seed must be a non-negative integer <= 2^63-1",
            )
        sampling["sampling_seed"] = seed
        requested_generation["seed"] = seed
    # Robustness (gap-1 port from the HF eval pipeline): an OPTIONAL mild repetition
    # penalty discourages the greedy repetition-loop failure mode at the source. Default
    # 1.0 = OFF (no behaviour change). Keep mild (1.03-1.08) if enabled — never use
    # small-n no_repeat_ngram, which bans the legitimately-repeated bbox_2d/text_content
    # keys. The pipeline's _ocr_page_result_is_suspicious -> Gemini fallback still
    # backstops any loop that slips through, so this is purely a latency/cleanliness
    # optimization and MUST re-pass the 130-page gold-GT gate before being enabled in
    # production. A client-sent repetition_penalty (A1 escalation) wins over the env.
    if "repetition_penalty" not in sampling:
        rep_penalty = _env_float("SGLANG_REPETITION_PENALTY", "1.0")
        if rep_penalty and rep_penalty != 1.0:
            sampling["repetition_penalty"] = rep_penalty

    # C1 (plan 028): optional xgrammar-constrained decoding. SGLang's SamplingParams
    # takes json_schema as a string; dict inputs are serialized here.
    # C2 default-on (plan 028 promotion): when the deployment sets
    # QWEN_JSON_SCHEMA_DEFAULT, requests that carry no json_schema of their own
    # decode under that schema — the constrained-decoding rollout is therefore a
    # CONFIG-ONLY change with a config-only rollback (unset the env, redeploy);
    # an explicit request json_schema always wins.
    json_schema = body.get("json_schema") or os.environ.get("QWEN_JSON_SCHEMA_DEFAULT", "").strip()
    if json_schema:
        sampling["json_schema"] = (
            json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
        )

    # A2 (plan 028): large-n no-repeat logits processor, per request. The engine
    # must run with enable_custom_logit_processor=true (SGLANG_ENGINE_EXTRA) or
    # SGLang rejects the request — surfaced as a 422, never silently dropped.
    custom_logit_processor = None
    no_repeat_ngram = int(body.get("no_repeat_ngram") or 0)
    if no_repeat_ngram > 1:
        custom_logit_processor = LargeNoRepeatNgramProcessor.to_str()
        sampling["custom_params"] = {"no_repeat_ngram_size": no_repeat_ngram}

    # B1 (plan 028): per-token logprobs for the chunk-confidence channel.
    return_logprob = bool(body.get("return_logprob"))

    t0 = time.perf_counter()
    try:
        output = await engine.async_generate(
            prompt=prompt_text,
            image_data=[image_uri],
            sampling_params=sampling,
            return_logprob=return_logprob,
            custom_logit_processor=custom_logit_processor,
            rid=request_id,
        )
    except ValueError as exc:
        # e.g. custom logit processor sent to an engine without the startup flag.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    wall_ms = (time.perf_counter() - t0) * 1000
    if not isinstance(output, dict):
        raise HTTPException(status_code=500, detail="SGLang returned an unexpected output")

    meta = output.get("meta_info") or {}
    output_ids = output.get("output_ids") or []
    # finish_reason: SGLang emits {"type": "stop"|"length"|"abort", ...} (or None
    # while streaming; /predict is non-streaming so it is set on every decode).
    # The client reads truncation from this directly instead of inferring it from
    # n_output_tokens >= max_new_tokens.
    fr = meta.get("finish_reason")
    finish_reason = (
        fr.get("type") if isinstance(fr, dict) else (fr if isinstance(fr, str) else None)
    )
    resp: dict[str, Any] = {
        "raw": output.get("text") or "",
        "wall_ms": wall_ms,
        "image_size": list(image.size),
        "n_input_tokens": int(meta.get("prompt_tokens") or 0),
        "n_output_tokens": int(meta.get("completion_tokens") or len(output_ids)),
        "finish_reason": finish_reason,
        "generation": {
            "requested": requested_generation,
            "effective": {
                **{
                    key: sampling[key]
                    for key in (
                        "temperature",
                        "top_p",
                        "top_k",
                        "min_p",
                        "repetition_penalty",
                        "max_new_tokens",
                    )
                    if key in sampling
                },
                "seed": sampling.get("sampling_seed"),
            },
        },
    }
    if return_logprob:
        # meta_info.output_token_logprobs: [(logprob, token_id, None), ...] — the
        # Engine API has no return_text_in_logprobs, so decode each id here (one id
        # per batch_decode element, exactly how SGLang's own detokenizer does it).
        token_logprobs = meta.get("output_token_logprobs") or []
        ids = [int(t[1]) for t in token_logprobs]
        texts = (
            processor.tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False)
            if ids
            else []
        )
        resp["output_token_logprobs"] = [
            [float(t[0]), int(t[1]), s] for t, s in zip(token_logprobs, texts, strict=False)
        ]
    return resp
