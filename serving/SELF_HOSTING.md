# Self-hosting the parse model

The API talks to any endpoint implementing the `/predict` contract in
[`server.py`](server.py). This directory is a complete recipe for serving the
published weights — [hanji-dev/hanji-parse-4b](https://huggingface.co/hanji-dev/hanji-parse-4b),
a merged Qwen3-VL-4B fine-tune (~8.3 GiB, bf16) — with
[SGLang](https://github.com/sgl-project/sglang).

**Hardware**: one NVIDIA GPU with ~12 GB+ VRAM (production ran an H100; an
A10G/L4/4090 works at lower throughput). CUDA 12.x.

## Option 1 — docker compose (from the repo root)

```bash
docker compose --profile gpu up
```

This starts the SGLang container with `server.py` mounted, downloads the
weights from Hugging Face on first boot, and starts the API pointed at it.

## Option 2 — plain docker

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/serving:/workspace -w /workspace \
  -v hf-cache:/root/.cache/huggingface \
  -e MODEL_PATH=hanji-dev/hanji-parse-4b \
  -e QWEN_MAX_IMAGE_PIXELS=2000000 \
  -e SGLANG_ATTENTION_BACKEND=fa3 \
  -e SGLANG_MM_ATTENTION_BACKEND=fa3 \
  -e SGLANG_ENGINE_EXTRA='{"speculative_algorithm":"NGRAM","speculative_num_draft_tokens":16,"speculative_ngram_max_bfs_breadth":10,"enable_custom_logit_processor":true}' \
  lmsysorg/sglang:v0.5.12.post1-cu129 \
  bash -c "pip install fastapi uvicorn pillow && uvicorn server:app --host 0.0.0.0 --port 8000"
```

Then point the API at it:

```bash
export PARSE_MODEL_URL=http://localhost:8000/predict
```

## The production engine configuration

These are the settings the model was measured and shipped with:

| Env var | Value | Why |
| --- | --- | --- |
| `MODEL_PATH` | `hanji-dev/hanji-parse-4b` | HF repo id or a local dir of merged weights |
| `QWEN_MAX_IMAGE_PIXELS` | `2000000` | The training pixel cap. The server downscales to ≤2 MP and floors each dimension to a multiple of 32 — the model's 0–1000 bboxes are calibrated to exactly this preprocessing. Do not change. |
| `SGLANG_ATTENTION_BACKEND` / `SGLANG_MM_ATTENTION_BACKEND` | `fa3` | FlashAttention-3 (H100). Use `flashinfer`/`triton` on older GPUs. |
| `SGLANG_MEM_FRACTION_STATIC` | `0.85` | KV-cache sizing |
| `SGLANG_MAX_RUNNING_REQUESTS` | `8` | Production concurrency per replica |
| `SGLANG_CHUNKED_PREFILL_SIZE` | `8192` | |
| `SGLANG_ENGINE_EXTRA` | `{"speculative_algorithm":"NGRAM", "speculative_num_draft_tokens":16, "speculative_ngram_max_bfs_breadth":10, "enable_custom_logit_processor":true}` | NGRAM speculative decoding drafts from the repeated JSON keys of the output contract — ~5× decode speedup at zero quality cost under greedy verification. |

The model is **prompt-coupled**: the client sends the exact production prompt
(`extract.parse_prompts.PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE`), greedy
decoding, `max_new_tokens=8192`. If you call `/predict` yourself, use that
prompt verbatim — any other phrasing produces off-distribution output.

## Byte-determinism (optional)

Stock SGLang with NGRAM speculation is not byte-deterministic across
co-batched requests: the n-gram draft cache is shared across requests, so the
accepted draft path (and, in rare ties, the emitted token) can depend on what
else is in the batch. Production closed this with two measures:

1. [`request_local_ngram.patch`](request_local_ngram.patch) — a small SGLang
   patch making the NGRAM draft trie request-local. Apply to the SGLang
   source tree inside the image (`patch -p1 < request_local_ngram.patch` in
   `sglang/jit_kernel/csrc/ngram_corpus`) and rebuild the JIT kernel.
2. `"enable_deterministic_inference": true` in `SGLANG_ENGINE_EXTRA`, which
   fixes batch-composition nondeterminism, plus
   `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True`.

With both, identical requests produce byte-identical output under
heterogeneous co-batching. If you don't need reproducibility guarantees you
can skip this — accuracy is unaffected either way.

## The /predict contract

```
POST /predict
{"image_b64": "<base64 PNG>", "prompt": "<the production prompt>", "max_new_tokens": 8192}
→ {"raw": "<model output>", "wall_ms": ..., "image_size": [w, h],
   "n_input_tokens": ..., "n_output_tokens": ..., "finish_reason": "...",
   "generation": {"requested": {...}, "effective": {...}}}
```

Optional request fields: sampling overrides (`temperature`, `top_p`, `top_k`,
`min_p`, `repetition_penalty`, `seed`), `return_logprob` (feeds the pipeline's
per-chunk confidence), `json_schema` (xgrammar-constrained decoding), and
`no_repeat_ngram` (large-n anti-loop logit processor). All defaults reproduce
the frozen greedy decode.
