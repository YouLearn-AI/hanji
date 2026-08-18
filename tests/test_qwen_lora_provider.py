"""Tests for the Qwen LoRA (blend_75_25) OCR provider.

Covers the OCRProvider protocol surface, endpoint/auth config, bbox_2d_json →
OCRPageResult schema translation (coords, categorization, markdown tables),
cold-start retry behavior, and a contract check that the production parser and
canonical prompt stay in lockstep with the bench adapter.
"""

from __future__ import annotations

import base64
import inspect

import httpx
import pytest

from extract.config import settings
from extract.core.ocr import get_provider
from extract.core.ocr.base import OCRPageResult
from extract.core.ocr.qwen_lora import (
    PROMPT_BBOX_2D_JSON_WITH_IMAGE,
    QwenLoraProvider,
    parse_bbox_2d_json,
    parse_qwen_lora_response,
)

# A real blend_75_25 generation, lifted verbatim from
# the banked production eval slice
# (case v1_0024013_technical_manual__p0001, raw_first_500). Used as the spike
# fixture so the parser is tested against true model output, not a mock-up.
SPIKE_RAW = (
    '[{"bbox_2d": [23, 132, 84, 170], "text_content": "Lynch LLC"}, '
    '{"bbox_2d": [20, 173, 162, 211], "text_content": "Ultrasonic Cleaning System"}, '
    '{"bbox_2d": [20, 226, 162, 244], "text_content": '
    '"Model PRO-2122 \\u00b7 DOC-M7JN4G \\u00b7 Rev B7 \\u00b7 2026-03-26"}, '
    '{"bbox_2d": [25, 282, 69, 301], "text_content": "\\u26a0 WARNING."}]'
)


# ---------------------------------------------------------------------------
# Protocol surface + config
# ---------------------------------------------------------------------------


def test_provider_satisfies_protocol_surface():
    provider = QwenLoraProvider(api_key="test", url="https://example.com/predict")
    assert provider.name == "qwen_lora"
    assert inspect.iscoroutinefunction(provider.ocr_page)
    sig = inspect.signature(provider.ocr_page)
    # The protocol surface plus the OPTIONAL A1 escalation kwarg (plan 028):
    # callers using the bare protocol never pass it, so the protocol holds.
    assert list(sig.parameters) == [
        "image_bytes",
        "page_width",
        "page_height",
        "sampling_overrides",
    ]
    assert sig.parameters["sampling_overrides"].default is None


def test_registry_lazy_loads_qwen_lora(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    provider = get_provider("qwen_lora")
    assert provider is not None
    assert provider.name == "qwen_lora"
    # Second lookup returns the registered singleton.
    assert get_provider("qwen_lora") is provider



def test_endpoint_prefers_explicit_url():
    provider = QwenLoraProvider(api_key="test", url="https://custom/predict")
    assert provider.endpoint_url == "https://custom/predict"




def test_parse_scales_normalized_coords_to_page_points():
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [500, 250, 1000, 500], "text_content": "hello"}]'},
        page_width=100,
        page_height=200,
    )
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "hello"
    # 500/1000*100=50, 250/1000*200=50, 1000/1000*100=100, 500/1000*200=100
    assert result.blocks[0].bbox == [50.0, 50.0, 100.0, 100.0]


def test_parse_clips_out_of_range_coords():
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [-10, 0, 1200, 1000], "text_content": "x"}]'},
        page_width=100,
        page_height=200,
    )
    assert result.blocks[0].bbox == [0.0, 0.0, 100.0, 200.0]


def test_parse_image_sentinel_becomes_figure():
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [0, 0, 500, 500], "text_content": "<image>"}]'},
        page_width=100,
        page_height=100,
    )
    assert result.blocks == []
    assert len(result.figures) == 1
    assert result.figures[0].bbox == [0.0, 0.0, 50.0, 50.0]


def test_parse_checkbox_record_stays_text():
    # Checkbox marks are plain glyphs in text ([x]/[ ] convention) — there is no
    # separate selection element. The <hw> handwriting sentinel (addendum-era
    # training convention) is stripped so it never leaks into customer text.
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [0, 0, 500, 500], "text_content": "<hw>[x] Consent to treat"}]'},
        page_width=100,
        page_height=100,
    )

    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block.text == "[x] Consent to treat"
    assert block.bbox == [0.0, 0.0, 50.0, 50.0]


def test_parse_markdown_table_becomes_ocrtable():
    md = "| Name | Value |\n| --- | --- |\n| Glucose | 123 |\n| HbA1c | 5.4 |"
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [0, 0, 1000, 1000], "text_content": ' + _json_str(md) + "}]"},
        page_width=100,
        page_height=100,
    )
    assert result.blocks == []
    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.n_rows == 3  # header + 2 body rows (delimiter row dropped)
    assert table.n_cols == 2
    assert [(c.text, c.row, c.col) for c in table.cells] == [
        ("Name", 0, 0),
        ("Value", 0, 1),
        ("Glucose", 1, 0),
        ("123", 1, 1),
        ("HbA1c", 2, 0),
        ("5.4", 2, 1),
    ]
    # Every cell shares the table bbox (model emits one bbox per table).
    assert all(c.bbox == table.bbox for c in table.cells)


def test_single_pipe_line_is_prose_not_table():
    # "a | b" with no delimiter row must NOT be misread as a table.
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [0, 0, 100, 50], "text_content": "Section 3 | Appendix"}]'},
        page_width=100,
        page_height=100,
    )
    assert result.tables == []
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "Section 3 | Appendix"


def test_parse_empty_text_is_dropped():
    result = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d": [0, 0, 10, 10], "text_content": "   "}]'},
        page_width=100,
        page_height=100,
    )
    assert result.blocks == [] and result.tables == [] and result.figures == []


def test_parse_strips_code_fence_and_trailing_tokens():
    raw = '```json\n[{"bbox_2d": [0, 0, 1000, 1000], "text_content": "x"}]\n```<|im_end|>'
    result = parse_qwen_lora_response({"raw": raw}, page_width=10, page_height=10)
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "x"


def test_parse_recovers_truncated_array_via_regex_fallback():
    # No closing ']' — the JSON loader fails, the regex fallback recovers the
    # two complete records (each is a well-formed object; only the array is cut).
    raw = '[{"bbox_2d": [0, 0, 100, 100], "text_content": "a"}, {"bbox_2d": [0, 100, 100, 200], "text_content": "b"}'
    result = parse_qwen_lora_response({"raw": raw}, page_width=1000, page_height=1000)
    texts = [b.text for b in result.blocks]
    assert texts == ["a", "b"]


def test_parse_preserves_generation_order():
    # 2026-07-30 reading-order fix: the model emits records in reading order
    # (column-aware, tables in place); the parser must NOT re-sort them by
    # geometry. A bbox-center sort interleaves the columns of a 2-column page —
    # measured −0.137 mean text_accuracy on extract-bench (see
    # the internal measurement records). Records
    # emitted bottom-then-top stay bottom-then-top, and each element carries
    # its generation index in ``seq`` for the page-assembly interleave.
    raw = (
        '[{"bbox_2d": [0, 800, 100, 900], "text_content": "first-emitted"}, '
        '{"bbox_2d": [0, 100, 100, 200], "text_content": "second-emitted"}]'
    )
    result = parse_qwen_lora_response({"raw": raw}, page_width=100, page_height=1000)
    assert [b.text for b in result.blocks] == ["first-emitted", "second-emitted"]
    assert [b.seq for b in result.blocks] == [0, 1]


def test_parse_seq_indexes_span_types():
    # seq is the RECORD index, shared across blocks/tables/figures/KVs, so the
    # assembly can interleave types at their emitted positions.
    raw = (
        '[{"bbox_2d": [0, 0, 100, 100], "text_content": "prose"}, '
        '{"bbox_2d": [0, 200, 100, 300], "text_content": "<image>"}, '
        '{"bbox_2d": [0, 400, 100, 500], "text_content": "| a | b |\\n| --- | --- |\\n| 1 | 2 |"}, '
        '{"bbox_2d": [0, 600, 100, 700], "text_content": "tail"}]'
    )
    result = parse_qwen_lora_response({"raw": raw}, page_width=100, page_height=1000)
    assert [b.seq for b in result.blocks] == [0, 3]
    assert [f.seq for f in result.figures] == [1]
    assert [t.seq for t in result.tables] == [2]


def test_parse_accepts_bare_list_and_string_envelopes():
    records = [{"bbox_2d": [0, 0, 100, 100], "text_content": "z"}]
    as_list = parse_qwen_lora_response(records, page_width=10, page_height=10)
    as_str = parse_qwen_lora_response(
        '[{"bbox_2d": [0, 0, 100, 100], "text_content": "z"}]', page_width=10, page_height=10
    )
    assert as_list.blocks[0].text == as_str.blocks[0].text == "z"


def test_spike_fixture_parses_clean(monkeypatch):
    """Real blend_75_25 output parses to the expected blocks (verification spike)."""
    result = parse_qwen_lora_response({"raw": SPIKE_RAW}, page_width=612, page_height=792)
    assert result.tables == [] and result.figures == []
    texts = [b.text for b in result.blocks]
    assert "Lynch LLC" in texts
    assert "Ultrasonic Cleaning System" in texts
    assert "⚠ WARNING." in texts  # unicode survives the round trip
    # 0-1000 normalized → page points, all within the page.
    for b in result.blocks:
        x0, y0, x1, y1 = b.bbox
        assert 0 <= x0 <= x1 <= 612
        assert 0 <= y0 <= y1 <= 792


# ---------------------------------------------------------------------------
# Cold-start retry behavior
# ---------------------------------------------------------------------------


# Capture the real client class once, before any test patches the module
# attribute (patching mod.httpx.AsyncClient mutates the shared httpx module, so
# a factory that referenced httpx.AsyncClient by name would recurse into itself).
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_client_factory(responses, *, sleeps):
    """Build a patch for httpx.AsyncClient backed by a scripted MockTransport.

    ``responses`` is a list of (status_code, json_or_exc). Each request pops the
    next entry; a BaseException entry is raised instead of returned.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = calls["n"]
        calls["n"] += 1
        status, body = responses[i]
        if isinstance(body, BaseException):
            raise body
        return httpx.Response(status, json=body)

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    return factory, calls


@pytest.fixture
def no_sleep(monkeypatch):
    import extract.core.ocr.qwen_lora as mod

    async def _noop(_):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _noop)


async def test_retry_on_503_then_success(monkeypatch, no_sleep):
    import extract.core.ocr.qwen_lora as mod

    factory, calls = _mock_client_factory(
        [
            (503, {}),
            (200, {"raw": '[{"bbox_2d": [0, 0, 1000, 1000], "text_content": "ok"}]'}),
        ],
        sleeps=[],
    )
    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    provider = QwenLoraProvider(api_key="k", url="https://example.com/predict")
    result = await provider.ocr_page(b"png", page_width=10, page_height=10)
    assert calls["n"] == 2  # one retry
    assert result.blocks[0].text == "ok"


async def test_retry_on_read_timeout_then_success(monkeypatch, no_sleep):
    import extract.core.ocr.qwen_lora as mod

    factory, calls = _mock_client_factory(
        [
            (0, httpx.ReadTimeout("timed out")),
            (200, {"raw": "[]"}),
        ],
        sleeps=[],
    )
    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    provider = QwenLoraProvider(api_key="k", url="https://example.com/predict")
    result = await provider.ocr_page(b"png", page_width=10, page_height=10)
    assert calls["n"] == 2
    assert isinstance(result, OCRPageResult)


async def test_no_retry_on_4xx(monkeypatch, no_sleep):
    import extract.core.ocr.qwen_lora as mod

    factory, calls = _mock_client_factory([(422, {"detail": "bad image"})], sleeps=[])
    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    provider = QwenLoraProvider(api_key="k", url="https://example.com/predict")
    with pytest.raises(httpx.HTTPStatusError):
        await provider.ocr_page(b"png", page_width=10, page_height=10)
    assert calls["n"] == 1  # deterministic — not retried


async def test_retry_exhausted_raises(monkeypatch, no_sleep):
    import extract.core.ocr.qwen_lora as mod

    factory, calls = _mock_client_factory([(503, {})] * 3, sleeps=[])
    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    provider = QwenLoraProvider(api_key="k", url="https://example.com/predict")
    with pytest.raises(httpx.HTTPStatusError):
        await provider.ocr_page(b"png", page_width=10, page_height=10)
    assert calls["n"] == 3


async def test_payload_sends_canonical_prompt_and_b64_image(monkeypatch, no_sleep):
    import extract.core.ocr.qwen_lora as mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"raw": "[]"})

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    provider = QwenLoraProvider(api_key="secret", url="https://example.com/predict")
    await provider.ocr_page(b"\x89PNG", page_width=10, page_height=10)

    assert captured["body"]["prompt"] == PROMPT_BBOX_2D_JSON_WITH_IMAGE
    assert captured["body"]["temperature"] == 0.0
    assert base64.b64decode(captured["body"]["image_b64"]) == b"\x89PNG"
    assert captured["auth"] == "Api-Key secret"


# ---------------------------------------------------------------------------
# Contract: production parser/prompt stay in lockstep with the bench adapter
# ---------------------------------------------------------------------------




def _json_str(s: str) -> str:
    import json

    return json.dumps(s)


# ---------------------------------------------------------------------------
# Truncation detection honors the EFFECTIVE per-request cap (plan 035 cycle-4)
# ---------------------------------------------------------------------------


async def _ocr_with_envelope(monkeypatch, envelope, *, overrides=None, max_new_tokens=8192):
    import extract.core.ocr.qwen_lora as mod

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope)

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)
    provider = QwenLoraProvider(api_key="secret", url="https://example.com/predict", max_new_tokens=max_new_tokens)
    return await provider.ocr_page(
        b"\x89PNG", page_width=10, page_height=10, sampling_overrides=overrides
    )


async def test_truncation_uses_instance_cap_without_override(monkeypatch, no_sleep):
    # no finish_reason -> infer from token count vs the instance cap (8192)
    env = {"raw": "[]", "n_output_tokens": 8192}
    r = await _ocr_with_envelope(monkeypatch, env, max_new_tokens=8192)
    assert r.truncated is True
    r2 = await _ocr_with_envelope(monkeypatch, {"raw": "[]", "n_output_tokens": 5000})
    assert r2.truncated is False


async def test_retry_below_higher_override_cap_not_truncated(monkeypatch, no_sleep):
    # THE FIX: a recovery retry overrides max_new_tokens upward (12288). A read
    # that finished NATURALLY at 9000 tokens (< 12288, no finish_reason) must NOT
    # be mis-flagged truncated just because 9000 >= the instance default (8192).
    env = {"raw": "[]", "n_output_tokens": 9000}
    r = await _ocr_with_envelope(
        monkeypatch, env, overrides={"no_repeat_ngram": 100, "max_new_tokens": 12288}
    )
    assert r.truncated is False
    # but hitting the higher cap IS truncation
    env_hit = {"raw": "[]", "n_output_tokens": 12288}
    r2 = await _ocr_with_envelope(
        monkeypatch, env_hit, overrides={"max_new_tokens": 12288}
    )
    assert r2.truncated is True


async def test_finish_reason_wins_over_token_heuristic(monkeypatch, no_sleep):
    # when the serving envelope carries a real finish_reason it is authoritative
    env = {"raw": "[]", "n_output_tokens": 12288, "finish_reason": "stop"}
    r = await _ocr_with_envelope(monkeypatch, env, overrides={"max_new_tokens": 12288})
    assert r.truncated is False


def test_recovery_is_loop_guard_only_inheriting_first_pass_cap():
    # cycle-4: the first pass owns the generous budget (8192), so recovery does
    # NOT escalate the cap (a direct e2e showed escalation served none of the
    # too-big docs). It is the loop-killer alone and inherits the first-pass cap
    # by omitting max_new_tokens — genuinely different from the first greedy pass
    # (turns a degenerate loop into a fast fallback) without extra latency.
    from extract.core.ocr.qwen_lora import RECOVERY_RETRY_SAMPLING

    assert RECOVERY_RETRY_SAMPLING == {"no_repeat_ngram": 100}
    assert "max_new_tokens" not in RECOVERY_RETRY_SAMPLING  # inherits first-pass budget
