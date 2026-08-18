"""Plan 028 workstream B — token→chunk confidence alignment (B2), the
confidence coercion fix (B3), and the contract pins that keep the evals2
candidate and the production provider on one prompt/schema."""

from __future__ import annotations

import json
import math

import pymupdf
import pytest

from extract.core.ocr.base import OCRBlock, OCRPageResult
from extract.core.ocr.chunk_confidence import (
    ChunkConfidence,
    _token_offsets,
    align_chunk_confidences,
)
from extract.core.ocr.qwen_lora import (
    CHUNK_JSON_SCHEMA,
    ESCALATED_RETRY_SAMPLING,
    PROMPT_BBOX_2D_JSON,
    parse_qwen_lora_response,
)
from extract.core.pdf import _chunks_from_ocr_page_result


def _tokens_for(raw: str, pieces: list[str], logprob: float = -0.1) -> list[list]:
    """Build an output_token_logprobs envelope whose texts concatenate to raw."""
    assert "".join(pieces) == raw
    return [[logprob, 1000 + i, p] for i, p in enumerate(pieces)]


# --------------------------------------------------------------------------- #
# B2 — alignment
# --------------------------------------------------------------------------- #
def test_single_chunk_alignment_mean_min_std():
    raw = '[{"bbox_2d":[1,2,3,4],"text_content":"hi there"}]'
    pieces = ['[{"bbox_2d"', ":[1,2,3,4],", '"text_content":"', "hi", " there", '"}]']
    toks = _tokens_for(raw, pieces)
    # give the two value tokens distinct logprobs
    toks[3][0] = -0.5
    toks[4][0] = -1.5
    out = align_chunk_confidences(raw, toks, ["hi there"])
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, ChunkConfidence)
    assert c.n_tokens == 2
    assert c.mean_logprob == pytest.approx(-1.0)
    assert c.min_logprob == pytest.approx(-1.5)
    assert c.std_logprob == pytest.approx(0.5)
    assert c.confidence == pytest.approx(math.exp(-1.0))


def test_multi_chunk_forward_cursor_handles_duplicate_text():
    raw = (
        '[{"bbox_2d":[0,0,10,10],"text_content":"total"},'
        '{"bbox_2d":[0,20,10,30],"text_content":"total"}]'
    )
    pieces = [
        '[{"bbox_2d":[0,0,10,10],"text_content":"',
        "total",
        '"},{"bbox_2d":[0,20,10,30],"text_content":"',
        "total",
        '"}]',
    ]
    toks = _tokens_for(raw, pieces)
    toks[1][0] = -0.2  # first "total"
    toks[3][0] = -2.0  # second "total"
    out = align_chunk_confidences(raw, toks, ["total", "total"])
    assert out[0].mean_logprob == pytest.approx(-0.2)
    assert out[1].mean_logprob == pytest.approx(-2.0)


def test_json_escaped_value_is_found():
    text = 'say "hi"\nplease'
    raw = json.dumps([{"bbox_2d": [1, 2, 3, 4], "text_content": text}], ensure_ascii=False)
    # one token per char — crude but valid (texts concatenate to raw)
    toks = _tokens_for(raw, list(raw))
    out = align_chunk_confidences(raw, toks, [text])
    assert out[0] is not None
    assert out[0].n_tokens == len(json.dumps(text, ensure_ascii=False)[1:-1])


def test_replacement_char_tokens_grouped_and_anchored():
    # Multi-byte split: two U+FFFD fragments stand in for one CJK char; the
    # following clean token re-anchors and the group still carries logprobs.
    raw = '[{"bbox_2d":[1,2,3,4],"text_content":"水 ok"}]'
    toks = [
        [-0.1, 1, '[{"bbox_2d":[1,2,3,4],"text_content":"'],
        [-3.0, 2, "�"],
        [-3.0, 3, "�"],
        [-0.2, 4, " ok"],
        [-0.1, 5, '"}]'],
    ]
    out = align_chunk_confidences(raw, toks, ["水 ok"])
    assert out[0] is not None
    # the fffd group + " ok" all overlap the value span
    assert out[0].n_tokens == 3


def test_unalignable_stream_returns_none_per_chunk():
    raw = '[{"bbox_2d":[1,2,3,4],"text_content":"hi"}]'
    toks = [[-0.1, 1, "completely"], [-0.1, 2, "different"]]
    out = align_chunk_confidences(raw, toks, ["hi"])
    assert out == [None]


def test_missing_value_yields_none_only_for_that_chunk():
    raw = '[{"bbox_2d":[1,2,3,4],"text_content":"present"}]'
    pieces = ['[{"bbox_2d":[1,2,3,4],"text_content":"', "present", '"}]']
    out = align_chunk_confidences(raw, _tokens_for(raw, pieces), ["absent", "present"])
    assert out[0] is None
    assert out[1] is not None


def test_empty_inputs():
    assert align_chunk_confidences("", [], []) == []
    assert align_chunk_confidences("x", [], ["a"]) == [None]
    assert align_chunk_confidences("", [[-0.1, 1, "x"]], ["a"]) == [None]


def test_token_offsets_trailing_special_token():
    raw = "abc"
    offsets = _token_offsets(raw, ["abc", "<|im_end|>"])
    assert offsets[0] == (0, 3)
    # trailing unmatched token collapses to the tail — zero-width, harmless
    assert offsets[1] == (3, 3)


# --------------------------------------------------------------------------- #
# provider integration — confidences attach through the real parse path
# --------------------------------------------------------------------------- #
def test_parse_qwen_lora_response_attaches_confidence_from_envelope():
    raw = '[{"bbox_2d":[100,100,400,140],"text_content":"hello world"}]'
    pieces = ['[{"bbox_2d":[100,100,400,140],"text_content":"', "hello", " world", '"}]']
    envelope = {
        "raw": raw,
        "n_output_tokens": 4,
        "output_token_logprobs": _tokens_for(raw, pieces, logprob=-0.25),
    }
    res = parse_qwen_lora_response(envelope, page_width=612, page_height=792)
    assert len(res.blocks) == 1
    assert res.blocks[0].confidence == pytest.approx(math.exp(-0.25))


def test_parse_qwen_lora_response_without_logprobs_is_none():
    res = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d":[1,2,3,4],"text_content":"hi"}]'},
        page_width=612,
        page_height=792,
    )
    assert res.blocks[0].confidence is None


# --------------------------------------------------------------------------- #
# B3 — the `or None` coercion trap: a REAL 0.0 must survive to the Chunk
# --------------------------------------------------------------------------- #
def test_zero_confidence_survives_chunk_mapping():
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    page_result = OCRPageResult(
        blocks=[OCRBlock(text="low conf text", bbox=[10, 10, 200, 30], confidence=0.0)]
    )
    chunks = _chunks_from_ocr_page_result(
        doc, 0, page_result, image_bytes=b"", skip_text=False, include_figures=False
    )
    assert len(chunks) == 1
    assert chunks[0].confidence == 0.0  # not None — the coercion trap is fixed


def test_none_confidence_stays_none():
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    page_result = OCRPageResult(
        blocks=[OCRBlock(text="no signal", bbox=[10, 10, 200, 30], confidence=None)]
    )
    chunks = _chunks_from_ocr_page_result(
        doc, 0, page_result, image_bytes=b"", skip_text=False, include_figures=False
    )
    assert chunks[0].confidence is None


# --------------------------------------------------------------------------- #
# contract pins
# --------------------------------------------------------------------------- #

def test_chunk_json_schema_accepts_contract_and_rejects_drift():
    import jsonschema

    ok = [{"bbox_2d": [0, 10, 990, 1000], "text_content": "a | b"}]
    jsonschema.validate(ok, CHUNK_JSON_SCHEMA)
    for bad in (
        [{"bbox_2d": [0, 10, 990], "text_content": "x"}],          # 3 coords
        [{"bbox_2d": [0, 10, 990, 1000]}],                          # no text
        [{"bbox_2d": [0, 1, 2, 3], "text_content": "x", "category": "t"}],  # extra key
        [{"bbox_2d": [0.5, 1, 2, 3], "text_content": "x"}],         # float coord
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, CHUNK_JSON_SCHEMA)



