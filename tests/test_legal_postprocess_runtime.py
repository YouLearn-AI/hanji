from __future__ import annotations

import json
from types import SimpleNamespace

from extract.core.legal_postprocess.runtime import (
    _pdf_inspector_evidence,
    apply_legal_postprocess,
)
from extract.core.ocr.base import OCRBlock, OCRPageResult
from extract.core.ocr.qwen_lora import parse_qwen_lora_response


class _Page:
    number = 0
    rotation = 0
    rect = SimpleNamespace(width=1000.0, height=1000.0)

    def __init__(self, words=()):
        self._words = list(words)

    def get_text(self, mode, *, sort=False):
        assert mode == "words"
        assert sort is True
        return self._words


def _transcript_fixture():
    words = []
    records = []
    for number in range(1, 16):
        y0 = 40 + number * 40
        words.extend(
            [
                (40, y0, 50, y0 + 10, str(number), 0, number, 0),
                (100, y0, 145, y0 + 10, f"answer{number}", 0, number, 1),
            ]
        )
        records.append(
            {
                "bbox_2d": [100, y0, 145, y0 + 10],
                "text_content": f"answer{number}",
            }
        )
    return words, json.dumps(records)


def test_abstention_returns_original_object_by_identity():
    result = OCRPageResult(raw="[]")
    processed, receipt = apply_legal_postprocess(result, _Page(), provider_name="qwen_lora")
    assert processed is result
    assert receipt["status"] == "abstain"


def test_qwen_transcript_route_builds_typed_table_and_preserves_confidence():
    words, raw = _transcript_fixture()
    result = parse_qwen_lora_response(raw, page_width=1000, page_height=1000)
    result.raw = {"raw": raw}
    for block in result.blocks:
        block.confidence = 0.93

    processed, receipt = apply_legal_postprocess(result, _Page(words), provider_name="qwen_lora")

    assert receipt["route"] == "transcript"
    assert len(processed.tables) == 1
    assert processed.tables[0].n_rows == 16  # empty contract header + 15 lines
    assert processed.tables[0].confidence == 0.93
    assert all(cell.confidence == 0.93 for cell in processed.tables[0].cells)


def test_gemini_final_read_uses_typed_xyxy_geometry_not_native_yxyx_raw():
    words, raw = _transcript_fixture()
    records = json.loads(raw)
    result = OCRPageResult(
        blocks=[
            OCRBlock(
                text=record["text_content"],
                bbox=[float(value) for value in record["bbox_2d"]],
                seq=index,
            )
            for index, record in enumerate(records)
        ],
        # Deliberately unusable as Qwen XYXY: the adapter must canonicalize the
        # typed Gemini result instead of consuming its native-YXYX raw response.
        raw='[{"bbox_2d":[80,100,90,145],"type":"text","text_content":"wrong"}]',
    )

    processed, receipt = apply_legal_postprocess(result, _Page(words), provider_name="gemini")

    assert receipt["route"] == "transcript"
    assert processed.tables[0].bbox[0] == 40.0
    assert "answer15" in processed.tables[0].cells[-1].text


def test_locator_title_requests_source_evidence_without_mutating_page():
    raw = json.dumps([{"bbox_2d": [200, 20, 800, 60], "text_content": "TABLE OF CONTENTS"}])
    result = parse_qwen_lora_response(raw, page_width=1000, page_height=1000)
    result.raw = raw

    processed, receipt = apply_legal_postprocess(result, _Page(), provider_name="qwen_lora")

    assert processed is result
    assert receipt["needs_locator_evidence"] is True
    assert receipt["status"] == "abstain"


def test_pdf_inspector_bytes_reads_only_requested_one_based_page(monkeypatch):
    import pdf_inspector

    calls = []

    def extract(data, pages):
        calls.append((data, pages))
        return [
            SimpleNamespace(
                page=2,
                text="TABLE OF AUTHORITIES",
                x=100.0,
                y=800.0,
                width=300.0,
                height=20.0,
            )
        ]

    monkeypatch.setattr(pdf_inspector, "extract_text_with_positions_bytes", extract)
    page = _Page()
    page.number = 1
    evidence = _pdf_inspector_evidence(b"pdf", page)

    assert calls == [(b"pdf", [2])]
    assert evidence["page"] == 2
    texts = [item["text"] for item in evidence["items"]]
    assert texts == ["TABLE OF AUTHORITIES"]
