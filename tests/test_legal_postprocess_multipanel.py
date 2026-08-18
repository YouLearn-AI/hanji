from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

from extract.core.legal_postprocess.cascade import Word
from extract.core.legal_postprocess.multipanel import transform
from extract.core.legal_postprocess.runtime import (
    _elements,
    _render_table,
    apply_legal_postprocess,
)
from extract.core.ocr.qwen_lora import parse_qwen_lora_response


def _labels(column: int) -> list[str]:
    if column == 0:
        return [str(row % 10) for row in range(1, 16)]
    return [str(row) for row in range(1, 16)]


def _table(labels: list[str], panel: int) -> str:
    rows = "\n".join(
        f"| {label} | panel{panel}row{ordinal} |"
        for ordinal, label in enumerate(labels, 1)
    )
    return f"|  |  |\n|---|---|\n{rows}"


def _fixture() -> tuple[str, list[Word]]:
    records = [{"bbox_2d": [450, 20, 550, 40], "text_content": "cover"}]
    words: list[Word] = []
    page_numbers = [100, 102, 101, 103]  # visual TL, TR, BL, BR; printed order is column-major
    panels = []
    for panel in range(4):
        row, column = divmod(panel, 2)
        x = 100 + 420 * column
        y = 120 + 430 * row
        labels = _labels(column)
        for ordinal, label in enumerate(labels, 1):
            baseline = y + (ordinal - 1) * 15
            words.extend(
                [
                    Word(label, x, baseline, x + 10, baseline + 10),
                    Word(
                        f"panel{panel}row{ordinal}",
                        x + 40,
                        baseline,
                        x + 170,
                        baseline + 10,
                    ),
                ]
            )
        page_number = page_numbers[panel]
        words.append(Word(str(page_number), x + 300, y - 25, x + 330, y - 15))
        panels.append(
            {
                "header": {"bbox_2d": [x + 120, y - 30, x + 250, y - 18], "text_content": f"header{page_number}"},
                "number": {"bbox_2d": [x + 300, y - 30, x + 330, y - 18], "text_content": str(page_number)},
                "table": {"bbox_2d": [x, y - 5, x + 320, y + 220], "text_content": _table(labels, panel)},
                "footer": {"bbox_2d": [x + 100, y + 225, x + 260, y + 235], "text_content": f"footer{page_number}"},
            }
        )

    # Model emission is horizontal-band order, with the last page number especially late.
    for row in range(2):
        left, right = panels[2 * row : 2 * row + 2]
        records.extend([left["header"], left["number"], right["header"], right["number"]])
        records.extend([left["table"], right["table"], left["footer"], right["footer"]])
    records.append({"bbox_2d": [450, 950, 550, 970], "text_content": "sheet footer"})
    return json.dumps(records), words


def _record_counter(raw: str) -> Counter[str]:
    return Counter(json.dumps(record, sort_keys=True) for record in json.loads(raw))


class _Page:
    number = 0
    rotation = 0
    rect = SimpleNamespace(width=1000.0, height=1000.0)

    def __init__(self, words: list[Word]):
        self.words = words

    def get_text(self, mode: str, *, sort: bool = False):
        assert mode == "words"
        assert sort is True
        return [
            (word.x0, word.y0, word.x1, word.y1, word.text, 0, index, 0)
            for index, word in enumerate(self.words)
        ]


def _typed_text(kind: str, item) -> str:
    return _render_table(item) if kind == "table" else item.text


def test_reorders_whole_panels_by_printed_page_and_preserves_records():
    raw, words = _fixture()

    transformed, receipt = transform(raw, words)

    assert receipt["status"] == "transformed"
    assert receipt["order_source"] == "printed_page_number"
    assert receipt["page_numbers"] == [100, 102, 101, 103]
    assert receipt["gutter_evidence"] == ["modulo_ten", "modulo_ten", "literal", "literal"]
    assert _record_counter(transformed) == _record_counter(raw)
    texts = [record["text_content"].splitlines()[0] for record in json.loads(transformed)]
    assert texts == [
        "cover",
        "header100",
        "100",
        "|  |  |",
        "footer100",
        "header101",
        "101",
        "|  |  |",
        "footer101",
        "header102",
        "102",
        "|  |  |",
        "footer102",
        "header103",
        "103",
        "|  |  |",
        "footer103",
        "sheet footer",
    ]


def test_missing_model_page_number_is_not_reconstructed():
    raw, words = _fixture()
    records = json.loads(raw)
    records = [record for record in records if record["text_content"] != "101"]
    raw = json.dumps(records)

    transformed, receipt = transform(raw, words)

    assert receipt["status"] == "transformed"
    assert "101" not in [record["text_content"] for record in json.loads(transformed)]
    assert _record_counter(transformed) == _record_counter(raw)


def test_already_panel_major_is_byte_identical_abstention():
    raw, words = _fixture()
    transformed, first = transform(raw, words)
    assert first["status"] == "transformed"

    second_output, second = transform(transformed, words)

    assert second == {
        "status": "abstain",
        "reason": "already_panel_major",
        "panels": 4,
        "order_source": "printed_page_number",
    }
    assert second_output == transformed


def test_fused_table_abstains_byte_identically():
    raw, words = _fixture()
    records = json.loads(raw)
    records = [record for record in records if not record["text_content"].startswith("|")]
    records.append(
        {
            "bbox_2d": [100, 115, 840, 770],
            "text_content": "|  |  |  |\n|---|---|---|\n"
            + "\n".join(f"| {row} | left{row} | {row} | right{row} |" for row in range(1, 16)),
        }
    )
    raw = json.dumps(records)

    transformed, receipt = transform(raw, words)

    assert receipt["reason"] == "panel_table_count_mismatch"
    assert transformed == raw


def test_truncated_panel_bbox_abstains_byte_identically():
    raw, words = _fixture()
    records = json.loads(raw)
    tables = [record for record in records if record["text_content"].startswith("|")]
    tables[-1]["bbox_2d"][3] = 700
    raw = json.dumps(records)

    transformed, receipt = transform(raw, words)

    assert receipt["reason"] == "panel_table_grounding_ambiguous"
    assert transformed == raw


def test_source_body_mismatch_abstains_byte_identically():
    raw, words = _fixture()
    records = json.loads(raw)
    table = next(record for record in records if record["text_content"].startswith("|"))
    table["text_content"] = table["text_content"].replace("panel0row8", "unrelated words")
    raw = json.dumps(records)

    transformed, receipt = transform(raw, words)

    assert receipt["reason"] == "weak_panel_source_alignment"
    assert transformed == raw


def test_partial_source_page_numbers_abstain_byte_identically():
    raw, words = _fixture()
    words = [word for word in words if word.text != "101"]

    transformed, receipt = transform(raw, words)

    assert receipt["reason"] == "partial_panel_page_numbers"
    assert transformed == raw


def test_missing_source_page_numbers_abstain_instead_of_guessing_visual_order():
    raw, words = _fixture()
    words = [word for word in words if word.text not in {"100", "101", "102", "103"}]

    transformed, receipt = transform(raw, words)

    assert receipt["reason"] == "missing_panel_page_numbers"
    assert transformed == raw


def test_nested_modulo_suffix_is_not_miscounted_as_a_second_panel():
    words = []
    for ordinal in range(1, 26):
        label = str(ordinal % 10)
        y = 100 + (ordinal - 1) * 30
        words.extend(
            [
                Word(label, 100, y, 110, y + 10),
                Word(f"row{ordinal}", 140, y, 220, y + 10),
            ]
        )
    raw = json.dumps(
        [{"bbox_2d": [100, 95, 300, 830], "text_content": _table(_labels(0), 0)}]
    )

    transformed, receipt = transform(raw, words)

    assert receipt == {
        "status": "abstain",
        "reason": "unsupported_panel_gutter_count",
        "gutter_candidates": 1,
    }
    assert transformed == raw


def test_non_table_record_inside_panel_body_abstains_byte_identically():
    raw, words = _fixture()
    records = json.loads(raw)
    records.append({"bbox_2d": [160, 180, 260, 195], "text_content": "stray body"})
    raw = json.dumps(records)

    transformed, receipt = transform(raw, words)

    assert receipt["reason"] == "panel_interior_non_table_record"
    assert transformed == raw


def test_production_adapter_reorders_typed_elements_and_preserves_confidence():
    raw, words = _fixture()
    result = parse_qwen_lora_response(raw, page_width=1000, page_height=1000)
    result.raw = {"raw": raw}
    expected_confidence = {}
    for sequence, (_, _, kind, item) in enumerate(_elements(result)):
        item.confidence = 0.5 + sequence / 100.0
        expected_confidence[(kind, _typed_text(kind, item), tuple(item.bbox))] = item.confidence

    processed, receipt = apply_legal_postprocess(
        result, _Page(words), provider_name="qwen_lora"
    )

    assert receipt["route"] == "multipanel"
    assert receipt["status"] == "transformed"
    emitted = [_typed_text(kind, item) for _, _, kind, item in _elements(processed)]
    assert emitted[0] == "cover"
    assert emitted[1:3] == ["header100", "100"]
    assert "panel0row15" in emitted[3]
    assert emitted[4] == "footer100"
    for _, _, kind, item in _elements(processed):
        key = (kind, _typed_text(kind, item), tuple(item.bbox))
        assert item.confidence == expected_confidence[key]


def test_gemini_final_read_uses_typed_elements_for_multipanel_route():
    raw, words = _fixture()
    result = parse_qwen_lora_response(raw, page_width=1000, page_height=1000)
    # Gemini's native raw convention is deliberately unusable as Qwen XYXY. The final-read
    # adapter must serialize the provider-neutral typed objects before routing.
    result.raw = '[{"bbox_2d":[80,100,90,145],"type":"text","text_content":"wrong"}]'

    processed, receipt = apply_legal_postprocess(result, _Page(words), provider_name="gemini")

    assert receipt["route"] == "multipanel"
    emitted = [_typed_text(kind, item) for _, _, kind, item in _elements(processed)]
    assert emitted[1:3] == ["header100", "100"]
    assert "panel0row15" in emitted[3]
    assert emitted[5:7] == ["header101", "101"]
