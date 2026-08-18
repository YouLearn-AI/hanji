"""Production adapter for the validated Plan 132 legal post-processing cascade.

The transforms operate on the model's normalized record contract. This module is the
only bridge to production ``OCRPageResult`` objects and native PDF pages. Every rejected
route returns the original object by identity; source text is evidence only and is never
copied into the customer response.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from extract.core.legal_postprocess.cascade import Word, transform_raw
from extract.core.legal_postprocess.concordance import transform as transform_concordance
from extract.core.legal_postprocess.locator import transform as transform_locator
from extract.core.legal_postprocess.locator_pdf_inspector import (
    eligibility as locator_fallback_eligibility,
)
from extract.core.legal_postprocess.locator_pdf_inspector import (
    transform as transform_locator_fallback,
)
from extract.core.legal_postprocess.multipanel import transform as transform_multipanel
from extract.core.legal_postprocess.wrapped import transform as transform_wrapped
from extract.core.ocr.base import OCRPageResult
from extract.core.ocr.qwen_lora import (
    _extract_raw_text,
    parse_bbox_2d_records,
    parse_qwen_lora_response,
)

_EDGE_TOLERANCE = 1.0


def _normalized_bbox(bbox: list[float], width: float, height: float) -> list[int]:
    return [
        int(round(1000.0 * bbox[0] / width)),
        int(round(1000.0 * bbox[1] / height)),
        int(round(1000.0 * bbox[2] / width)),
        int(round(1000.0 * bbox[3] / height)),
    ]


def _render_table(table) -> str:
    if table.n_rows <= 0 or table.n_cols <= 0:
        return ""
    grid = [["" for _ in range(table.n_cols)] for _ in range(table.n_rows)]
    for cell in table.cells:
        for row_delta in range(cell.row_span):
            for col_delta in range(cell.col_span):
                row = cell.row + row_delta
                col = cell.col + col_delta
                if 0 <= row < table.n_rows and 0 <= col < table.n_cols and not grid[row][col]:
                    grid[row][col] = cell.text.replace("|", r"\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(grid[0]) + " |",
        "| " + " | ".join(["---"] * table.n_cols) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in grid[1:])
    return "\n".join(lines)


def _elements(page_result: OCRPageResult):
    entries: list[tuple[float, int, str, Any]] = []

    def add(items, kind: str) -> None:
        for item in items:
            seq = getattr(item, "seq", None)
            entries.append(
                (float(seq) if seq is not None else float("inf"), len(entries), kind, item)
            )

    add(page_result.blocks, "text")
    add(page_result.key_values, "kv")
    add(page_result.tables, "table")
    add(page_result.figures, "image")
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return entries


def _canonical_raw(page_result: OCRPageResult, width: float, height: float) -> str:
    records = []
    for _, _, kind, item in _elements(page_result):
        if kind in ("text", "kv"):
            text = item.text
        elif kind == "table":
            text = _render_table(item)
        else:
            text = "<image>"
        record: dict[str, Any] = {
            "bbox_2d": _normalized_bbox(item.bbox, width, height),
            "text_content": text,
        }
        if kind == "kv":
            record["type"] = "kv"
        records.append(record)
    return json.dumps(records, ensure_ascii=False)


def _source_raw(
    page_result: OCRPageResult, *, provider_name: str, width: float, height: float
) -> str:
    if provider_name.startswith("qwen") and page_result.raw is not None:
        raw = _extract_raw_text(page_result.raw)
        if raw:
            return raw
    return _canonical_raw(page_result, width, height)


def _native_words(page) -> tuple[list[Word], list[tuple]]:
    width = float(page.rect.width)
    height = float(page.rect.height)
    words: list[Word] = []
    tuples: list[tuple] = []
    for item in page.get_text("words", sort=True):
        x0, y0, x1, y1, text = item[:5]
        text = str(text)
        if not text.strip() or not (x0 < x1 and y0 < y1):
            continue
        normalized = (
            text,
            1000.0 * float(x0) / width,
            1000.0 * float(y0) / height,
            1000.0 * float(x1) / width,
            1000.0 * float(y1) / height,
        )
        tuples.append(normalized)
        words.append(Word(*normalized))
    return words, tuples


def _pdf_inspector_evidence(pdf_bytes: bytes, page) -> dict:
    if int(page.rotation or 0) % 360:
        raise ValueError("rotated_page_unsupported")
    import pdf_inspector

    page_no = int(page.number) + 1
    width = float(page.rect.width)
    height = float(page.rect.height)
    positioned = pdf_inspector.extract_text_with_positions_bytes(pdf_bytes, pages=[page_no])
    items = []
    for item in positioned:
        if int(item.page) != page_no:
            continue
        text = str(item.text)
        x = float(item.x)
        y = float(item.y)
        item_width = float(item.width)
        item_height = float(item.height)
        if not text.strip() or item_width <= 0 or item_height <= 0:
            continue
        bbox = [
            1000.0 * x / width,
            1000.0 * (height - y - item_height) / height,
            1000.0 * (x + item_width) / width,
            1000.0 * (height - y) / height,
        ]
        if not (
            -_EDGE_TOLERANCE <= bbox[0] < bbox[2] <= 1000.0 + _EDGE_TOLERANCE
            and -_EDGE_TOLERANCE <= bbox[1] < bbox[3] <= 1000.0 + _EDGE_TOLERANCE
        ):
            raise ValueError("pdf_inspector_item_outside_page")
        items.append(
            {
                "text": text,
                "bbox_2d": [round(min(1000.0, max(0.0, value)), 4) for value in bbox],
            }
        )
    return {
        "schema": "pdf_inspector_positioned_text/v1",
        "engine": "pdf-inspector==0.2.6",
        "page": page_no,
        "page_width": width,
        "page_height": height,
        "items": items,
    }


def _cascade(raw: str, words: list[Word], native: list[tuple], evidence: dict | None):
    route_receipts: dict[str, dict] = {}

    transformed, receipt = transform_raw(raw, words)
    route_receipts["transcript"] = receipt
    if receipt["status"] == "transformed":
        return transformed, {**receipt, "route": "transcript", "routes": route_receipts}

    transformed, receipt = transform_multipanel(raw, words)
    route_receipts["multipanel"] = receipt
    if receipt["status"] == "transformed":
        return transformed, {**receipt, "route": "multipanel", "routes": route_receipts}

    transformed, receipt = transform_concordance(raw, native)
    route_receipts["concordance"] = receipt
    if receipt["status"] == "transformed":
        return transformed, {**receipt, "route": "concordance", "routes": route_receipts}

    transformed, receipt = transform_locator(raw)
    route_receipts["locator"] = receipt
    if receipt["status"] == "transformed":
        return transformed, {**receipt, "route": "locator", "routes": route_receipts}

    fallback_eligibility = locator_fallback_eligibility(raw)
    if fallback_eligibility["eligible"] and evidence is None:
        route_receipts["locator_pdf_inspector"] = {
            "status": "abstain",
            "reason": "locator_source_geometry_not_loaded",
        }
        return raw, {
            "status": "abstain",
            "reason": "no_route_accepted",
            "needs_locator_evidence": True,
            "routes": route_receipts,
        }
    if fallback_eligibility["eligible"]:
        transformed, receipt = transform_locator_fallback(raw, evidence or {})
        route_receipts["locator_pdf_inspector"] = receipt
        if receipt["status"] == "transformed":
            return transformed, {
                **receipt,
                "route": "locator_pdf_inspector",
                "routes": route_receipts,
            }
    else:
        route_receipts["locator_pdf_inspector"] = {
            "status": "abstain",
            "reason": fallback_eligibility["reason"],
        }

    transformed, receipt = transform_wrapped(raw, native)
    route_receipts["wrapped"] = receipt
    if receipt["status"] == "transformed":
        return transformed, {**receipt, "route": "wrapped", "routes": route_receipts}
    return raw, {"status": "abstain", "reason": "no_route_accepted", "routes": route_receipts}


def _signature(record) -> tuple:
    return (*record[:4], record[4], record[5])


def _intersection_fraction(left, right) -> float:
    ix = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    area = max(0.0, (left[2] - left[0]) * (left[3] - left[1]))
    return ix * iy / area if area else 0.0


def _apply_confidence(
    original: OCRPageResult, result: OCRPageResult, source_raw: str, transformed_raw: str
) -> None:
    source_elements = {
        int(seq): item for seq, _, _, item in _elements(original) if seq != float("inf")
    }
    source_records = parse_bbox_2d_records(source_raw)
    confidence_by_signature: dict[tuple, deque] = defaultdict(deque)
    source_boxes: list[tuple[tuple[int, int, int, int], float | None]] = []
    for index, record in enumerate(source_records):
        item = source_elements.get(index)
        confidence = getattr(item, "confidence", None) if item is not None else None
        confidence_by_signature[_signature(record)].append(confidence)
        source_boxes.append((record[:4], confidence))

    output_records = parse_bbox_2d_records(transformed_raw)
    output_elements = {
        int(seq): item for seq, _, _, item in _elements(result) if seq != float("inf")
    }
    for index, record in enumerate(output_records):
        item = output_elements.get(index)
        if item is None:
            continue
        queue = confidence_by_signature.get(_signature(record))
        if queue:
            confidence = queue.popleft()
        else:
            overlaps = [
                confidence
                for bbox, confidence in source_boxes
                if confidence is not None and _intersection_fraction(bbox, record[:4]) >= 0.5
            ]
            confidence = min(overlaps) if overlaps else None
        item.confidence = confidence
        if hasattr(item, "cells"):
            for cell in item.cells:
                cell.confidence = confidence


def apply_legal_postprocess(
    page_result: OCRPageResult,
    page,
    *,
    provider_name: str,
    pdf_bytes: bytes | None = None,
) -> tuple[OCRPageResult, dict]:
    """Apply the cascade to one final served OCR page, failing closed on any error."""
    try:
        width = float(page.rect.width)
        height = float(page.rect.height)
        raw = _source_raw(page_result, provider_name=provider_name, width=width, height=height)
        words, native = _native_words(page)
        evidence = _pdf_inspector_evidence(pdf_bytes, page) if pdf_bytes is not None else None
        transformed, receipt = _cascade(raw, words, native, evidence)
        if receipt["status"] != "transformed":
            return page_result, receipt
        parsed = parse_qwen_lora_response(transformed, page_width=width, page_height=height)
        if not any((parsed.blocks, parsed.tables, parsed.figures, parsed.key_values)):
            return page_result, {"status": "abstain", "reason": "transformed_parse_empty"}
        parsed.raw = transformed
        _apply_confidence(page_result, parsed, raw, transformed)
        return parsed, receipt
    except Exception as exc:  # fail closed; page content never enters the receipt/log
        return page_result, {
            "status": "abstain",
            "reason": "postprocess_exception",
            "error_type": type(exc).__name__,
        }
