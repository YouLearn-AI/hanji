"""Fail-closed ordering normalization for already-split transcript panels.

This module does not split, rebuild, or transcribe a multipanel page. It accepts only an
existing one-table-per-panel model result whose tables bind one-to-one to regular native
PDF transcript gutters. Accepted model records are reordered as whole objects; their text
and bboxes are never edited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from extract.core.legal_postprocess.cascade import (
    ALIGN_SEARCH_RADIUS_PITCH,
    GUTTER_X_TOLERANCE,
    MIN_GUTTER_ROWS,
    MIN_TOKEN_F1,
    PITCH_REL_TOLERANCE,
    ROW_Y_TOLERANCE,
    ExistingLineTable,
    Word,
    _candidate_gutters,
    _existing_line_tables,
    _existing_table_alignment,
    _tokens,
)

SUPPORTED_PANEL_COUNTS = (2, 4)
PANEL_BAND_PITCHES = 2.5
RECORD_PANEL_MIN_X_FRACTION = 0.8
GUTTER_BODY_INSET = 12.0
MIN_COMPACT_CHAR_SIMILARITY = 0.9

CONSTANTS = {
    "supported_panel_counts": list(SUPPORTED_PANEL_COUNTS),
    "panel_band_pitches": PANEL_BAND_PITCHES,
    "record_panel_min_x_fraction": RECORD_PANEL_MIN_X_FRACTION,
    "gutter_body_inset": GUTTER_BODY_INSET,
    "min_compact_char_similarity": MIN_COMPACT_CHAR_SIMILARITY,
}


@dataclass(frozen=True)
class PanelGutter:
    words: tuple[Word, ...]
    labels: tuple[str, ...]
    pitch: float
    evidence: str

    @property
    def row_y(self) -> dict[int, float]:
        return {index: word.yc for index, word in enumerate(self.words, 1)}

    @property
    def x_center(self) -> float:
        return median(word.xc for word in self.words)


@dataclass(frozen=True)
class PanelAnchor:
    gutter: PanelGutter
    table: ExistingLineTable
    bbox: tuple[float, float, float, float]
    min_alignment: float
    mean_alignment: float


def _literal_gutters(words: list[Word]) -> list[PanelGutter]:
    return [
        PanelGutter(gutter.words, tuple(word.text for word in gutter.words), gutter.pitch, "literal")
        for gutter in _candidate_gutters(words)
    ]


def _modulo_gutters(words: list[Word]) -> list[PanelGutter]:
    """Recognize a source layer that exposes only the printed ones-place glyph.

    Some condensed transcript PDFs paint the tens glyph outside the native text layer.
    Their visible/source sequence is therefore ``1..9,0..9,0..5``. The regular pitch,
    fixed x position, 15-row minimum, and exact first-column equality to an existing model
    table are the proof; the inferred ordinal is never emitted.
    """

    digits: dict[str, list[Word]] = {str(number): [] for number in range(10)}
    for word in words:
        if word.text in digits and word.xc <= 950:
            digits[word.text].append(word)

    candidates: list[PanelGutter] = []
    for first in digits["1"]:
        sequence = [first]
        for ordinal in range(2, 29):
            choices = [
                word
                for word in digits[str(ordinal % 10)]
                if word.yc > sequence[-1].yc
                and abs(word.xc - first.xc) <= GUTTER_X_TOLERANCE
            ]
            if not choices:
                break
            if len(sequence) == 1:
                chosen = min(choices, key=lambda word: word.yc)
            else:
                pitch = median(
                    right.yc - left.yc
                    for left, right in zip(sequence, sequence[1:], strict=False)
                )
                choices = [
                    word
                    for word in choices
                    if abs((word.yc - sequence[-1].yc) - pitch)
                    <= PITCH_REL_TOLERANCE * pitch
                ]
                if not choices:
                    break
                chosen = min(
                    choices,
                    key=lambda word: abs((word.yc - sequence[-1].yc) - pitch),
                )
            sequence.append(chosen)

        if len(sequence) < MIN_GUTTER_ROWS:
            continue
        gaps = [
            right.yc - left.yc for left, right in zip(sequence, sequence[1:], strict=False)
        ]
        pitch = median(gaps)
        if not 8 <= pitch <= 80:
            continue
        if any(abs(gap - pitch) > PITCH_REL_TOLERANCE * pitch for gap in gaps):
            continue
        if sequence[-1].yc - sequence[0].yc < 200:
            continue
        if max(word.xc for word in sequence) - min(word.xc for word in sequence) > GUTTER_X_TOLERANCE:
            continue
        candidates.append(
            PanelGutter(
                tuple(sequence),
                tuple(word.text for word in sequence),
                pitch,
                "modulo_ten",
            )
        )
    return candidates


def _panel_gutters(words: list[Word]) -> list[PanelGutter]:
    unique: list[PanelGutter] = []
    # Prefer the literal source witness if both representations identify the same gutter.
    for candidate in [*_literal_gutters(words), *_modulo_gutters(words)]:
        if any(
            abs(candidate.x_center - prior.x_center) <= GUTTER_X_TOLERANCE
            and (
                abs(candidate.words[0].yc - prior.words[0].yc) <= prior.pitch
                or (
                    candidate.words[0].yc >= prior.words[0].yc - prior.pitch
                    and candidate.words[-1].yc <= prior.words[-1].yc + prior.pitch
                )
            )
            for prior in unique
        ):
            continue
        unique.append(candidate)
    return sorted(unique, key=lambda gutter: (gutter.words[0].yc, gutter.x_center))


def _record_bbox(record: dict) -> tuple[float, float, float, float] | None:
    bbox = record.get("bbox_2d") or []
    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = map(float, bbox)
    except (TypeError, ValueError):
        return None
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        return None
    return x0, y0, x1, y1


def _next_gutter_x(gutter: PanelGutter, gutters: list[PanelGutter]) -> float:
    candidates = [
        other.x_center
        for other in gutters
        if other.x_center > gutter.x_center
        and abs(other.words[0].yc - gutter.words[0].yc) <= gutter.pitch
    ]
    return min(candidates) if candidates else 1000.0 + GUTTER_BODY_INSET


def _source_row_texts(
    words: list[Word], gutter: PanelGutter, gutters: list[PanelGutter]
) -> dict[int, str]:
    x0 = max(word.x1 for word in gutter.words) + GUTTER_BODY_INSET
    x1 = _next_gutter_x(gutter, gutters) - GUTTER_BODY_INSET
    rows: dict[int, str] = {}
    for ordinal, y_center in gutter.row_y.items():
        members = sorted(
            (
                word
                for word in words
                if x0 < word.xc < x1
                and abs(word.yc - y_center) <= ROW_Y_TOLERANCE * gutter.pitch
            ),
            key=lambda word: word.x0,
        )
        rows[ordinal] = " ".join(word.text for word in members)
    return rows


def _panel_alignment(left: str, right: str) -> float:
    """Treat text-layer glyph fragmentation as exact without weakening content proof."""

    score = _existing_table_alignment(left, right)
    left_compact = "".join(character for character in left.casefold() if character.isalnum())
    right_compact = "".join(character for character in right.casefold() if character.isalnum())
    if left_compact and left_compact == right_compact:
        return 1.0
    compact_similarity = SequenceMatcher(None, left_compact, right_compact).ratio()
    if compact_similarity >= MIN_COMPACT_CHAR_SIMILARITY:
        return compact_similarity
    return score


def _bind_tables(
    records: list[dict], words: list[Word], gutters: list[PanelGutter]
) -> tuple[list[PanelAnchor] | None, str, dict]:
    tables = _existing_line_tables(records)
    if len(tables) != len(gutters):
        return None, "panel_table_count_mismatch", {
            "gutter_candidates": len(gutters),
            "line_tables": len(tables),
        }

    anchors: list[PanelAnchor] = []
    bound_gutters: set[int] = set()
    for table in tables:
        bbox = _record_bbox(records[table.record_index])
        if bbox is None:
            return None, "panel_table_bbox_invalid", {"record_index": table.record_index}
        x0, y0, _x1, y1 = bbox
        candidates = []
        for index, gutter in enumerate(gutters):
            first_y = gutter.words[0].yc
            last_y = gutter.words[-1].yc
            if (
                abs(gutter.x_center - x0) <= GUTTER_X_TOLERANCE
                and y0 - ALIGN_SEARCH_RADIUS_PITCH * gutter.pitch <= first_y
                and y1 + ALIGN_SEARCH_RADIUS_PITCH * gutter.pitch >= last_y
            ):
                candidates.append((index, gutter))
        if len(candidates) != 1:
            return None, "panel_table_grounding_ambiguous", {
                "record_index": table.record_index,
                "candidate_gutters": len(candidates),
            }
        gutter_index, gutter = candidates[0]
        if gutter_index in bound_gutters:
            return None, "duplicate_panel_table_binding", {"gutter_index": gutter_index}

        labels = tuple(row[0].strip() if row else "" for row in table.rows)
        if labels != gutter.labels:
            return None, "panel_line_labels_mismatch", {
                "record_index": table.record_index,
                "model_labels": list(labels),
                "source_labels": list(gutter.labels),
            }
        source_rows = _source_row_texts(words, gutter, gutters)
        if sum(bool(_tokens(text)) for text in source_rows.values()) < MIN_GUTTER_ROWS:
            return None, "insufficient_panel_source_rows", {"record_index": table.record_index}
        similarities = []
        for ordinal, row in enumerate(table.rows, 1):
            body = " ".join(cell.strip() for cell in row[1:] if cell.strip())
            similarity = _panel_alignment(body, source_rows[ordinal])
            if similarity < MIN_TOKEN_F1:
                return None, "weak_panel_source_alignment", {
                    "record_index": table.record_index,
                    "row": ordinal,
                    "similarity": similarity,
                }
            similarities.append(similarity)
        bound_gutters.add(gutter_index)
        anchors.append(
            PanelAnchor(
                gutter,
                table,
                bbox,
                min(similarities),
                sum(similarities) / len(similarities),
            )
        )

    if len(bound_gutters) != len(gutters):
        return None, "incomplete_panel_table_binding", {}
    anchors.sort(key=lambda anchor: (anchor.gutter.words[0].yc, anchor.gutter.x_center))
    return anchors, "ok", {}


def _grid_rows(anchors: list[PanelAnchor]) -> list[list[PanelAnchor]] | None:
    rows: list[list[PanelAnchor]] = []
    for anchor in anchors:
        if not rows:
            rows.append([anchor])
            continue
        reference = median(item.gutter.words[0].yc for item in rows[-1])
        tolerance = max(item.gutter.pitch for item in rows[-1])
        if abs(anchor.gutter.words[0].yc - reference) <= tolerance:
            rows[-1].append(anchor)
        else:
            rows.append([anchor])
    for row in rows:
        row.sort(key=lambda anchor: anchor.gutter.x_center)

    if len(anchors) == 4:
        if [len(row) for row in rows] != [2, 2]:
            return None
        if any(
            abs(rows[0][column].gutter.x_center - rows[1][column].gutter.x_center)
            > GUTTER_X_TOLERANCE
            for column in range(2)
        ):
            return None
    elif len(anchors) == 2:
        if [len(row) for row in rows] not in ([2], [1, 1]):
            return None
        if len(rows) == 2 and (
            abs(rows[0][0].gutter.x_center - rows[1][0].gutter.x_center)
            > GUTTER_X_TOLERANCE
        ):
            return None
    else:
        return None
    return rows


def _x_bounds(rows: list[list[PanelAnchor]]) -> dict[int, tuple[float, float]]:
    bounds: dict[int, tuple[float, float]] = {}
    for row in rows:
        cuts = [
            (left.bbox[2] + right.bbox[0]) / 2.0
            for left, right in zip(row, row[1:], strict=False)
        ]
        edges = [0.0, *cuts, 1000.0]
        for index, anchor in enumerate(row):
            bounds[anchor.table.record_index] = (edges[index], edges[index + 1])
    return bounds


def _header_page_number(
    words: list[Word], anchor: PanelAnchor, x_bounds: tuple[float, float]
) -> tuple[int | None, str | None]:
    gutter = anchor.gutter
    first_y = gutter.words[0].yc
    lower = first_y - PANEL_BAND_PITCHES * gutter.pitch
    upper = first_y - 0.25 * gutter.pitch
    candidates = {
        int(word.text)
        for word in words
        if word.text.isdecimal()
        and x_bounds[0] <= word.xc <= x_bounds[1]
        and lower <= word.yc <= upper
    }
    if not candidates:
        return None, None
    if len(candidates) != 1:
        return None, "ambiguous_panel_page_number"
    return next(iter(candidates)), None


def _panel_order(
    words: list[Word], rows: list[list[PanelAnchor]], bounds: dict[int, tuple[float, float]]
) -> tuple[list[PanelAnchor] | None, str, list[int | None]]:
    visual = [anchor for row in rows for anchor in row]
    values: list[int | None] = []
    for anchor in visual:
        value, error = _header_page_number(words, anchor, bounds[anchor.table.record_index])
        if error:
            return None, error, values
        values.append(value)
    present = [value for value in values if value is not None]
    if not present:
        # The physical grid does not prove the transcript's logical order. Condensed
        # sheets occur in both row-major and column-major order, so native printed page
        # numbers are required before changing model emission order.
        return None, "missing_panel_page_numbers", values
    if len(present) != len(values):
        return None, "partial_panel_page_numbers", values
    if len(set(present)) != len(present):
        return None, "duplicate_panel_page_numbers", values
    ordered_values = sorted(present)
    if ordered_values != list(range(ordered_values[0], ordered_values[0] + len(ordered_values))):
        return None, "nonconsecutive_panel_page_numbers", values
    by_number = sorted(zip(present, visual, strict=True), key=lambda item: item[0])
    return [anchor for _, anchor in by_number], "printed_page_number", values


def _record_panel(
    bbox: tuple[float, float, float, float],
    anchors: list[PanelAnchor],
    bounds: dict[int, tuple[float, float]],
) -> tuple[PanelAnchor | None, str | None]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    y_center = (y0 + y1) / 2.0
    candidates = []
    for anchor in anchors:
        bx0, bx1 = bounds[anchor.table.record_index]
        x_overlap = max(0.0, min(x1, bx1) - max(x0, bx0)) / width
        gutter = anchor.gutter
        if x_overlap < RECORD_PANEL_MIN_X_FRACTION:
            continue
        if not (
            gutter.words[0].yc - PANEL_BAND_PITCHES * gutter.pitch
            <= y_center
            <= gutter.words[-1].yc + PANEL_BAND_PITCHES * gutter.pitch
        ):
            continue
        candidates.append(anchor)
    if not candidates:
        return None, None
    if len(candidates) != 1:
        return None, "record_panel_ambiguous"
    return candidates[0], None


def _panel_record_rank(
    index: int, bbox: tuple[float, float, float, float], anchor: PanelAnchor
) -> int | None:
    if index == anchor.table.record_index:
        return 1
    if bbox[3] <= anchor.bbox[1]:
        return 0
    if bbox[1] >= anchor.bbox[3]:
        return 2
    return None


def transform(raw: str, words: list[Word]) -> tuple[str, dict]:
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {"status": "abstain", "reason": "invalid_model_json"}
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return raw, {"status": "abstain", "reason": "invalid_model_schema"}

    gutters = _panel_gutters(words)
    if len(gutters) not in SUPPORTED_PANEL_COUNTS:
        return raw, {
            "status": "abstain",
            "reason": "unsupported_panel_gutter_count",
            "gutter_candidates": len(gutters),
        }
    anchors, reason, detail = _bind_tables(records, words, gutters)
    if anchors is None:
        return raw, {"status": "abstain", "reason": reason, **detail}
    rows = _grid_rows(anchors)
    if rows is None:
        return raw, {"status": "abstain", "reason": "irregular_panel_grid"}
    bounds = _x_bounds(rows)
    ordered_anchors, order_source, page_numbers = _panel_order(words, rows, bounds)
    if ordered_anchors is None:
        return raw, {
            "status": "abstain",
            "reason": order_source,
            "page_numbers": page_numbers,
        }

    table_indices = {anchor.table.record_index for anchor in anchors}
    grouped: dict[int, list[tuple[int, int, dict]]] = {
        anchor.table.record_index: [] for anchor in anchors
    }
    prefix: list[tuple[int, dict]] = []
    suffix: list[tuple[int, dict]] = []
    min_table_y = min(anchor.bbox[1] for anchor in anchors)
    max_table_y = max(anchor.bbox[3] for anchor in anchors)
    for index, record in enumerate(records):
        if index in table_indices:
            anchor = next(anchor for anchor in anchors if anchor.table.record_index == index)
            grouped[anchor.table.record_index].append((1, index, record))
            continue
        bbox = _record_bbox(record)
        if bbox is None:
            return raw, {"status": "abstain", "reason": "record_bbox_invalid", "record_index": index}
        anchor, error = _record_panel(bbox, anchors, bounds)
        if error:
            return raw, {"status": "abstain", "reason": error, "record_index": index}
        if anchor is not None:
            rank = _panel_record_rank(index, bbox, anchor)
            if rank is None:
                return raw, {
                    "status": "abstain",
                    "reason": "panel_interior_non_table_record",
                    "record_index": index,
                }
            grouped[anchor.table.record_index].append((rank, index, record))
        elif bbox[3] <= min_table_y:
            prefix.append((index, record))
        elif bbox[1] >= max_table_y:
            suffix.append((index, record))
        else:
            return raw, {"status": "abstain", "reason": "unbound_interior_record", "record_index": index}

    output_entries = [*prefix]
    for anchor in ordered_anchors:
        output_entries.extend(
            (index, record)
            for _, index, record in sorted(
                grouped[anchor.table.record_index], key=lambda item: (item[0], item[1])
            )
        )
    output_entries.extend(suffix)
    if len(output_entries) != len(records) or len({index for index, _ in output_entries}) != len(records):
        return raw, {"status": "abstain", "reason": "record_conservation_failure"}
    source_order = list(range(len(records)))
    output_order = [index for index, _ in output_entries]
    if output_order == source_order:
        return raw, {
            "status": "abstain",
            "reason": "already_panel_major",
            "panels": len(anchors),
            "order_source": order_source,
        }

    output = [record for _, record in output_entries]
    return json.dumps(output, ensure_ascii=False), {
        "status": "transformed",
        "reason": "source_bound_multipanel_order",
        "panels": len(anchors),
        "order_source": order_source,
        "page_numbers": page_numbers,
        "gutter_evidence": [anchor.gutter.evidence for anchor in ordered_anchors],
        "min_similarity": min(anchor.min_alignment for anchor in anchors),
        "mean_similarity": sum(anchor.mean_alignment for anchor in anchors) / len(anchors),
        "moved_records": sum(index != source for index, source in enumerate(output_order)),
    }
