#!/usr/bin/env python3
"""Fail-closed transcript table serialization over frozen model predictions.

The postprocessor never reads gold, case names, or eval strata.  Its independent
input is page-word geometry (native PDF words in the active MVP; Textract only in
the frozen reference arm). A page fires only when
the words prove one regular, consecutive line-number gutter and every non-empty
source row binds bijectively to verbatim text already emitted by the model.

The resulting JSONL retains the evals2 frozen-row schema so it can be rescored by
``scripts/rescore_line_table.py`` without another model call.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from extract.core.legal_postprocess.concordance import (
    CONSTANTS as CONCORDANCE_CONSTANTS,
)
from extract.core.legal_postprocess.concordance import (
    transform as transform_concordance,
)
from extract.core.legal_postprocess.locator import (
    CONSTANTS as LOCATOR_TABLE_CONSTANTS,
)
from extract.core.legal_postprocess.locator import (
    transform as transform_locator_table,
)
from extract.core.legal_postprocess.locator_pdf_inspector import (
    CONSTANTS as LOCATOR_PDF_INSPECTOR_CONSTANTS,
)
from extract.core.legal_postprocess.locator_pdf_inspector import (
    eligibility as locator_fallback_eligibility,
)
from extract.core.legal_postprocess.locator_pdf_inspector import (
    transform as transform_locator_pdf_inspector,
)
from extract.core.legal_postprocess.markdown import _escaped_split, _record_tables
from extract.core.legal_postprocess.wrapped import (
    CONSTANTS as WRAPPED_CELL_CONSTANTS,
)
from extract.core.legal_postprocess.wrapped import (
    transform as transform_wrapped_cell,
)

MIN_GUTTER_ROWS = 15
GUTTER_X_TOLERANCE = 20.0
PITCH_REL_TOLERANCE = 0.35
ROW_Y_TOLERANCE = 0.55
ALIGN_SEARCH_RADIUS_PITCH = 1.25
MIN_TOKEN_F1 = 0.50

_TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class Gutter:
    words: tuple[Word, ...]
    pitch: float

    @property
    def row_y(self) -> dict[int, float]:
        return {int(word.text): word.yc for word in self.words}


def _tokens(text: str) -> Counter[str]:
    return Counter(t.casefold() for t in _TOKEN_RE.findall(text))


def _token_f1(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def _existing_table_alignment(left: str, right: str) -> float:
    """Allow only whitespace split/merge equivalence beyond ordinary token F1."""
    score = _token_f1(left, right)
    left_compact = "".join(_TOKEN_RE.findall(left.casefold()))
    right_compact = "".join(_TOKEN_RE.findall(right.casefold()))
    if left_compact and left_compact == right_compact:
        return 1.0
    return score


def _load_words(path: Path) -> list[Word]:
    payload = json.loads(path.read_text())
    out = []
    for item in payload.get("words") or []:
        bb = item.get("bb") or []
        if len(bb) != 4:
            continue
        try:
            x0, y0, x1, y1 = map(float, bb)
        except (TypeError, ValueError):
            continue
        if x0 < x1 and y0 < y1:
            out.append(Word(str(item.get("text") or ""), x0, y0, x1, y1))
    return out


def _candidate_gutters(words: Iterable[Word]) -> list[Gutter]:
    digits: dict[int, list[Word]] = {}
    for word in words:
        if word.text.isdecimal() and 1 <= int(word.text) <= 28 and word.xc <= 950:
            digits.setdefault(int(word.text), []).append(word)

    candidates: list[Gutter] = []
    for first in digits.get(1, []):
        seq = [first]
        for number in range(2, 29):
            choices = [
                word
                for word in digits.get(number, [])
                if word.yc > seq[-1].yc and abs(word.xc - first.xc) <= GUTTER_X_TOLERANCE
            ]
            if not choices:
                break
            if len(seq) == 1:
                chosen = min(choices, key=lambda word: word.yc)
            else:
                current_pitch = median(b.yc - a.yc for a, b in zip(seq, seq[1:], strict=False))
                chosen = min(choices, key=lambda word: abs(word.yc - seq[-1].yc - current_pitch))
            seq.append(chosen)

        if len(seq) < MIN_GUTTER_ROWS:
            continue
        gaps = [b.yc - a.yc for a, b in zip(seq, seq[1:], strict=False)]
        pitch = median(gaps)
        if not 8 <= pitch <= 80:
            continue
        if any(abs(gap - pitch) > PITCH_REL_TOLERANCE * pitch for gap in gaps):
            continue
        if seq[-1].yc - seq[0].yc < 200:
            continue
        if max(word.xc for word in seq) - min(word.xc for word in seq) > GUTTER_X_TOLERANCE:
            continue
        candidates.append(Gutter(tuple(seq), pitch))

    # Multiple seeds can rediscover the same physical gutter. Collapse those,
    # but retain disjoint gutters so compressed pages fail closed.
    unique: list[Gutter] = []
    for candidate in sorted(candidates, key=lambda g: (-len(g.words), g.words[0].xc)):
        if any(
            abs(candidate.words[0].xc - prior.words[0].xc) <= GUTTER_X_TOLERANCE
            and abs(candidate.words[0].yc - prior.words[0].yc) <= prior.pitch
            for prior in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _row_groups(words: Iterable[Word], gutter: Gutter) -> dict[int, list[Word]]:
    row_y = gutter.row_y
    gutter_ids = {id(word) for word in gutter.words}
    right_edge = max(word.x1 for word in gutter.words)
    grouped: dict[int, list[Word]] = {number: [] for number in row_y}
    for word in words:
        if id(word) in gutter_ids or word.xc <= right_edge + 12:
            continue
        number = min(row_y, key=lambda row: abs(word.yc - row_y[row]))
        if abs(word.yc - row_y[number]) <= ROW_Y_TOLERANCE * gutter.pitch:
            grouped[number].append(word)
    return {number: sorted(items, key=lambda word: word.x0) for number, items in grouped.items()}


def _row_witness(groups: dict[int, list[Word]]) -> dict[int, str]:
    return {number: " ".join(word.text for word in items) for number, items in groups.items()}


def _bbox(words: Iterable[Word]) -> list[int] | None:
    items = list(words)
    if not items:
        return None
    return [
        int(min(word.x0 for word in items)),
        int(min(word.y0 for word in items)),
        int(max(word.x1 for word in items)),
        int(max(word.y1 for word in items)),
    ]


def _source_table_bbox(gutter: Gutter, groups: dict[int, list[Word]]) -> list[int]:
    words = [*gutter.words, *(word for row in groups.values() for word in row)]
    bbox = _bbox(words)
    assert bbox is not None
    return bbox


@dataclass(frozen=True)
class ExistingLineTable:
    record_index: int
    rows: tuple[tuple[str, ...], ...]
    header_text: str


def _existing_line_tables(records: list[dict]) -> list[ExistingLineTable]:
    """Find renderer-visible or nearly-renderable model line tables.

    The renderer path owns valid GFM. The strict pipe-row fallback admits only a
    pure record of numeric rows and covers a common model defect: omitting the
    delimiter row entirely. It never repairs arbitrary prose or mixed records.
    """
    attempts: list[ExistingLineTable] = []
    for record_index, record in enumerate(records):
        text = str(record.get("text_content") or "")
        tables, outside = _record_tables(text, record_index)
        for table in tables:
            rows = [list(row) for row in table.rows]
            header = list(table.header)
            header_text = ""
            if header and header[0].strip().isdecimal():
                rows.insert(0, header)
            else:
                header_text = " ".join(cell for cell in header if cell).strip()
            numeric = sum(bool(row and row[0].strip().isdecimal()) for row in rows)
            if (
                not outside
                and table.overflow_rows == 0
                and numeric >= MIN_GUTTER_ROWS
                and numeric == len(rows)
            ):
                attempts.append(
                    ExistingLineTable(
                        record_index,
                        tuple(tuple(cell for cell in row) for row in rows),
                        header_text,
                    )
                )

        if tables:
            continue
        source_rows = []
        pure = True
        for line in text.splitlines():
            if not line.strip():
                continue
            if not _PIPE_ROW_RE.match(line):
                pure = False
                break
            cells = _escaped_split(line)
            if not cells or not cells[0].strip().isdecimal():
                pure = False
                break
            source_rows.append(tuple(cells))
        if pure and len(source_rows) >= MIN_GUTTER_ROWS:
            attempts.append(ExistingLineTable(record_index, tuple(source_rows), ""))
    return attempts


def _line_centers(record: dict) -> list[tuple[str, float]] | None:
    bb = record.get("bbox_2d") or []
    if len(bb) != 4:
        return None
    try:
        x0, y0, x1, y1 = map(float, bb)
    except (TypeError, ValueError):
        return None
    if not (x0 < x1 and y0 < y1):
        return None
    lines = str(record.get("text_content") or "").split("\n")
    if not lines:
        return None
    height = (y1 - y0) / len(lines)
    return [(line.strip(), y0 + (index + 0.5) * height) for index, line in enumerate(lines)]


def _model_output_words(raw: str) -> list[Word] | None:
    """Project v99 records into word-like row evidence without another source.

    This arm intentionally does not infer missing gutter digits.  Only standalone
    decimal model lines can establish the numbered gutter, so pages where v99 did
    not expose that structure abstain.  Body lines retain their model record's
    horizontal extent and receive equal vertical slices, matching ``_line_centers``.
    """
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return None

    words: list[Word] = []
    for record in records:
        centers = _line_centers(record)
        if centers is None:
            continue
        x0, y0, x1, y1 = map(float, record["bbox_2d"])
        line_height = (y1 - y0) / len(centers)
        for index, (text, _center) in enumerate(centers):
            if not text:
                continue
            line_y0 = y0 + index * line_height
            words.append(Word(text, x0, line_y0, x1, line_y0 + line_height))
    return words


def transform_model_only(raw: str) -> tuple[str, dict]:
    words = _model_output_words(raw)
    if words is None:
        return raw, {"status": "abstain", "reason": "invalid_model_json"}
    transformed, report = transform_raw(raw, words)
    report["evidence_source"] = "model_output_only"
    return transformed, report


def _remove_source_absent_numeric_affix(
    text: str, number: int, source_text: str
) -> tuple[str, str | None]:
    """Remove a source-disproven edge structural number from body text.

    The independently observed gutter has already established ``number``.  A
    matching prefix is moved into column one.  A different leading decimal is
    removed only when its removal STRICTLY improves agreement with the source
    row (v99 sometimes numbers logical utterances rather than printed lines).
    A genuine numeric sentence start remains because the source row contains it.
    """
    candidates = []
    prefix = re.match(r"^(\d{1,2})(?:[ \t]+)(\S.*)$", text)
    if prefix:
        candidates.append((prefix.group(2), prefix.group(1)))
    suffix = re.match(r"^(.*\S)(?:[ \t]+)(\d{1,2})$", text)
    if suffix:
        candidates.append((suffix.group(1), suffix.group(2)))
    for body, token in candidates:
        if int(token) == number:
            return body, "moved_to_gutter"
    original_score = _token_f1(text, source_text)
    for body, _token in candidates:
        if _token_f1(body, source_text) > original_score:
            return body, "source_rejected"
    return text, None


def _canonicalize_existing_table(
    raw: str,
    records: list[dict],
    words: list[Word],
    gutter: Gutter,
    groups: dict[int, list[Word]],
    witness: dict[int, str],
    attempt: ExistingLineTable,
) -> tuple[str, dict]:
    assignments: dict[int, str] = {}
    similarities: list[float] = []
    lexical_numbers: list[int] = []
    for row in attempt.rows:
        if not row or not row[0].strip().isdecimal():
            return raw, {"status": "abstain", "reason": "non_numeric_existing_row"}
        number = int(row[0].strip())
        if number not in gutter.row_y or number in assignments:
            return raw, {"status": "abstain", "reason": "existing_row_off_gutter"}
        body = " ".join(cell.strip() for cell in row[1:] if cell.strip())
        similarity = _existing_table_alignment(body, witness.get(number, ""))
        if similarity < MIN_TOKEN_F1:
            return raw, {
                "status": "abstain",
                "reason": "weak_existing_table_alignment",
                "row": number,
                "similarity": similarity,
            }
        assignments[number] = body
        lexical_numbers.append(number)
        similarities.append(similarity)

    if lexical_numbers != list(range(1, max(lexical_numbers) + 1)):
        return raw, {"status": "abstain", "reason": "non_consecutive_existing_rows"}
    nonempty_source_rows = {number for number, text in witness.items() if _tokens(text)}
    if not nonempty_source_rows.issubset(assignments):
        return raw, {
            "status": "abstain",
            "reason": "unbound_existing_source_rows",
            "missing_rows": sorted(nonempty_source_rows - set(assignments)),
        }

    header_record = None
    if attempt.header_text:
        record_bbox = records[attempt.record_index].get("bbox_2d") or []
        if len(record_bbox) != 4:
            return raw, {"status": "abstain", "reason": "existing_header_bbox_unavailable"}
        x0, y0, x1, _y1 = map(float, record_bbox)
        header_words = [
            word
            for word in words
            if x0 <= word.xc <= x1 and y0 <= word.yc < gutter.words[0].yc - 0.5 * gutter.pitch
        ]
        header_text = " ".join(
            word.text for word in sorted(header_words, key=lambda w: (w.yc, w.x0))
        )
        if _token_f1(attempt.header_text, header_text) < MIN_TOKEN_F1:
            return raw, {"status": "abstain", "reason": "weak_existing_header_alignment"}
        header_bbox = _bbox(header_words)
        if header_bbox is None:
            return raw, {"status": "abstain", "reason": "existing_header_bbox_unavailable"}
        header_record = {"bbox_2d": header_bbox, "text_content": attempt.header_text}

    def escape_cell(text: str) -> str:
        return text.replace("|", "\\|")

    last_row = max(gutter.row_y)
    gfm = "|  |  |\n|---|---|\n" + "\n".join(
        f"| {number} | {escape_cell(assignments.get(number, ''))} |"
        for number in range(1, last_row + 1)
    )
    table = {"bbox_2d": _source_table_bbox(gutter, groups), "text_content": gfm}
    output = []
    for index, record in enumerate(records):
        if index == attempt.record_index:
            if header_record is not None:
                output.append(header_record)
            output.append(table)
        else:
            output.append(record)
    return json.dumps(output, ensure_ascii=False), {
        "status": "transformed",
        "reason": "canonicalized_source_bound_line_table",
        "rows": last_row,
        "bound_rows": len(assignments),
        "preserved_header": bool(header_record),
        "min_similarity": min(similarities),
        "mean_similarity": sum(similarities) / len(similarities),
    }


def transform_raw(raw: str, words: list[Word]) -> tuple[str, dict]:
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {"status": "abstain", "reason": "invalid_model_json"}
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return raw, {"status": "abstain", "reason": "invalid_model_schema"}
    gutters = _candidate_gutters(words)
    if len(gutters) != 1:
        return raw, {
            "status": "abstain",
            "reason": "no_single_gutter" if not gutters else "multiple_gutters",
            "gutter_candidates": len(gutters),
        }
    gutter = gutters[0]
    row_y = gutter.row_y
    groups = _row_groups(words, gutter)
    witness = _row_witness(groups)
    existing = _existing_line_tables(records)
    if len(existing) > 1:
        return raw, {"status": "abstain", "reason": "multiple_existing_line_tables"}
    if existing:
        return _canonicalize_existing_table(
            raw, records, words, gutter, groups, witness, existing[0]
        )
    band_lo = min(row_y.values()) - ROW_Y_TOLERANCE * gutter.pitch
    band_hi = max(row_y.values()) + ROW_Y_TOLERANCE * gutter.pitch
    gutter_right = max(word.x1 for word in gutter.words)

    assignments: dict[int, tuple[int, str]] = {}
    absorbed: set[int] = set()
    similarities: list[float] = []
    moved_prefix_rows: set[int] = set()
    rejected_prefix_rows: set[int] = set()
    rejected_prefix_tokens: list[str] = []
    shifted_alignment_rows: set[int] = set()

    # v99 sometimes emits the printed gutter as one record per digit.  Those
    # records are structural duplicates of the table's first column and must be
    # consumed.  Identity is proven by BOTH the visible number and its source
    # gutter position; an arbitrary numeric body record never qualifies.
    model_gutter_rows: dict[int, int] = {}
    for record_index, record in enumerate(records):
        centers = _line_centers(record)
        if centers is None or not centers:
            continue
        if any(not text.isdecimal() or int(text) not in row_y for text, _ in centers):
            continue
        bb = list(map(float, record["bbox_2d"]))
        x_center = (bb[0] + bb[2]) / 2
        matched_numbers = [int(text) for text, _ in centers]
        if len(set(matched_numbers)) != len(matched_numbers):
            continue
        if all(
            abs(x_center - gutter.words[number - 1].xc) <= GUTTER_X_TOLERANCE
            and abs(y_center - row_y[number]) <= ALIGN_SEARCH_RADIUS_PITCH * gutter.pitch
            for number, (_, y_center) in zip(matched_numbers, centers, strict=True)
        ):
            if any(number in model_gutter_rows for number in matched_numbers):
                return raw, {"status": "abstain", "reason": "duplicate_model_gutter_row"}
            for number in matched_numbers:
                model_gutter_rows[number] = record_index
            absorbed.add(record_index)

    for record_index, record in enumerate(records):
        if record_index in absorbed:
            continue
        centers = _line_centers(record)
        if centers is None:
            continue
        bb = list(map(float, record["bbox_2d"]))
        overlap = max(0.0, min(bb[3], band_hi) - max(bb[1], band_lo))
        substantially_in_band = overlap >= 0.5 * (bb[3] - bb[1]) and bb[2] > gutter_right + 12
        if not substantially_in_band:
            continue

        record_rows: list[tuple[int, str, float]] = []
        for text, y_center in centers:
            if not text:
                continue
            nearest = min(row_y, key=lambda row: abs(y_center - row_y[row]))
            candidates = []
            for number in row_y:
                if number in assignments or any(number == row for row, _, _ in record_rows):
                    continue
                distance = abs(y_center - row_y[number]) / gutter.pitch
                if distance > ALIGN_SEARCH_RADIUS_PITCH:
                    continue
                body_text, prefix_action = _remove_source_absent_numeric_affix(
                    text, number, witness.get(number, "")
                )
                similarity = _token_f1(body_text, witness.get(number, ""))
                candidates.append((similarity, -distance, number, body_text, prefix_action))
            if not candidates:
                return raw, {"status": "abstain", "reason": "model_line_off_grid"}
            similarity, _neg_distance, number, body_text, prefix_action = max(candidates)
            if similarity < MIN_TOKEN_F1:
                return raw, {
                    "status": "abstain",
                    "reason": "weak_source_alignment",
                    "row": number,
                    "similarity": similarity,
                }
            record_rows.append((number, body_text, similarity))
            if number != nearest:
                shifted_alignment_rows.add(number)
            if prefix_action == "moved_to_gutter":
                moved_prefix_rows.add(number)
            elif prefix_action == "source_rejected":
                rejected_prefix_rows.add(number)
                before_tokens = Counter(text.split())
                after_tokens = Counter(body_text.split())
                removed = list((before_tokens - after_tokens).elements())
                if len(removed) != 1:
                    return raw, {"status": "abstain", "reason": "numeric_affix_accounting"}
                rejected_prefix_tokens.extend(removed)
        if not record_rows:
            return raw, {"status": "abstain", "reason": "empty_body_record"}
        for number, text, similarity in record_rows:
            assignments[number] = (record_index, text)
            similarities.append(similarity)
        absorbed.add(record_index)

    nonempty_source_rows = {number for number, text in witness.items() if _tokens(text)}
    if not nonempty_source_rows:
        return raw, {"status": "abstain", "reason": "empty_source_rows"}
    if not nonempty_source_rows.issubset(assignments):
        return raw, {
            "status": "abstain",
            "reason": "unbound_source_rows",
            "missing_rows": sorted(nonempty_source_rows - set(assignments)),
        }
    if len(assignments) < MIN_GUTTER_ROWS:
        return raw, {
            "status": "abstain",
            "reason": "insufficient_bound_rows",
            "bound_rows": len(assignments),
            "nonempty_source_rows": len(nonempty_source_rows),
        }

    # The transform is a serialization only: every model line included in the
    # table appears verbatim once, and no source-witness text is substituted.
    before = Counter(
        token
        for index in absorbed - set(model_gutter_rows.values())
        for token in str(records[index].get("text_content") or "").split()
    )
    after = Counter(token for _, text in assignments.values() for token in text.split())
    after.update(str(number) for number in moved_prefix_rows)
    # Source-disproven logical-list prefixes are the only permitted deletion.
    # Account for them explicitly in the conservation certificate.
    after.update(rejected_prefix_tokens)
    if before != after:
        return raw, {"status": "abstain", "reason": "token_conservation_failure"}

    def escape_cell(text: str) -> str:
        return text.replace("|", "\\|")

    last_row = max(row_y)
    gfm = "|  |  |\n|---|---|\n" + "\n".join(
        f"| {number} | {escape_cell(assignments.get(number, (-1, ''))[1])} |"
        for number in range(1, last_row + 1)
    )
    table_bbox = _source_table_bbox(gutter, groups)
    table = {"bbox_2d": table_bbox, "text_content": gfm}
    insert_at = min(absorbed)
    output = []
    for index, record in enumerate(records):
        if index == insert_at:
            output.append(table)
        if index not in absorbed:
            output.append(record)
    return json.dumps(output, ensure_ascii=False), {
        "status": "transformed",
        "reason": "source_bound_single_gutter",
        "rows": last_row,
        "bound_rows": len(assignments),
        "absorbed_records": len(absorbed),
        "absorbed_model_gutter_records": len(model_gutter_rows),
        "moved_embedded_gutter_affixes": len(moved_prefix_rows),
        "source_rejected_numeric_affixes": len(rejected_prefix_rows),
        "text_assisted_shifted_rows": len(shifted_alignment_rows),
        "min_similarity": min(similarities),
        "mean_similarity": sum(similarities) / len(similarities),
    }


def main() -> int:
    # Local import avoids a module cycle: the production multipanel route reuses the
    # single-gutter evidence primitives defined above.
    from extract.core.legal_postprocess.multipanel import CONSTANTS as MULTIPANEL_CONSTANTS
    from extract.core.legal_postprocess.multipanel import transform as transform_multipanel

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", required=True, type=Path)
    parser.add_argument("--words-dir", type=Path)
    parser.add_argument(
        "--locator-evidence-dir",
        type=Path,
        help=(
            "Optional pdf_inspector_positioned_text/v1 directory. The source-backed "
            "locator fallback runs only after the existing model-only locator abstains."
        ),
    )
    parser.add_argument(
        "--evidence-source",
        choices=("external_words", "model_output_only"),
        default="external_words",
    )
    parser.add_argument(
        "--routes",
        choices=(
            "transcript",
            "multipanel",
            "concordance",
            "locator",
            "tables",
            "both",
            "all",
            "wrapped",
        ),
        default="transcript",
        help=(
            "Offline route(s) to apply; 'tables' tries concordance then locator "
            "without rerunning transcript geometry, 'both' preserves the historical "
            "transcript+concordance cascade, 'wrapped' runs only the wrapped-cell "
            "table normalizer, and 'all' tries every route."
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        parser.error("refusing to overwrite --out or --report")
    transcript_enabled = args.routes in {"transcript", "both", "all"}
    multipanel_enabled = args.routes in {"transcript", "multipanel", "both", "all"}
    concordance_enabled = args.routes in {"concordance", "tables", "both", "all"}
    locator_enabled = args.routes in {"locator", "tables", "all"}
    wrapped_enabled = args.routes in {"wrapped", "all"}
    if transcript_enabled and args.evidence_source == "external_words" and args.words_dir is None:
        parser.error("--words-dir is required with --evidence-source external_words")

    counts: Counter[str] = Counter()
    reports = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.per_case.open() as source, args.out.open("w") as destination:
        for line in source:
            row = json.loads(line)
            case_id = row["case_id"]
            if row.get("unavailable"):
                result = {"status": "abstain", "reason": "candidate_unavailable"}
            else:
                original = row["raw"]
                transcript_result = None
                multipanel_result = None
                concordance_result = None
                locator_result = None
                locator_fallback_result = None
                # Native words load once per case; the concordance route's
                # rebuild uses the same evidence source the transcript route
                # already trusts.  The model_output_only arm stays model-only.
                native_words = None
                loaded_words: list[Word] = []
                if args.words_dir is not None and args.evidence_source == "external_words":
                    words_path = args.words_dir / f"{case_id}.json"
                    if words_path.exists():
                        loaded_words = _load_words(words_path)
                        native_words = [(w.text, w.x0, w.y0, w.x1, w.y1) for w in loaded_words]
                if transcript_enabled:
                    if args.evidence_source == "model_output_only":
                        transformed, transcript_result = transform_model_only(original)
                    elif native_words is None:
                        transformed = original
                        transcript_result = {
                            "status": "abstain",
                            "reason": "word_geometry_unavailable",
                        }
                    else:
                        transformed, transcript_result = transform_raw(original, loaded_words)
                        transcript_result["evidence_source"] = "external_words"
                    if transcript_result["status"] == "transformed":
                        if args.routes != "transcript":
                            transcript_result["route"] = "transcript"
                        row["raw"] = transformed
                        result = transcript_result
                    else:
                        row["raw"] = original
                if (
                    multipanel_enabled
                    and (transcript_result is None or transcript_result["status"] != "transformed")
                ):
                    if args.evidence_source != "external_words" or not loaded_words:
                        multipanel_transformed = original
                        multipanel_result = {
                            "status": "abstain",
                            "reason": "word_geometry_unavailable",
                        }
                    else:
                        multipanel_transformed, multipanel_result = transform_multipanel(
                            original, loaded_words
                        )
                    if transcript_result is not None:
                        transcript_result["multipanel_reason"] = multipanel_result["reason"]
                    if multipanel_result["status"] == "transformed":
                        multipanel_result["route"] = "multipanel"
                        row["raw"] = multipanel_transformed
                        result = multipanel_result
                if concordance_enabled and (
                    (transcript_result is None or transcript_result["status"] != "transformed")
                    and (multipanel_result is None or multipanel_result["status"] != "transformed")
                ):
                    transformed, concordance_result = transform_concordance(original, native_words)
                    if concordance_result["status"] == "transformed":
                        row["raw"] = transformed
                        result = concordance_result
                if (
                    locator_enabled
                    and (transcript_result is None or transcript_result["status"] != "transformed")
                    and (multipanel_result is None or multipanel_result["status"] != "transformed")
                    and (
                        concordance_result is None or concordance_result["status"] != "transformed"
                    )
                ):
                    transformed, locator_result = transform_locator_table(original)
                    if locator_result["status"] == "transformed":
                        row["raw"] = transformed
                        result = locator_result
                    elif args.locator_evidence_dir is not None:
                        eligibility = locator_fallback_eligibility(original)
                        evidence_path = args.locator_evidence_dir / f"{case_id}.json"
                        if not eligibility["eligible"]:
                            transformed = original
                            locator_fallback_result = {
                                "status": "abstain",
                                "reason": eligibility["reason"],
                            }
                        elif evidence_path.exists():
                            evidence = json.loads(evidence_path.read_text())
                            transformed, locator_fallback_result = transform_locator_pdf_inspector(
                                original, evidence
                            )
                        else:
                            transformed = original
                            locator_fallback_result = {
                                "status": "abstain",
                                "reason": "locator_source_geometry_unavailable",
                            }
                        if locator_fallback_result["status"] == "transformed":
                            locator_fallback_result["primary_locator_reason"] = locator_result[
                                "reason"
                            ]
                            row["raw"] = transformed
                            result = locator_fallback_result
                        else:
                            locator_result = {
                                **locator_result,
                                "locator_fallback_reason": locator_fallback_result["reason"],
                            }
                wrapped_result = None
                if (
                    wrapped_enabled
                    and (transcript_result is None or transcript_result["status"] != "transformed")
                    and (multipanel_result is None or multipanel_result["status"] != "transformed")
                    and (
                        concordance_result is None or concordance_result["status"] != "transformed"
                    )
                    and (locator_result is None or locator_result["status"] != "transformed")
                    and (
                        locator_fallback_result is None
                        or locator_fallback_result["status"] != "transformed"
                    )
                ):
                    transformed, wrapped_result = transform_wrapped_cell(original, native_words)
                    if wrapped_result["status"] == "transformed":
                        row["raw"] = transformed
                        result = wrapped_result
                if transcript_result is not None and transcript_result["status"] == "transformed":
                    result = transcript_result
                elif multipanel_result is not None and multipanel_result["status"] == "transformed":
                    result = multipanel_result
                elif (
                    concordance_result is not None and concordance_result["status"] == "transformed"
                ):
                    result = concordance_result
                elif (
                    locator_fallback_result is not None
                    and locator_fallback_result["status"] == "transformed"
                ):
                    result = locator_fallback_result
                elif locator_result is not None and locator_result["status"] == "transformed":
                    result = locator_result
                elif wrapped_result is not None and wrapped_result["status"] == "transformed":
                    result = wrapped_result
                elif args.routes == "transcript":
                    result = transcript_result
                elif args.routes == "multipanel":
                    result = multipanel_result
                elif args.routes == "concordance":
                    result = concordance_result
                elif args.routes == "locator":
                    result = locator_result
                elif args.routes == "wrapped":
                    result = wrapped_result
                else:
                    route_results = {
                        "transcript": transcript_result,
                        "multipanel": multipanel_result,
                        "concordance": concordance_result,
                        "locator": locator_result,
                        "wrapped": wrapped_result,
                    }
                    result = {"status": "abstain", "reason": "no_route_accepted"}
                    result.update(
                        {
                            f"{name}_reason": item["reason"]
                            for name, item in route_results.items()
                            if item is not None
                        }
                    )
            counts[f"{result['status']}:{result['reason']}"] += 1
            reports.append({"case_id": case_id, **result})
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.routes == "multipanel":
        report_schema = "transcript_postprocess_prototype/v7"
    elif args.routes == "transcript":
        report_schema = "transcript_postprocess_prototype/v2"
    elif args.routes in {"concordance", "both"}:
        report_schema = "legal_postprocess_prototype/v3"
    elif wrapped_enabled:
        report_schema = "legal_postprocess_prototype/v6"
    elif args.locator_evidence_dir is not None:
        report_schema = "legal_postprocess_prototype/v5"
    else:
        report_schema = "legal_postprocess_prototype/v4"

    report = {
        "schema": report_schema,
        "input": str(args.per_case),
        "evidence_source": args.evidence_source,
        "words_dir": None if args.words_dir is None else str(args.words_dir),
        "locator_evidence_dir": (
            None if args.locator_evidence_dir is None else str(args.locator_evidence_dir)
        ),
        "output": str(args.out),
        "constants": {
            "min_gutter_rows": MIN_GUTTER_ROWS,
            "gutter_x_tolerance": GUTTER_X_TOLERANCE,
            "pitch_rel_tolerance": PITCH_REL_TOLERANCE,
            "row_y_tolerance": ROW_Y_TOLERANCE,
            "alignment_search_radius_pitch": ALIGN_SEARCH_RADIUS_PITCH,
            "min_token_f1": MIN_TOKEN_F1,
            "multipanel": MULTIPANEL_CONSTANTS,
        },
        "counts": dict(sorted(counts.items())),
        "per_case": reports,
    }
    if args.routes != "transcript":
        report["routes"] = args.routes
        if concordance_enabled:
            report["constants"]["concordance"] = CONCORDANCE_CONSTANTS
        if locator_enabled:
            report["constants"]["locator_table"] = LOCATOR_TABLE_CONSTANTS
            if args.locator_evidence_dir is not None:
                report["constants"]["locator_pdf_inspector_fallback"] = (
                    LOCATOR_PDF_INSPECTOR_CONSTANTS
                )
        if wrapped_enabled:
            report["constants"]["wrapped_cell"] = WRAPPED_CELL_CONSTANTS
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
