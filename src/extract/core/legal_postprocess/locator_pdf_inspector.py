"""Fail-closed source-geometry fallback for TOC/TOA locator tables.

The source layer proves layout and supplies the whole-table/title geometry. All emitted
text comes verbatim from model records. The existing model-only locator route must run
first; this module is eligible only after that route abstains.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

from extract.core.legal_postprocess.locator import render_bound_table
from extract.core.legal_postprocess.markdown import _record_tables

MIN_ENTRIES = 6
MIN_LEADER_ENTRIES = 6
MIN_ENTRY_TOKEN_F1 = 0.65
LINE_CENTER_TOLERANCE_HEIGHT = 0.60
SECTION_GAP_HEIGHT = 1.50

_TITLE = re.compile(
    r"^\s*TABLE\s+OF\s+(CONTENTS|AUTHORITIES)(?:\s*\((?:CONT(?:INUE)?D|CONT['’]D)\))?\s*$",
    re.IGNORECASE,
)
_LEADER = re.compile(r"(?:\.\s*){5,}|…{2,}")
_SOURCE_LEADER = re.compile(r"(?:\.\s*){2,}|…{2,}")
_LOCATOR = re.compile(
    r"(?:passim|[A-Za-z]{1,3}-?\d+(?:[-–]\d+)?|\d+(?:[-–]\d+)?|[ivxlcdm]+)"
    r"(?:\s*,\s*(?:passim|[A-Za-z]{1,3}-?\d+(?:[-–]\d+)?|\d+(?:[-–]\d+)?|[ivxlcdm]+))*$",
    re.IGNORECASE,
)
_PAGE_HEADER = re.compile(r"pages?(?:\(s\))?", re.IGNORECASE)
_TOKEN = re.compile(r"[\w§$]+(?:['’.-][\w]+)*", re.UNICODE)
_ROMAN_PREFIX = re.compile(r"^(?:[IVXLCDM]+|[A-Z]|\d+)\.?$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceItem:
    text: str
    bbox: tuple[float, float, float, float]

    @property
    def yc(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class SourceLine:
    text: str
    bbox: tuple[float, float, float, float]
    items: tuple[SourceItem, ...]

    @property
    def yc(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def text_height(self) -> float:
        return median(item.height for item in self.items)


@dataclass(frozen=True)
class SourceEntry:
    left: str
    locator: str
    bbox: tuple[float, float, float, float]
    leader: bool
    section: SourceLine | None


@dataclass(frozen=True)
class ModelEntry:
    record_index: int
    left: str
    locator: str


def _space(text: str) -> str:
    return " ".join(text.split())


def _tokens(text: str) -> Counter[str]:
    return Counter(token.casefold() for token in _TOKEN.findall(_LEADER.sub(" ", text)))


def _token_f1(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    overlap = sum((a & b).values())
    if not a or not b or not overlap:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def _locator_key(text: str) -> str:
    return re.sub(r"[\s–-]+", "", text).casefold()


def _section_key(text: str) -> str:
    # Small-caps PDFs may expose ``CASES`` as separate ``C`` / ``ASES`` spans.
    return re.sub(r"[^\w]+", "", text).casefold()


def _union(boxes) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _source_items(evidence: dict) -> list[SourceItem] | None:
    if evidence.get("schema") != "pdf_inspector_positioned_text/v1":
        return None
    items: list[SourceItem] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for value in evidence.get("items", []):
        if not isinstance(value, dict):
            return None
        text = str(value.get("text") or "")
        box = value.get("bbox_2d")
        if not text.strip() or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            bbox = tuple(float(item) for item in box)
        except (TypeError, ValueError):
            return None
        if not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000):
            return None
        key = (_space(text), tuple(round(item, 3) for item in bbox))
        if key in seen:
            continue
        seen.add(key)
        items.append(SourceItem(text, bbox))
    return items


def _lines(items: list[SourceItem]) -> list[SourceLine]:
    groups: list[list[SourceItem]] = []
    for item in sorted(items, key=lambda value: (value.yc, value.bbox[0])):
        if groups:
            center = sum(value.yc for value in groups[-1]) / len(groups[-1])
            height = max([item.height, *(value.height for value in groups[-1])])
            if abs(item.yc - center) <= LINE_CENTER_TOLERANCE_HEIGHT * height:
                groups[-1].append(item)
                continue
        groups.append([item])
    lines = []
    for group in groups:
        ordered = sorted(group, key=lambda value: value.bbox[0])
        lines.append(
            SourceLine(
                _space(" ".join(value.text.strip() for value in ordered)),
                _union(value.bbox for value in ordered),
                tuple(ordered),
            )
        )
    return lines


def _split_trailing(text: str, *, require_source_leader: bool = False) -> tuple[str, str] | None:
    stripped = text.strip()
    match = _LOCATOR.search(stripped)
    if match is None or match.start() == 0:
        return None
    prefix = stripped[: match.start()]
    if require_source_leader and _SOURCE_LEADER.search(prefix) is None:
        return None
    if not stripped[match.start() - 1].isspace() and stripped[match.start() - 1] not in ".…":
        return None
    left = prefix.strip()
    locator = match.group(0).strip()
    left = _LEADER.sub(" ", left).rstrip(" .…")
    return (left.strip(), locator) if left.strip() else None


def _is_section(line: SourceLine, next_line: SourceLine) -> bool:
    tokens = _TOKEN.findall(line.text)
    pitch = next_line.yc - line.yc
    if not 1 <= len(tokens) <= 8 or pitch <= SECTION_GAP_HEIGHT * max(
        line.text_height, next_line.text_height
    ):
        return False
    if _split_trailing(line.text, require_source_leader=True) is not None or _LEADER.search(
        line.text
    ):
        return False
    if len(tokens) == 1 and _ROMAN_PREFIX.fullmatch(tokens[0]):
        return False
    return bool(re.match(r"^[A-Za-z]", line.text)) and not line.text.rstrip().endswith((",", ";"))


def _source_structure(lines: list[SourceLine]):
    title_lines = [line for line in lines if _TITLE.fullmatch(line.text)]
    if len(title_lines) != 1:
        return None, "no_unambiguous_source_locator_title"
    title = title_lines[0]
    start = lines.index(title) + 1
    header: SourceLine | None = None
    pending: list[SourceLine] = []
    pending_section: SourceLine | None = None
    entries: list[SourceEntry] = []
    for offset, line in enumerate(lines[start:], start=start):
        if _PAGE_HEADER.fullmatch(line.text):
            if header is not None:
                return None, "ambiguous_source_page_header"
            header = line
            continue
        next_line = lines[offset + 1] if offset + 1 < len(lines) else None
        if not pending and next_line is not None and _is_section(line, next_line):
            if pending_section is not None:
                return None, "adjacent_source_sections"
            pending_section = line
            continue
        split = _split_trailing(line.text, require_source_leader=True)
        if split is None:
            pending.append(line)
            continue
        pending.append(line)
        left_tail, locator = split
        source_left = _space(" ".join([part.text for part in pending[:-1]] + [left_tail]))
        bbox = _union(part.bbox for part in pending)
        entries.append(
            SourceEntry(
                source_left,
                locator,
                bbox,
                any(_LEADER.search(part.text) for part in pending),
                pending_section,
            )
        )
        pending = []
        pending_section = None
    if pending_section is not None:
        return None, "incomplete_source_entry_run"
    # Footer material after the complete run is allowed only when it has no leader and
    # no trailing locator; otherwise the source grouping is incomplete.
    if pending and any(
        _LEADER.search(line.text) or _split_trailing(line.text, require_source_leader=True)
        for line in pending
    ):
        return None, "incomplete_source_entry_run"
    if len(entries) < MIN_ENTRIES:
        return None, "insufficient_source_locator_entries"
    if sum(entry.leader for entry in entries) < MIN_LEADER_ENTRIES:
        return None, "insufficient_source_leader_evidence"
    values = [(_space(entry.left).casefold(), _locator_key(entry.locator)) for entry in entries]
    if len(values) != len(set(values)):
        return None, "duplicate_source_locator_entries"
    return (title, header, entries), None


def _model_entries(records: list[dict], *, after_index: int) -> list[ModelEntry]:
    entries = []
    for index, record in enumerate(records):
        if index <= after_index:
            continue
        text = record.get("text_content")
        if not isinstance(text, str):
            continue
        split = _split_trailing(text)
        if split is not None:
            entries.append(ModelEntry(index, *split))
    return entries


def _model_title(records: list[dict]) -> tuple[str, int] | None:
    found = []
    for index, record in enumerate(records):
        match = _TITLE.fullmatch(str(record.get("text_content") or ""))
        if match:
            found.append((match.group(1).casefold(), index))
    return found[0] if len(found) == 1 else None


def _visible_tables(records: list[dict]) -> int:
    return sum(
        len(_record_tables(str(record.get("text_content") or ""), index)[0])
        for index, record in enumerate(records)
    )


def eligibility(raw: str) -> dict:
    """Cheap model-only gate to avoid opening a PDF on ineligible pages."""
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"eligible": False, "reason": "invalid_model_json"}
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return {"eligible": False, "reason": "invalid_model_schema"}
    if _model_title(records) is None:
        return {"eligible": False, "reason": "no_unambiguous_model_locator_title"}
    if _visible_tables(records):
        return {"eligible": False, "reason": "already_has_rendered_table"}
    return {"eligible": True, "reason": "standalone_model_locator_title"}


def transform(raw: str, evidence: dict) -> tuple[str, dict]:
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {"status": "abstain", "reason": "invalid_model_json"}
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return raw, {"status": "abstain", "reason": "invalid_model_schema"}
    title_model = _model_title(records)
    if title_model is None:
        return raw, {"status": "abstain", "reason": "no_unambiguous_model_locator_title"}
    family, title_index = title_model
    if _visible_tables(records):
        return raw, {"status": "abstain", "reason": "already_has_rendered_table"}

    items = _source_items(evidence)
    if not items:
        return raw, {"status": "abstain", "reason": "source_geometry_unavailable"}
    structure, source_error = _source_structure(_lines(items))
    if structure is None:
        return raw, {"status": "abstain", "reason": source_error}
    title_source, header_source, source_entries = structure
    source_match = _TITLE.fullmatch(title_source.text)
    if source_match is None or source_match.group(1).casefold() != family:
        return raw, {"status": "abstain", "reason": "model_source_locator_family_mismatch"}

    model_entries = _model_entries(records, after_index=title_index)
    if len(model_entries) != len(source_entries):
        return raw, {
            "status": "abstain",
            "reason": "incomplete_model_source_entry_binding",
            "model_entries": len(model_entries),
            "source_entries": len(source_entries),
        }
    similarities = []
    for model, source in zip(model_entries, source_entries, strict=True):
        comparison_left = model.left
        model_lines = comparison_left.splitlines()
        if (
            source.section is not None
            and model_lines
            and _section_key(model_lines[0]) == _section_key(source.section.text)
        ):
            comparison_left = "\n".join(model_lines[1:]).strip()
        similarity = _token_f1(comparison_left, source.left)
        similarities.append(similarity)
        if _locator_key(model.locator) != _locator_key(source.locator):
            return raw, {"status": "abstain", "reason": "locator_binding_mismatch"}
        if similarity < MIN_ENTRY_TOKEN_F1:
            return raw, {
                "status": "abstain",
                "reason": "weak_model_source_entry_binding",
                "min_similarity": similarity,
            }

    consumed = {entry.record_index for entry in model_entries}
    rows: list[tuple[float, str, str]] = []
    previous_index = title_index
    section_count = 0
    for model, source in zip(model_entries, source_entries, strict=True):
        left = model.left
        if source.section is not None:
            section_text: str | None = None
            model_lines = left.splitlines()
            if model_lines and _section_key(model_lines[0]) == _section_key(source.section.text):
                section_text = model_lines[0]
                left = "\n".join(model_lines[1:]).strip()
            else:
                matches = [
                    index
                    for index in range(previous_index + 1, model.record_index)
                    if index not in consumed
                    and _section_key(str(records[index].get("text_content") or ""))
                    == _section_key(source.section.text)
                ]
                if len(matches) == 1:
                    section_index = matches[0]
                    section_text = str(records[section_index]["text_content"])
                    consumed.add(section_index)
            if section_text is None or not left:
                return raw, {"status": "abstain", "reason": "source_section_not_bound"}
            rows.append((model.record_index - 0.5, section_text, ""))
            section_count += 1
        rows.append((float(model.record_index), left, model.locator))
        previous_index = model.record_index

    header = ["", ""]
    source_boxes = [entry.bbox for entry in source_entries]
    if header_source is not None:
        header_matches = [
            index
            for index, record in enumerate(records)
            if index not in consumed
            and _space(str(record.get("text_content") or "")).casefold()
            == header_source.text.casefold()
        ]
        if len(header_matches) != 1:
            return raw, {"status": "abstain", "reason": "source_header_not_bound"}
        header_index = header_matches[0]
        header[1] = str(records[header_index]["text_content"])
        consumed.add(header_index)
        source_boxes.append(header_source.bbox)

    for entry in source_entries:
        if entry.section is not None:
            source_boxes.append(entry.section.bbox)
    table_box = list(_union(source_boxes))
    title_record = dict(records[title_index])
    title_record["bbox_2d"] = list(title_source.bbox)
    transformed, render_error = render_bound_table(
        records,
        consumed=consumed,
        rows=rows,
        header=header,
        table_box=table_box,
        replacements={title_index: title_record},
    )
    if transformed is None or render_error is not None:
        return raw, {"status": "abstain", "reason": render_error}
    return transformed, {
        "status": "transformed",
        "reason": "source_bound_locator_entries",
        "route": "locator_table",
        "locator_family": family,
        "evidence_source": "pdf_inspector_positioned_text",
        "entries": len(model_entries),
        "leader_entries": sum(entry.leader for entry in source_entries),
        "section_rows": section_count,
        "header": header,
        "absorbed_records": len(consumed),
        "min_similarity": min(similarities),
        "mean_similarity": sum(similarities) / len(similarities),
        "table_bbox": table_box,
        "title_bbox": list(title_source.bbox),
    }


CONSTANTS = {
    "min_entries": MIN_ENTRIES,
    "min_leader_entries": MIN_LEADER_ENTRIES,
    "min_entry_token_f1": MIN_ENTRY_TOKEN_F1,
    "line_center_tolerance_height": LINE_CENTER_TOLERANCE_HEIGHT,
    "section_gap_height": SECTION_GAP_HEIGHT,
}
