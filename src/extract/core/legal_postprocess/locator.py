"""Fail-closed, model-output-only TOC/TOA locator-table serializer.

It reads only parse-model records
and their normalized geometry. A page is transformed only when an explicit,
standalone table-of-contents/authorities title and either a dense dot-leader run or a
family-specific header witness independently prove the two-column entry -> page-locator
structure. Evidence modes are never mixed on a page.
Every abstention returns the input bytes unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from extract.core.legal_postprocess.markdown import _record_tables

MIN_ENTRIES = 6
MAX_ENTRY_X0 = 320.0
MIN_ENTRY_X1 = 700.0
MIN_ENTRY_WIDTH = 450.0
MAX_HEADER_GAP = 90.0
MAX_TITLE_TO_ENTRY_GAP = 180.0
MAX_MEAN_ENTRY_PITCH = 100.0
MAX_ENTRY_GAP = 160.0

_TITLE = re.compile(
    r"^\s*TABLE\s+OF\s+(CONTENTS|AUTHORITIES)(?:\s*\((?:CONT(?:INUE)?D|CONT['’]D)\))?\s*$",
    re.IGNORECASE,
)
_LEADER = re.compile(r"(?:\.\s*){5,}|…{2,}")
_LOCATOR = re.compile(
    r"(?:[A-Za-z]{1,3}-?\d+(?:[-–]\d+)?|\d+(?:[-–]\d+)?|[ivxlcdm]+)"
    r"(?:\s*,\s*(?:[A-Za-z]{1,3}-?\d+(?:[-–]\d+)?|\d+(?:[-–]\d+)?|[ivxlcdm]+))*$",
    re.IGNORECASE,
)
_LEFT_HEADER = re.compile(r"cases?", re.IGNORECASE)
_RIGHT_HEADER = re.compile(r"pages?(?:\(s\))?", re.IGNORECASE)
_SECTION = re.compile(
    r"(?:cases?|statutes?|rules?|regulations?|other\s+authorities)\s*:?\s*",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[\w§$]+(?:['’.-][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class Entry:
    record_index: int
    bbox: tuple[float, float, float, float]
    text: str
    locator: str
    evidence: str

    @property
    def yc(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass(frozen=True)
class AuthorityHeaderWitness:
    indices: tuple[int, int]
    evidence: str


def _bbox(record: dict) -> tuple[float, float, float, float] | None:
    value = record.get("bbox_2d")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
        return None
    return box


def _split_entry(text: str) -> tuple[str, str] | None:
    matches = list(_LEADER.finditer(text))
    if not matches:
        return None
    marker = matches[-1]
    left = text[: marker.start()].strip()
    right = text[marker.end() :].strip()
    if not left or not _LOCATOR.fullmatch(right):
        return None
    return left, right


def _split_header_bound_entry(text: str, *, require_multiline: bool) -> tuple[str, str] | None:
    """Split an entry whose independent page header proves the last token.

    The caller supplies a family-specific header witness. TOA entries additionally
    require multiple source lines; TOC entries may be one line because the standalone
    title, right-aligned ``Page`` header, dense run, and wide geometry independently
    prove the entry/locator relation.
    """
    if require_multiline and "\n" not in text:
        return None
    match = _LOCATOR.search(text.strip())
    if match is None or match.start() == 0:
        return None
    stripped = text.strip()
    if not stripped[match.start() - 1].isspace():
        return None
    left = stripped[: match.start()].strip()
    right = match.group(0).strip()
    if not left or not right:
        return None
    return left, right


def _escape(text: str) -> str:
    return " ".join(text.replace("|", "\\|").splitlines()).strip()


def _bag(text: str) -> Counter[str]:
    return Counter(token.casefold() for token in _TOKEN.findall(_LEADER.sub(" ", text)))


def _visible_tables(records: list[dict]) -> int:
    return sum(
        len(_record_tables(str(record.get("text_content") or ""), index)[0])
        for index, record in enumerate(records)
    )


def _title(records: list[dict]) -> tuple[str, int, tuple[float, float, float, float]] | None:
    found = []
    for index, record in enumerate(records):
        match = _TITLE.fullmatch(str(record.get("text_content") or ""))
        box = _bbox(record)
        if match and box is not None:
            found.append((match.group(1).casefold(), index, box))
    # Repeated titles can delimit independent sections/panels and are ambiguous.
    return found[0] if len(found) == 1 else None


def _paired_authority_headers(
    records: list[dict], title_box: tuple[float, float, float, float]
) -> AuthorityHeaderWitness | None:
    """Return one independently proven TOA header relationship.

    Legal TOAs use both conventional column headers (``Case`` / ``Page(s)`` on
    one baseline) and a vertically stacked form where ``Page(s)`` labels the
    right column and ``Cases:``, ``Statutes:``, etc. starts the first section
    immediately below it. Both forms require the same standalone TOA title;
    ambiguity between multiple candidate relationships remains an abstention.
    """
    left: list[tuple[int, tuple[float, float, float, float]]] = []
    right: list[tuple[int, tuple[float, float, float, float]]] = []
    sections: list[tuple[int, tuple[float, float, float, float]]] = []
    for index, record in enumerate(records):
        text = str(record.get("text_content") or "").strip()
        box = _bbox(record)
        if box is None or box[1] < title_box[3] or box[1] - title_box[3] > MAX_TITLE_TO_ENTRY_GAP:
            continue
        if _LEFT_HEADER.fullmatch(text) and box[0] <= MAX_ENTRY_X0:
            left.append((index, box))
        if _RIGHT_HEADER.fullmatch(text) and box[0] >= 500:
            right.append((index, box))
        if _SECTION.fullmatch(text) and box[0] <= MAX_ENTRY_X0:
            sections.append((index, box))
    candidates = [
        AuthorityHeaderWitness((left_index, right_index), "paired_headers_trailing_locator")
        for left_index, left_box in left
        for right_index, right_box in right
        if left_box[2] < right_box[0]
        and abs((left_box[1] + left_box[3]) - (right_box[1] + right_box[3])) <= 20
    ]
    candidates.extend(
        AuthorityHeaderWitness((section_index, right_index), "section_header_trailing_locator")
        for section_index, section_box in sections
        for right_index, right_box in right
        if section_box[2] < right_box[0] and 0 <= section_box[1] - right_box[3] <= MAX_HEADER_GAP
    )
    return candidates[0] if len(candidates) == 1 else None


def _contents_page_header(
    records: list[dict], title_box: tuple[float, float, float, float]
) -> int | None:
    """Return the unique right-aligned ``Page`` header below a TOC title."""
    found: list[int] = []
    for index, record in enumerate(records):
        text = str(record.get("text_content") or "").strip()
        box = _bbox(record)
        if box is None or box[1] < title_box[3] or box[1] - title_box[3] > MAX_TITLE_TO_ENTRY_GAP:
            continue
        if _RIGHT_HEADER.fullmatch(text) and box[0] >= 500:
            found.append(index)
    return found[0] if len(found) == 1 else None


def _has_repeated_run(entries: list[Entry]) -> bool:
    values = [(_escape(entry.text).casefold(), entry.locator.casefold()) for entry in entries]
    for width in range(MIN_ENTRIES, len(values) // 2 + 1):
        for start in range(0, len(values) - 2 * width + 1):
            if values[start : start + width] == values[start + width : start + 2 * width]:
                return True
    return False


def render_bound_table(
    records: list[dict],
    *,
    consumed: set[int],
    rows: list[tuple[float, str, str]],
    header: list[str],
    table_box: list[float] | None = None,
    replacements: dict[int, dict] | None = None,
) -> tuple[str | None, str | None]:
    """Render one token-conserving table from already-proven model records.

    Both the model-only route and the source-backed experimental fallback use this
    single serializer. ``rows`` may split one model record into a section row followed
    by an entry row; conservation is checked over the consumed records as a whole.
    """
    ordered_rows = sorted(rows, key=lambda item: item[0])
    rendered = [f"| {_escape(header[0])} | {_escape(header[1])} |", "|---|---|"]
    rendered.extend(f"| {_escape(left)} | {_escape(right)} |" for _, left, right in ordered_rows)
    table_text = "\n".join(rendered)

    before = Counter()
    for index in consumed:
        before.update(_bag(str(records[index].get("text_content") or "")))
    if before != _bag(table_text):
        return None, "token_conservation_failure"
    parsed, outside = _record_tables(table_text, 0)
    if len(parsed) != 1 or outside or parsed[0].overflow_rows:
        return None, "renderer_contract_failure"
    visible = " ".join(parsed[0].header + [cell for row in parsed[0].rows for cell in row])
    if before != _bag(visible):
        return None, "renderer_token_conservation_failure"

    if table_box is None:
        boxes = [_bbox(records[index]) for index in consumed]
        if any(box is None for box in boxes):
            return None, "consumed_bbox_unavailable"
        concrete_boxes = [box for box in boxes if box is not None]
        table_box = [
            min(box[0] for box in concrete_boxes),
            min(box[1] for box in concrete_boxes),
            max(box[2] for box in concrete_boxes),
            max(box[3] for box in concrete_boxes),
        ]

    table = {"bbox_2d": table_box, "text_content": table_text}
    output: list[dict] = []
    insert_at = min(consumed)
    replacements = replacements or {}
    for index, record in enumerate(records):
        if index == insert_at:
            output.append(table)
        if index not in consumed:
            output.append(replacements.get(index, record))
    return json.dumps(output, ensure_ascii=False), None


def transform(raw: str) -> tuple[str, dict]:
    """Return ``(raw_or_transformed, route_receipt)``; every failure abstains."""
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {"status": "abstain", "reason": "invalid_model_json"}
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return raw, {"status": "abstain", "reason": "invalid_model_schema"}

    title = _title(records)
    if title is None:
        return raw, {"status": "abstain", "reason": "no_unambiguous_standalone_locator_title"}
    family, _title_index, title_box = title
    if _visible_tables(records):
        return raw, {"status": "abstain", "reason": "already_has_rendered_table"}

    paired_headers = (
        _paired_authority_headers(records, title_box) if family == "authorities" else None
    )
    contents_header = _contents_page_header(records, title_box) if family == "contents" else None
    witness_indices = (
        list(paired_headers.indices)
        if paired_headers is not None
        else [contents_header]
        if contents_header is not None
        else []
    )
    witness_bottom = (
        max(
            (_bbox(records[index]) or (0.0, 0.0, 0.0, title_box[3]))[3] for index in witness_indices
        )
        if witness_indices
        else None
    )
    dot_leader_entries = sum(
        1
        for record in records
        if isinstance(record.get("text_content"), str)
        and _split_entry(record["text_content"]) is not None
        and (box := _bbox(record)) is not None
        and box[0] <= MAX_ENTRY_X0
        and box[2] >= MIN_ENTRY_X1
        and box[2] - box[0] >= MIN_ENTRY_WIDTH
    )
    # Never mix evidence modes. A complete dot-leader run is the stronger witness;
    # header-bound splitting is a checkpoint-compatibility fallback only when that
    # primary run is unavailable.
    allow_header_fallback = dot_leader_entries < MIN_ENTRIES

    entries: list[Entry] = []
    for index, record in enumerate(records):
        text = record.get("text_content")
        box = _bbox(record)
        split = _split_entry(text) if isinstance(text, str) else None
        evidence = "dot_leader"
        if (
            allow_header_fallback
            and split is None
            and isinstance(text, str)
            and box is not None
            and witness_bottom is not None
            and box[1] > witness_bottom
        ):
            if paired_headers is not None:
                split = _split_header_bound_entry(text, require_multiline=True)
                evidence = paired_headers.evidence
            elif contents_header is not None:
                split = _split_header_bound_entry(text, require_multiline=False)
                evidence = "page_header_trailing_locator"
        if box is None or split is None:
            continue
        if box[0] > MAX_ENTRY_X0 or box[2] < MIN_ENTRY_X1 or box[2] - box[0] < MIN_ENTRY_WIDTH:
            continue
        entries.append(Entry(index, box, *split, evidence))
    if len(entries) < MIN_ENTRIES:
        return raw, {
            "status": "abstain",
            "reason": "insufficient_locator_entries",
            "entries": len(entries),
        }
    if _has_repeated_run(entries):
        return raw, {"status": "abstain", "reason": "duplicate_locator_entry_run"}
    if any(right.yc <= left.yc for left, right in zip(entries, entries[1:], strict=False)):
        return raw, {"status": "abstain", "reason": "nonmonotone_entry_geometry"}
    gaps = [right.yc - left.yc for left, right in zip(entries, entries[1:], strict=False)]
    mean_pitch = sum(gaps) / len(gaps)
    if (
        entries[0].bbox[1] <= title_box[3]
        or entries[0].bbox[1] - title_box[3] > MAX_TITLE_TO_ENTRY_GAP
    ):
        return raw, {"status": "abstain", "reason": "title_not_bound_to_entry_run"}
    if mean_pitch > MAX_MEAN_ENTRY_PITCH or max(gaps) > MAX_ENTRY_GAP:
        return raw, {"status": "abstain", "reason": "dispersed_locator_entries"}

    first_y = entries[0].bbox[1]
    last_y = entries[-1].bbox[3]
    consumed = {entry.record_index for entry in entries}
    header = ["", ""]
    section_rows: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        if index in consumed:
            continue
        text = str(record.get("text_content") or "").strip()
        box = _bbox(record)
        if box is None:
            continue
        if first_y - MAX_HEADER_GAP <= box[1] < first_y:
            if _RIGHT_HEADER.fullmatch(text) and not header[1]:
                header[1] = text
                consumed.add(index)
            elif _LEFT_HEADER.fullmatch(text) and text.casefold() == "case" and not header[0]:
                header[0] = text
                consumed.add(index)
            elif _SECTION.fullmatch(text):
                section_rows.append((index, text))
                consumed.add(index)
        elif first_y <= box[1] <= last_y and _SECTION.fullmatch(text):
            section_rows.append((index, text))
            consumed.add(index)

    rows: list[tuple[float, str, str]] = [
        (entry.record_index, entry.text, entry.locator) for entry in entries
    ]
    rows.extend((index, text, "") for index, text in section_rows)
    boxes = [_bbox(records[index]) for index in consumed]
    if any(box is None for box in boxes):
        return raw, {"status": "abstain", "reason": "consumed_bbox_unavailable"}
    concrete_boxes = [box for box in boxes if box is not None]
    table_box = [
        min(box[0] for box in concrete_boxes),
        min(box[1] for box in concrete_boxes),
        max(box[2] for box in concrete_boxes),
        max(box[3] for box in concrete_boxes),
    ]
    transformed, render_error = render_bound_table(
        records,
        consumed=consumed,
        rows=rows,
        header=header,
        table_box=table_box,
    )
    if render_error is not None or transformed is None:
        return raw, {"status": "abstain", "reason": render_error}
    return transformed, {
        "status": "transformed",
        "reason": "explicit_title_dense_locator_entries",
        "route": "locator_table",
        "locator_family": family,
        "entries": len(entries),
        "entry_evidence": dict(Counter(entry.evidence for entry in entries)),
        "section_rows": len(section_rows),
        "header": header,
        "absorbed_records": len(consumed),
        "table_bbox": table_box,
    }


CONSTANTS = {
    "min_entries": MIN_ENTRIES,
    "max_entry_x0": MAX_ENTRY_X0,
    "min_entry_x1": MIN_ENTRY_X1,
    "min_entry_width": MIN_ENTRY_WIDTH,
    "max_header_gap": MAX_HEADER_GAP,
    "max_title_to_entry_gap": MAX_TITLE_TO_ENTRY_GAP,
    "max_mean_entry_pitch": MAX_MEAN_ENTRY_PITCH,
    "max_entry_gap": MAX_ENTRY_GAP,
}
