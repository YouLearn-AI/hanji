"""Fail-closed wrapped-cell table normalizer (Plan 132 route, 2026-08-09).

REGRESSION_ROOTCAUSE_DEEP item 0b: a model that transcribes printed
line-wraps literally emits a markdown row spread over several physical lines —
only the first line carries the row's leading pipes and only the last closes
the final cell (the p0165 docket exhibit: 47 physical lines for a 5-row
table).  Such a row is unparseable as GFM, so TEDS collapses while the token
content stays intact.

The route joins a row's continuation lines back into their cell, and fires
only when EVERY gate holds:

* the candidate record is a pure table: its first line opens a pipe row, its
  last line closes one, and every line is a pipe row, the delimiter row, or a
  continuation of the row still open above it;
* exactly one delimiter row, directly under the header row (GFM shape);
* after joining, every row parses to the delimiter's cell count (trailing
  ALL-EMPTY overflow cells are trimmed — they carry no tokens);
* native-word geometry proves each continuation line is a printed wrap: the
  line's words match a native printed segment lying inside the record's box
  and indented deep into the table, where a genuine new row would start at
  the table's left column instead;
* token conservation is exact — joining is whitespace-only.

The route never fabricates cell text, link markup, or rows, never touches any
other record, and abstains byte-identically everywhere a gate fails.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from extract.core.legal_postprocess.concordance import _native_segments
from extract.core.legal_postprocess.markdown import _escaped_split

MIN_CONTINUATION_MATCH_F1 = 0.60
MIN_WRAP_INDENT_FRACTION = 0.15
RECORD_BAND_TOLERANCE = 25.0

_DELIMITER_ROW = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)


def _tokens(text: str) -> Counter[str]:
    return Counter(t.casefold() for t in _TOKEN_RE.findall(text))


def _token_f1(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 1.0
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def _content_bag(text: str) -> Counter[str]:
    """Whitespace token bag with the pipe scaffolding removed.

    Joining physical lines and trimming EMPTY overflow cells may only ever
    move pipes and whitespace; every content token must survive exactly.
    """
    return Counter(token for token in text.replace("|", " ").split())


def _parse_rows(text: str):
    """Group the record's physical lines into pipe rows with continuations.

    Returns ``(rows, error)``.  Each row is ``{"lines": [...], "delimiter":
    bool}``; a row stays open until a line ends with ``|``.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].lstrip().startswith("|"):
        return None, "no_wrapped_table_candidate"
    rows: list[dict] = []
    open_row: dict | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            if open_row is not None:
                # A new pipe row may not begin while the previous row is
                # still unclosed: the reading would be ambiguous.
                return None, "unclosed_row"
            row = {"lines": [stripped], "delimiter": bool(_DELIMITER_ROW.match(stripped))}
            if stripped.endswith("|"):
                rows.append(row)
            else:
                open_row = row
        else:
            if open_row is None:
                return None, "continuation_outside_row"
            open_row["lines"].append(stripped)
            if stripped.endswith("|"):
                rows.append(open_row)
                open_row = None
    if open_row is not None:
        return None, "unclosed_row"
    return rows, None


def _candidate_records(records: list[dict]) -> list[int]:
    out = []
    for index, record in enumerate(records):
        text = record.get("text_content")
        if not isinstance(text, str):
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 3 or not lines[0].lstrip().startswith("|"):
            continue
        if not lines[-1].rstrip().endswith("|"):
            continue
        if not any(not line.lstrip().startswith("|") for line in lines):
            continue
        out.append(index)
    return out


def transform(raw: str, native_words: list[tuple] | None = None) -> tuple[str, dict]:
    """Return ``(raw_or_transformed, route_receipt)``; every failure abstains."""
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw, {"status": "abstain", "reason": "invalid_model_json"}
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return raw, {"status": "abstain", "reason": "invalid_model_schema"}

    candidates = _candidate_records(records)
    if not candidates:
        return raw, {"status": "abstain", "reason": "no_wrapped_table_candidate"}
    if len(candidates) > 1:
        return raw, {"status": "abstain", "reason": "multiple_wrapped_table_candidates"}
    if native_words is None:
        return raw, {"status": "abstain", "reason": "native_words_unavailable"}
    record_index = candidates[0]
    record = records[record_index]
    text = str(record["text_content"])

    bbox = record.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return raw, {"status": "abstain", "reason": "candidate_bbox_unavailable"}
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return raw, {"status": "abstain", "reason": "candidate_bbox_unavailable"}
    if not (x0 < x1 and y0 < y1):
        return raw, {"status": "abstain", "reason": "candidate_bbox_unavailable"}

    rows, error = _parse_rows(text)
    if error is not None:
        return raw, {"status": "abstain", "reason": error}
    delimiters = [index for index, row in enumerate(rows) if row["delimiter"]]
    if delimiters != [1]:
        return raw, {"status": "abstain", "reason": "no_delimiter_row"}
    ncol = len(_escaped_split(rows[1]["lines"][0]))

    # Native geometry witness: every continuation line must match a printed
    # segment inside the record's box, indented past the row-start margin —
    # where a genuine (pipe-dropped) NEW ROW would print at the table's left
    # column and must abstain.
    segments = [
        segment
        for segment in _native_segments(native_words)
        if (
            y0 - RECORD_BAND_TOLERANCE <= segment.yc <= y1 + RECORD_BAND_TOLERANCE
            and segment.bbox[0] >= x0 - RECORD_BAND_TOLERANCE
            and segment.bbox[2] <= x1 + RECORD_BAND_TOLERANCE
        )
    ]
    segment_tokens = [(segment, _tokens(segment.text)) for segment in segments]
    min_indent = MIN_WRAP_INDENT_FRACTION * (x1 - x0)
    joined_continuations = 0
    weakest_match = 1.0
    smallest_indent = None
    for row in rows:
        for line in row["lines"][1:]:
            content = line.rstrip("|").strip() if line.endswith("|") else line
            line_tokens = _tokens(content)
            if not line_tokens:
                if content.strip():
                    # A continuation carrying only non-word marks (a stray
                    # separator, dashes) has no matchable witness, so its
                    # wrap can never be geometry-proven.
                    return raw, {
                        "status": "abstain",
                        "reason": "wrap_geometry_unproven",
                        "unmatched_line": content[:80],
                        "match_f1": 0.0,
                    }
                continue
            # Best native witness by token F1, with exact printed-text equality
            # as the tie-breaker: the same tokens can print both as a leading
            # column's cell and as a wrap fragment ("09/30/2014" the date vs
            # "09/30/2014)" the wrapped sentence tail), and only the raw text
            # separates them.
            line_norm = " ".join(content.casefold().split())
            best_key = (0.0, 0)
            best_segments: list = []
            for candidate_segment, seg_tokens in segment_tokens:
                f1 = _token_f1(line_tokens, seg_tokens)
                exact = int(" ".join(candidate_segment.text.casefold().split()) == line_norm)
                key = (f1, exact)
                if key > best_key:
                    best_key, best_segments = key, [candidate_segment]
                elif key == best_key:
                    best_segments.append(candidate_segment)
            score = best_key[0]
            if not best_segments or score < MIN_CONTINUATION_MATCH_F1:
                return raw, {
                    "status": "abstain",
                    "reason": "wrap_geometry_unproven",
                    "unmatched_line": content[:80],
                    "match_f1": round(score, 4),
                }
            # EVERY equally-best witness must be printed at wrap depth; if the
            # line could equally be a row-leading cell, geometry is ambiguous.
            indent = min(segment.bbox[0] - x0 for segment in best_segments)
            if indent < min_indent:
                return raw, {
                    "status": "abstain",
                    "reason": "wrap_geometry_unproven",
                    "unindented_line": content[:80],
                    "indent": round(indent, 1),
                    "min_indent": round(min_indent, 1),
                }
            weakest_match = min(weakest_match, score)
            smallest_indent = indent if smallest_indent is None else min(smallest_indent, indent)
            joined_continuations += 1
    if not joined_continuations:
        return raw, {"status": "abstain", "reason": "no_wrapped_table_candidate"}

    trimmed_cells = 0
    joined_rows: list[str] = []
    for index, row in enumerate(rows):
        joined = " ".join(row["lines"])
        if index == 1:
            joined_rows.append(joined)
            continue
        cells = _escaped_split(joined)
        if len(cells) > ncol:
            overflow = cells[ncol:]
            if any(cell.strip() for cell in overflow):
                return raw, {
                    "status": "abstain",
                    "reason": "cell_count_mismatch",
                    "row": index,
                    "cells": len(cells),
                    "expected": ncol,
                }
            trimmed_cells += len(overflow)
            cells = cells[:ncol]
            joined = "| " + " | ".join(cell.strip() for cell in cells) + " |"
        elif len(cells) < ncol:
            return raw, {
                "status": "abstain",
                "reason": "cell_count_mismatch",
                "row": index,
                "cells": len(cells),
                "expected": ncol,
            }
        joined_rows.append(joined)

    normalized = "\n".join(joined_rows)
    if _content_bag(normalized) != _content_bag(text):
        return raw, {"status": "abstain", "reason": "wrap_token_conservation_failure"}

    output = [dict(item) for item in records]
    output[record_index]["text_content"] = normalized
    return json.dumps(output, ensure_ascii=False), {
        "status": "transformed",
        "reason": "cell_internal_wraps_joined",
        "route": "wrapped_cell_table",
        # Every non-delimiter table row, INCLUDING the GFM header row — in
        # this family the header carries the page's leading orphan
        # continuation, so it is content, not furniture.
        "rows": len(joined_rows) - 1,
        "columns": ncol,
        "joined_continuation_lines": joined_continuations,
        "trimmed_empty_overflow_cells": trimmed_cells,
        "weakest_continuation_match_f1": round(weakest_match, 4),
        "smallest_continuation_indent": round(smallest_indent, 1),
    }


CONSTANTS = {
    "min_continuation_match_f1": MIN_CONTINUATION_MATCH_F1,
    "min_wrap_indent_fraction": MIN_WRAP_INDENT_FRACTION,
    "record_band_tolerance": RECORD_BAND_TOLERANCE,
}
