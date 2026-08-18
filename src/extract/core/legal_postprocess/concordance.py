"""Fail-closed concordance/word-index serializer.

The primary path consumes only the
parse model's records and their normalized 0..1000 geometry.  It never reads
page identity, taxonomy, gold, or a secondary model.

Native-words extension (2026-08-09, REGRESSION_ROOTCAUSE_DEEP item 0a): a
complete model-emitted table is preserved unless its renderer-visible
structure proves a defect: a non-empty header, renderer-dropped cells,
overlapping duplicate tables, a model/native column-count contradiction, a
citation-only continuation cell after a complete entry in the same column, or
a per-column entry-count contradiction (2026-08-11 plan-146 extension: the
column's cell count disagrees with the native lane's entry count AND no
admissible reading of the lane's count-marker evidence can reproduce the
model's cells — ambiguous/degraded marker glyphs always read in the model's
favor, so only a contradiction that survives every classification fires).
Only then is the table rebuilt from native PDF words.  Count-marker-bearing
native lane rows prove the semantic body extent; the model bbox is an overlap
witness, not a crop (it may itself be defective).  A cap-truncated table uses
its salvaged bbox as the same overlap witness.
The rebuild remains fail-closed on family evidence, native lane geometry,
model↔native token agreement, and exact token conservation.  It also backs
the one model-record abstention that is a pure evidence deficit — a real short
printed column with fewer than three count markers — when native words
corroborate the structure.  Without native words every case abstains.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

from extract.core.legal_postprocess import entry_major
from extract.core.legal_postprocess.markdown import (
    _record_tables,
    _salvage_truncated_table_chunk,
    repair_truncated_json,
)

PAGE_WIDTH = 1000.0
MIN_TOTAL_CITATIONS = 75
MIN_LANE_CITATIONS = 20
MIN_LANE_SEPARATION = 120.0
MAX_FURNITURE_WIDTH = 450.0
MAX_LANES = 6
MIN_COUNT_MARKERS = 10
MIN_LANE_COUNT_MARKERS = 3
TILE_SEAM_PITCH_MULTIPLIER = 1.5
MAX_TILE_SEAM_GAP = 16.0
# Native-words rebuild (family-proven model-table discard) constants.
NATIVE_SEGMENT_GAP = 15.0
NATIVE_LINE_Y_TOLERANCE = 0.6
NATIVE_LINE_HEIGHT_TOLERANCE = 0.3
NATIVE_MAX_DEFICIENT_MARKER_LANES = 1
NATIVE_MARKER_COMPONENT_GAP_HEIGHTS = 3.0
MIN_NATIVE_AGREEMENT = 0.50
CONSUME_MIN_CITATIONS = 3
MIN_DUPLICATE_BBOX_OVERLAP = 0.50
MAX_UNSALVAGED_REMAINDER_TOKENS = 8

_CITE = re.compile(r"(?<!\d)\d{1,5}(?::|/)\d{1,4}(?:[,;]\d{1,4})*")
_TABLE_LINE = re.compile(r"(?m)^\s*\|")
_COUNT_MARKER = re.compile(r"[\[(]\d+[\])]")
# Entry-count audit marker evidence (2026-08-11).  The strict form tolerates the
# word-split tokenization some text layers apply to `[ 3 ]`; the two trace forms
# classify a token as an AMBIGUOUS marker candidate when glyph remapping degraded
# either its digits (wrapper glyphs survive) or its wrapper (a single digit run
# survives).  Citation tokens carry several digit runs and never qualify.
_COUNT_MARKER_TOLERANT = re.compile(r"[\[(]\s*\d+\s*[\])]")
_DIGIT_RUN = re.compile(r"\d+")
_MARKER_TRACE_WRAPPED = re.compile(r"^[^\w\s].*[^\w\s]$", re.DOTALL)
_CITE_ONLY_LINE = re.compile(r"^\s*(?:\d{1,5}(?::|/)\d{1,4}[\d,;./:–—-]*\s*)+$")
_LEADING_DASH = re.compile(r"^\s*[-–—]\s+")
_RULE_PREFIX_COUNT = re.compile(r"^(?P<rule>[-–—]{5,}\w?)\s+(?P<body>[\[(]\d+[\])].*)$")
_TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class Fragment:
    record_index: int
    text: str
    bbox: tuple[float, float, float, float]
    citations: int

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def xc(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def yc(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


def _bbox(record: dict) -> tuple[float, float, float, float] | None:
    value = record.get("bbox_2d")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
        return None
    return box


def _citation_count(text: str) -> int:
    return len(_CITE.findall(text))


def _bbox_overlap_fraction(
    candidate: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> float:
    """Return the fraction of ``candidate`` covered by ``container``."""
    intersection_width = max(0.0, min(candidate[2], container[2]) - max(candidate[0], container[0]))
    intersection_height = max(
        0.0, min(candidate[3], container[3]) - max(candidate[1], container[1])
    )
    area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    return intersection_width * intersection_height / area


def _cluster_lanes(anchors: list[Fragment]) -> list[list[Fragment]]:
    """Cluster citation-bearing left edges without page-specific coordinates."""
    groups: list[list[Fragment]] = []
    for fragment in sorted(anchors, key=lambda item: item.x):
        if not groups:
            groups.append([fragment])
            continue
        center = sum(item.x * item.citations for item in groups[-1]) / sum(
            item.citations for item in groups[-1]
        )
        # Records within one printed lane vary slightly with indentation.  A
        # gap large enough to be a distinct lane is resolved after clustering.
        if fragment.x - center < MIN_LANE_SEPARATION:
            groups[-1].append(fragment)
        else:
            groups.append([fragment])
    strong = [
        group for group in groups if sum(item.citations for item in group) >= MIN_LANE_CITATIONS
    ]
    # Never silently discard a weak extra citation lane: it may be a real
    # short printed column, and leaving it outside the table violates the
    # one-table contract. Ambiguity is an abstention, not partial coverage.
    return strong if len(strong) == len(groups) and 2 <= len(strong) <= MAX_LANES else []


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _line_hint(text: str, left_edge: float, lane_center: float) -> bool:
    """Head-vs-continuation geometry hint for one printed line.

    Left-edge geometry resolves the otherwise ambiguous bare numeric-term vs
    citation-continuation case.  A line that is only citation tokens is a
    continuation whatever its geometry; the term-cite separator dash printed
    at a wrap boundary (``- 54:11``) belongs to that class too — no
    concordance style opens an entry with a bare dash followed by cites.
    """
    body = _LEADING_DASH.sub("", text)
    if _CITE_ONLY_LINE.fullmatch(body):
        return False
    return left_edge <= lane_center + 18.0


def _rule_prefix_as_count_attachment(text: str) -> str:
    """Move a text-layer horizontal-rule token behind its count/cites.

    Some born-digital table rules are encoded as a long dash token fused to
    one terminal glyph.  Immediately before a count marker, that decoration
    cannot be a lexical entry head.  Moving it to the fragment suffix makes
    the count marker a continuation signal while preserving the exact token
    bag consumed by the rebuild.
    """
    match = _RULE_PREFIX_COUNT.fullmatch(text)
    if match is None:
        return text
    return f"{match.group('body')} {match.group('rule')}"


def _tile_seam_tolerance(
    previous: Fragment,
    previous_lines: list[str],
    current: Fragment,
    current_lines: list[str],
) -> float:
    """One local line pitch, capped, for exact tile-overlap removal.

    Model checkpoints slice the same printed column into slightly different
    record boxes.  A fixed normalized-point seam threshold is therefore not
    checkpoint-stable.  The shorter record's per-line height is the conservative
    local pitch witness; exact suffix/prefix equality remains independently
    required before any line is removed.
    """
    pitches = [
        (item.bbox[3] - item.bbox[1]) / len(lines)
        for item, lines in (
            (previous, previous_lines),
            (current, current_lines),
        )
        if lines
    ]
    if not pitches:
        return 0.0
    return min(MAX_TILE_SEAM_GAP, TILE_SEAM_PITCH_MULTIPLIER * min(pitches))


def _marker_interval(text: str) -> tuple[int, int]:
    """Count-marker evidence in one model cell / native line segment.

    Returns ``(strict, strict + ambiguous)`` — the interval of marker counts the
    text admits.  Strict matches collapse an immediately-adjacent identical
    duplicate (the fake-bold overprint signature: the same glyphs emitted twice
    at sub-line offset).  Outside strict matches, a token retaining either
    structural trace of a degraded marker — a single digit run (``{3}``, ``[3``,
    bare ``3``, ``I3I``) or wrapper glyphs at both ends (``[E]``) — MAY be a
    marker whose other glyphs the text layer remapped, so it widens the interval
    instead of forcing either reading.
    """
    kept: list[tuple[str, int]] = []
    for match in _COUNT_MARKER_TOLERANT.finditer(text):
        normalized = re.sub(r"\s+", "", match.group())
        if (
            kept
            and kept[-1][0] == normalized
            and not text[kept[-1][1] : match.start()].strip()
        ):
            kept[-1] = (normalized, match.end())
            continue
        kept.append((normalized, match.end()))
    ambiguous = sum(
        1
        for token in _COUNT_MARKER_TOLERANT.sub(" ", text).split()
        if len(_DIGIT_RUN.findall(token)) == 1 or _MARKER_TRACE_WRAPPED.match(token)
    )
    return len(kept), len(kept) + ambiguous


def _marker_partition_feasible(
    cells: tuple[tuple[int, int], ...], segments: tuple[tuple[int, int], ...]
) -> bool:
    """Can the model column be an honest reading of the native lane?

    Tests whether the lane's ordered per-segment marker intervals admit ANY
    partition into consecutive runs, one per model cell in order, whose
    achievable run sums intersect each cell's own interval.  An empty run is
    legal only for a cell that can be marker-free.  Feasibility ties every
    marker discrepancy to a specific entry boundary while granting every
    ambiguous glyph its most model-favorable reading; only a contradiction that
    survives all of that is a proven defect.
    """
    # Bottom-up over (cell_index, segment_index) — iterative, so a degenerate
    # thousand-cell table can never blow the recursion limit into exception
    # telemetry (codex round 2 finding; the fail-closed outcome was unchanged).
    total_segments = len(segments)
    feasible = [False] * (total_segments + 1)
    feasible[total_segments] = True
    for low, high in reversed(cells):
        next_feasible = feasible
        feasible = [False] * (total_segments + 1)
        for segment_index in range(total_segments, -1, -1):
            if low == 0 and next_feasible[segment_index]:
                feasible[segment_index] = True
                continue
            run_low = run_high = 0
            for cursor in range(segment_index, total_segments):
                run_low += segments[cursor][0]
                run_high += segments[cursor][1]
                if run_low > high:
                    break
                if run_high >= low and next_feasible[cursor + 1]:
                    feasible[segment_index] = True
                    break
    return feasible[0]


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


def _native_segments(native_words: list[tuple]) -> list[Fragment]:
    """Native PDF words -> one Fragment per printed line per column.

    Words cluster into printed lines by y-center, refusing words whose glyph
    height is incompatible with the line's — an overprinted furniture stamp in
    a larger font shares a body row's y-band but not its type size, and
    merging it would weld the stamp across every printed column.  Lines split
    into column segments at horizontal gaps no intra-line word spacing
    produces.
    """
    words = []
    seen: set[tuple] = set()
    for item in native_words:
        try:
            text, x0, y0, x1, y1 = str(item[0]), *(float(v) for v in item[1:5])
        except (TypeError, ValueError, IndexError):
            continue
        if not (text and x0 < x1 and y0 < y1 and all(math.isfinite(v) for v in (x0, y0, x1, y1))):
            continue
        # Shadow text layers duplicate every word at (near-)identical
        # coordinates; keeping both would double every rebuilt token while
        # still passing the internal conservation check.
        key = (text, round(x0), round(y0), round(x1), round(y1))
        if key in seen:
            continue
        seen.add(key)
        words.append((text, x0, y0, x1, y1))
    if not words:
        return []
    page_height = median(w[4] - w[2] for w in words)
    lines: list[list[tuple]] = []
    for word in sorted(words, key=lambda w: ((w[2] + w[4]) / 2, w[1])):
        yc = (word[2] + word[4]) / 2
        height = word[4] - word[2]
        placed = False
        for line in reversed(lines[-3:]):
            if (
                abs(yc - median((x[2] + x[4]) / 2 for x in line))
                <= NATIVE_LINE_Y_TOLERANCE * page_height
                and abs(height - median(x[4] - x[2] for x in line))
                <= NATIVE_LINE_HEIGHT_TOLERANCE * page_height
            ):
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    fragments: list[Fragment] = []
    for line in lines:
        line.sort(key=lambda w: w[1])
        segment = [line[0]]
        for word in line[1:]:
            if word[1] - segment[-1][3] > NATIVE_SEGMENT_GAP:
                fragments.append(segment)
                segment = [word]
            else:
                segment.append(word)
        fragments.append(segment)
    return [
        Fragment(
            index,
            " ".join(w[0] for w in segment),
            (
                min(w[1] for w in segment),
                min(w[2] for w in segment),
                max(w[3] for w in segment),
                max(w[4] for w in segment),
            ),
            _citation_count(" ".join(w[0] for w in segment)),
        )
        for index, segment in enumerate(fragments)
    ]


def _rebuild_from_native_words(
    native_words: list[tuple],
    model_evidence_text: str,
    trusted_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[dict | None, dict]:
    """Build the concordance table purely from native PDF words.

    Fail-closed: family-scale citation density, 2..6 strong lanes, count-marker
    evidence (at most one real short printed column below the per-lane floor),
    model↔native token agreement, entry-major regrouping, and exact token
    conservation of the rendered table against every assigned native segment.
    Returns ``(table_record, receipt)``; ``table_record`` is None on abstention.
    """
    fragments = _native_segments(native_words)
    if not fragments:
        return None, {"reason": "native_words_unavailable"}
    anchors = [
        item
        for item in fragments
        if item.citations and item.bbox[2] - item.bbox[0] <= MAX_FURNITURE_WIDTH
    ]
    total_citations = sum(item.citations for item in anchors)
    if total_citations < MIN_TOTAL_CITATIONS:
        return None, {
            "reason": "native_insufficient_citation_density",
            "native_citations": total_citations,
        }
    lanes = _cluster_lanes(anchors)
    if not lanes:
        return None, {"reason": "native_ambiguous_column_lanes"}
    count_markers_by_lane = [
        sum(len(_COUNT_MARKER.findall(item.text)) for item in lane) for lane in lanes
    ]
    deficient = sum(1 for markers in count_markers_by_lane if markers < MIN_LANE_COUNT_MARKERS)
    if (
        sum(count_markers_by_lane) < MIN_COUNT_MARKERS
        or deficient > NATIVE_MAX_DEFICIENT_MARKER_LANES
    ):
        return None, {
            "reason": "native_bare_term_format_unproven",
            "count_markers_by_lane": count_markers_by_lane,
        }
    lane_centers = [
        sum(item.x * item.citations for item in lane) / sum(item.citations for item in lane)
        for lane in lanes
    ]
    if any(
        b - a < MIN_LANE_SEPARATION for a, b in zip(lane_centers, lane_centers[1:], strict=False)
    ):
        return None, {"reason": "native_lanes_not_separated"}

    marker_anchors = [item for item in anchors if _COUNT_MARKER.search(item.text)]
    if not marker_anchors:
        return None, {"reason": "native_marker_body_unavailable"}
    marker_height = median(item.bbox[3] - item.bbox[1] for item in marker_anchors)
    components: list[list[Fragment]] = []
    for item in sorted(marker_anchors, key=lambda fragment: (fragment.bbox[1], fragment.x)):
        if (
            not components
            or item.bbox[1] - max(x.bbox[3] for x in components[-1])
            > NATIVE_MARKER_COMPONENT_GAP_HEIGHTS * marker_height
        ):
            components.append([item])
        else:
            components[-1].append(item)
    if trusted_bbox is None:
        body = [item for component in components for item in component]
    else:
        overlapping = []
        for component in components:
            y0 = min(item.bbox[1] for item in component)
            y1 = max(item.bbox[3] for item in component)
            overlap = max(0.0, min(y1, trusted_bbox[3]) - max(y0, trusted_bbox[1]))
            if overlap:
                overlapping.append(component)
        if not overlapping:
            return None, {"reason": "native_marker_body_bbox_ambiguous"}
        # Section-letter rows can create several vertical gaps inside one word
        # index.  Every count-supported component intersecting the model table
        # is part of its candidate body; marker-free components (such as ECF
        # citation furniture) never enter this set.
        body = [item for component in overlapping for item in component]
    # Count markers authorize semantic expansion beyond a model bbox that can
    # itself be too narrow. Citation-only continuations carry weaker evidence:
    # they may extend the envelope only inside the model's vertical table band.
    # With no model table (the older deficient-lane route), preserve its prior
    # all-anchor extent behavior.
    extent_witnesses = list(body)
    if trusted_bbox is None:
        extent_witnesses.extend(anchors)
    else:
        extent_witnesses.extend(
            item for item in anchors if trusted_bbox[1] <= item.yc <= trusted_bbox[3]
        )
    body_y0 = min(item.bbox[1] for item in extent_witnesses)
    body_y1 = max(item.bbox[3] for item in extent_witnesses)
    assigned: list[list[Fragment]] = [[] for _ in lanes]
    for item in fragments:
        width = item.bbox[2] - item.bbox[0]
        if width > MAX_FURNITURE_WIDTH or item.bbox[3] < body_y0 or item.bbox[1] > body_y1:
            continue
        lane_index = min(range(len(lanes)), key=lambda i: abs(item.x - lane_centers[i]))
        if abs(item.x - lane_centers[lane_index]) >= MIN_LANE_SEPARATION:
            continue
        assigned[lane_index].append(item)
    if any(not lane for lane in assigned):
        return None, {"reason": "native_empty_assigned_lane"}

    # The model's discarded table must attest the same page text the native
    # words carry: a stale or unrelated text layer may never be substituted
    # for what the model actually read.
    native_text = "\n".join(item.text for lane in assigned for item in lane)
    agreement = _token_f1(model_evidence_text, native_text)
    if agreement < MIN_NATIVE_AGREEMENT:
        return None, {
            "reason": "native_model_agreement_failure",
            "native_agreement": round(agreement, 4),
        }

    column_fragments: list[list[str]] = []
    hints: list[list[bool | None]] = []
    marker_intervals: list[list[tuple[int, int]]] = []
    for lane_index, lane in enumerate(assigned):
        ordered = sorted(lane, key=lambda item: (item.yc, item.x, item.record_index))
        column_fragments.append([_rule_prefix_as_count_attachment(item.text) for item in ordered])
        hints.append([_line_hint(item.text, item.x, lane_centers[lane_index]) for item in ordered])
        marker_intervals.append([_marker_interval(item.text) for item in ordered])
    grouped = entry_major.regroup_columns(column_fragments, hints=hints, column_bounded=True)
    if grouped is None:
        return None, {"reason": "native_unresolved_entry_boundaries"}
    grouped = [[_escape_cell(cell) for cell in column] for column in grouped]
    rendered = entry_major.render(grouped)
    if rendered is None:
        return None, {"reason": "native_empty_grouped_table"}

    before = Counter()
    for lane in assigned:
        for item in lane:
            before.update(entry_major.bag(item.text))
    if before != entry_major.bag(rendered):
        return None, {"reason": "native_token_conservation_failure"}

    box = [
        min(item.bbox[0] for lane in assigned for item in lane),
        min(item.bbox[1] for lane in assigned for item in lane),
        max(item.bbox[2] for lane in assigned for item in lane),
        max(item.bbox[3] for lane in assigned for item in lane),
    ]
    receipt = {
        "native_citations": total_citations,
        "count_markers_by_lane": count_markers_by_lane,
        "columns": len(lanes),
        "entries_by_column": [len(column) for column in grouped],
        "native_agreement": round(agreement, 4),
        "table_bbox": [int(v) for v in box],
        **(
            {"model_table_bbox": [int(v) if float(v).is_integer() else v for v in trusted_bbox]}
            if trusted_bbox is not None
            else {}
        ),
        "_native_token_set": set(_tokens(native_text)),
        "_native_marker_intervals": marker_intervals,
    }
    return {"bbox_2d": receipt["table_bbox"], "text_content": rendered}, receipt


def _model_table_audit(
    records: list[dict], table_record_indexes: set[int]
) -> tuple[str | None, int | None, list[list[tuple[int, int]]] | None]:
    """Return a proven renderer defect, the model's column count, and — on the
    clean single-table path only — each column's ordered per-cell marker
    intervals (the entry-count audit's model-side evidence; cell semantics match
    the scorer's ``_columns``: one interval per non-empty body cell).

    A citation-only cell is allowed before the first complete entry because a
    page can begin mid-entry.  The same cell after a count-marked entry proves
    that the entry-major grouping split one physical entry into multiple cells.
    Any parse ambiguity abstains; it is not evidence of a repairable defect.
    """
    found = []
    for index in sorted(table_record_indexes):
        text = str(records[index].get("text_content") or "")
        tables, _outside_content = _record_tables(text, index)
        found.extend((table, _bbox(records[index])) for table in tables)
    if not found:
        return None, None, None
    if len(found) > 1:
        boxes = [box for _table, box in found if box is not None]
        if len(boxes) != len(found):
            return None, None, None
        for left_index, left in enumerate(boxes):
            for right in boxes[left_index + 1 :]:
                if min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
                    left[1], right[1]
                ):
                    return (
                        "overlapping_model_tables",
                        max(len(table.header) for table, _box in found),
                        None,
                    )
        return None, None, None
    table = found[0][0]
    columns = len(table.header)
    if columns < 2:
        return None, columns, None
    if any(cell.strip() for cell in table.header):
        return "nonempty_rendered_header", columns, None
    if table.overflow_rows:
        return "renderer_dropped_cells", columns, None
    cell_markers: list[list[tuple[int, int]]] = [[] for _ in range(columns)]
    for column in range(columns):
        saw_complete_entry = False
        for row in table.rows:
            cell = row[column].strip() if column < len(row) else ""
            if not cell:
                continue
            cell_markers[column].append(_marker_interval(cell))
            if _COUNT_MARKER.search(cell):
                saw_complete_entry = True
                continue
            body = _LEADING_DASH.sub("", cell)
            if saw_complete_entry and _citation_count(body) and _CITE_ONLY_LINE.fullmatch(body):
                return "split_citation_continuation_cell", columns, None
    return None, columns, cell_markers


def _table_bbox(
    records: list[dict],
    table_record_indexes: set[int],
    truncated_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    boxes = [
        box for index in sorted(table_record_indexes) if (box := _bbox(records[index])) is not None
    ]
    if truncated_bbox is not None:
        boxes.append(truncated_bbox)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _native_table_output(
    records: list[dict],
    native_words: list[tuple],
    trigger: str,
    table_record_indexes: set[int],
    truncated_tail_text: str,
    trusted_bbox: tuple[float, float, float, float] | None = None,
    model_defect: str | None = None,
    model_columns: int | None = None,
    model_cell_markers: list[list[tuple[int, int]]] | None = None,
) -> tuple[str, dict] | None:
    """Discard the model's table, rebuild from native words, keep furniture.

    Consumes every table-bearing record plus any citation-dense plain record
    (its content is concordance body now represented by the rebuild); records
    kept alongside the table carry at most incidental citations.  Returns
    ``None`` on any abstention so the caller can preserve its existing
    fail-closed receipt.
    """
    consumed = set(table_record_indexes)
    # A complete table already contains the model's body.  Preserve every
    # non-table record initially; citation density alone is not authority to
    # absorb neighboring legal content.  Once the accepted native body is
    # known, a separate record can be removed only if geometry and native
    # tokens independently prove it duplicates that body.  The older
    # model-record fallback (no table indexes) still consumes the proven lane
    # records it replaces.
    if not table_record_indexes and not truncated_tail_text:
        for index, record in enumerate(records):
            text = record.get("text_content")
            if isinstance(text, str) and _citation_count(text) >= CONSUME_MIN_CITATIONS:
                consumed.add(index)
    evidence_parts = [str(records[index].get("text_content") or "") for index in sorted(consumed)]
    if truncated_tail_text:
        evidence_parts.append(truncated_tail_text)
    table, receipt = _rebuild_from_native_words(
        native_words,
        "\n".join(evidence_parts),
        trusted_bbox=trusted_bbox,
    )
    if table is None:
        return None
    if model_columns is not None:
        native_columns = int(receipt["columns"])
        if model_defect is None and model_columns != native_columns:
            model_defect = "model_native_column_count_mismatch"
        if model_defect is None and model_cell_markers is not None:
            # Entry-count audit (2026-08-11): a kept table must also agree with
            # the native lanes on per-column entry counts.  A column is proven
            # wrong only when its cell count contradicts the rebuild's entry
            # count AND no admissible reading of the lane's marker evidence can
            # reproduce its cells (see _marker_partition_feasible).
            entries_by_column = receipt["entries_by_column"]
            native_intervals = receipt["_native_marker_intervals"]
            for column, cell_intervals in enumerate(model_cell_markers):
                if len(cell_intervals) == entries_by_column[column]:
                    continue
                if _marker_partition_feasible(
                    tuple(cell_intervals), tuple(native_intervals[column])
                ):
                    continue
                model_defect = "model_native_entry_count_mismatch"
                receipt["model_entries_by_column"] = [
                    len(intervals) for intervals in model_cell_markers
                ]
                receipt["model_marker_counts_by_column"] = [
                    sum(low for low, _high in intervals) for intervals in model_cell_markers
                ]
                receipt["native_marker_counts_by_lane"] = [
                    sum(low for low, _high in intervals) for intervals in native_intervals
                ]
                break
        if model_defect is None:
            return None
    native_token_set = receipt.pop("_native_token_set")
    receipt.pop("_native_marker_intervals")
    native_bbox = tuple(float(v) for v in table["bbox_2d"])
    # A model table bbox is still valid whole-table geometry when it contains
    # every accepted native body word.  Preserve that envelope instead of
    # shrinking the table to word ink.  If any native edge escapes it, the bbox
    # is itself part of the model defect and cannot crop or replace the native
    # extent (the split-lane case exercises this branch).
    if (
        trusted_bbox is not None
        and trusted_bbox[0] <= native_bbox[0]
        and trusted_bbox[1] <= native_bbox[1]
        and trusted_bbox[2] >= native_bbox[2]
        and trusted_bbox[3] >= native_bbox[3]
    ):
        receipt["native_table_bbox"] = receipt["table_bbox"]
        receipt["table_bbox"] = [int(v) for v in trusted_bbox]
        table["bbox_2d"] = receipt["table_bbox"]
    # Uncorroborated-discard guard: a NON-table line packed into a
    # table-bearing record (a page header welded onto the table) is real page
    # content, not degenerate table junk; discarding it is only legal when
    # the native words the rebuild consumed corroborate its tokens.  Pipe
    # lines stay exempt — replacing a degenerate table's rows with the
    # native rebuild is this route's entire purpose — and plain lane records
    # are the model's own transcription of the table body, held to the
    # page-level agreement witness instead (a one-token OCR-noise line like
    # ``mmessenger@`` must not veto a rebuild that corrects it).
    for index in sorted(table_record_indexes):
        for line in str(records[index].get("text_content") or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("|"):
                continue
            line_tokens = _tokens(stripped)
            if not line_tokens:
                continue
            covered = sum(
                count for token, count in line_tokens.items() if token in native_token_set
            )
            if covered / sum(line_tokens.values()) < MIN_NATIVE_AGREEMENT:
                return None
    # Some checkpoints emit one printed table lane as a plain record beside a
    # malformed Markdown table.  The rebuild already contains that native
    # lane, so retaining the record duplicates page text.  Absorb it only with
    # three independent witnesses: concordance citations, substantial overlap
    # with the accepted native table extent, and token corroboration by the
    # exact native body used to build the replacement.  Legal furniture that
    # merely resembles a citation record therefore remains untouched.
    rebuilt_bbox = native_bbox
    for index, record in enumerate(records):
        if index in consumed:
            continue
        text = record.get("text_content")
        box = _bbox(record)
        if (
            not isinstance(text, str)
            or _citation_count(text) < CONSUME_MIN_CITATIONS
            or box is None
            or _bbox_overlap_fraction(box, rebuilt_bbox) < MIN_DUPLICATE_BBOX_OVERLAP
        ):
            continue
        record_tokens = _tokens(text)
        if not record_tokens:
            continue
        covered = sum(count for token, count in record_tokens.items() if token in native_token_set)
        if covered / sum(record_tokens.values()) >= MIN_NATIVE_AGREEMENT:
            consumed.add(index)
    output: list[dict] = []
    inserted = False
    insert_at = min(consumed) if consumed else len(records)
    for index, record in enumerate(records):
        if index == insert_at:
            output.append(table)
            inserted = True
        if index not in consumed:
            output.append(record)
    if not inserted:
        output.append(table)
    transformed = json.dumps(output, ensure_ascii=False)
    return transformed, {
        "status": "transformed",
        "reason": "native_words_table_rebuild",
        "route": "concordance",
        "trigger": trigger,
        **({"model_defect": model_defect} if model_defect else {}),
        "absorbed_records": len(consumed),
        "dropped_truncated_tail": bool(truncated_tail_text),
        **receipt,
    }


def _family_evidence(texts: list[str]) -> bool:
    """Concordance family witness on the model's own (discarded) table text."""
    joined = "\n".join(texts)
    return (
        _citation_count(joined) >= MIN_TOTAL_CITATIONS
        and len(_COUNT_MARKER.findall(joined)) >= MIN_COUNT_MARKERS
    )


def transform(raw: str, native_words: list[tuple] | None = None) -> tuple[str, dict]:
    """Return ``(raw_or_transformed, route_receipt)``; every failure abstains."""
    truncated_tail_text = ""
    truncated_bbox = None
    truncated = False
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # A decode that hit the token cap mid-record is invalid JSON.  Without
        # native words that stays a hard abstention.  With them, the complete
        # prefix records plus the truncated table text (whole cells only, the
        # harness's own salvage) may still prove the family for a rebuild.
        if native_words is None:
            return raw, {"status": "abstain", "reason": "invalid_model_json"}
        repaired = repair_truncated_json(raw)
        try:
            records = json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            return raw, {"status": "abstain", "reason": "invalid_model_json"}
        salvaged = _salvage_truncated_table_chunk(raw)
        if salvaged is not None:
            truncated_tail_text = str(salvaged[4]) if len(salvaged) > 4 else ""
            try:
                truncated_bbox = tuple(float(value) for value in salvaged[:4])
            except (TypeError, ValueError):
                return raw, {"status": "abstain", "reason": "invalid_model_json"}
        else:
            # Not a salvageable truncated table.  A cap-cut record's stub is a
            # handful of structural tokens; anything more is content this
            # route may not silently discard — and if the repaired prefix is
            # not literally a prefix of the raw, the failure is not the
            # understood truncation shape at all.
            if not (repaired.endswith("]") and raw.startswith(repaired[:-1])):
                return raw, {"status": "abstain", "reason": "invalid_model_json"}
            remainder = raw[len(repaired) - 1 :]
            remainder_tokens = [
                token
                for token in _TOKEN_RE.findall(remainder)
                if token not in ("bbox_2d", "text_content")
            ]
            if len(remainder_tokens) > MAX_UNSALVAGED_REMAINDER_TOKENS:
                return raw, {"status": "abstain", "reason": "invalid_model_json"}
        truncated = True
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        if truncated:
            return raw, {"status": "abstain", "reason": "invalid_model_json"}
        return raw, {"status": "abstain", "reason": "invalid_model_schema"}

    fragments: list[Fragment] = []
    table_record_indexes: set[int] = set()
    for index, record in enumerate(records):
        text = record.get("text_content")
        box = _bbox(record)
        if not isinstance(text, str) or box is None:
            continue
        # Images remain independent response records even when their boxes sit
        # inside a dense concordance lane. They are not lexical entry fragments.
        if text.strip() == "<image>":
            continue
        if _TABLE_LINE.search(text):
            table_record_indexes.add(index)
            continue
        fragments.append(Fragment(index, text.strip(), box, _citation_count(text)))

    if table_record_indexes or truncated:
        # Partial/single-column model tables were the source of all known
        # unsafe entry-boundary rewrites, so the model-record path never
        # unwraps them.  When the model's own table text proves the
        # concordance family, the native-words rebuild replaces it; every
        # other case abstains exactly as before.
        abstain_reason = "invalid_model_json" if truncated else "preformatted_table_ambiguous"
        table_texts = [
            str(records[index].get("text_content") or "") for index in sorted(table_record_indexes)
        ]
        if truncated_tail_text:
            table_texts.append(truncated_tail_text)
        if native_words is None or not _family_evidence(table_texts):
            return raw, {"status": "abstain", "reason": abstain_reason}
        model_defect = None
        model_columns = None
        model_cell_markers = None
        if not truncated:
            model_defect, model_columns, model_cell_markers = _model_table_audit(
                records, table_record_indexes
            )
            if model_columns is None:
                return raw, {"status": "abstain", "reason": "preformatted_table_ambiguous"}
        trusted_bbox = _table_bbox(records, table_record_indexes, truncated_bbox)
        if trusted_bbox is None:
            return raw, {"status": "abstain", "reason": abstain_reason}
        result = _native_table_output(
            records,
            native_words,
            "truncated_model_table" if truncated else "model_emitted_table",
            table_record_indexes,
            truncated_tail_text,
            trusted_bbox=trusted_bbox,
            model_defect=model_defect,
            model_columns=model_columns,
            model_cell_markers=model_cell_markers,
        )
        if result is None:
            return raw, {"status": "abstain", "reason": abstain_reason}
        return result

    anchors = [
        item
        for item in fragments
        if item.citations and item.bbox[2] - item.bbox[0] <= MAX_FURNITURE_WIDTH
    ]
    total_citations = sum(item.citations for item in anchors)
    if total_citations < MIN_TOTAL_CITATIONS:
        return raw, {
            "status": "abstain",
            "reason": "insufficient_citation_density",
            "citations": total_citations,
        }
    lanes = _cluster_lanes(anchors)
    if not lanes:
        return raw, {"status": "abstain", "reason": "ambiguous_column_lanes"}
    count_markers_by_lane = [
        sum(len(_COUNT_MARKER.findall(item.text)) for item in lane) for lane in lanes
    ]
    count_markers = sum(count_markers_by_lane)
    native_deficient_lane_proven = False
    if count_markers < MIN_COUNT_MARKERS or min(count_markers_by_lane) < MIN_LANE_COUNT_MARKERS:
        # Count-marker evidence must live inside every detected citation lane;
        # retained full-width furniture can never satisfy this gate.  When the
        # only deficit is one real short printed column, the native words can
        # corroborate the lane structure and take over the serialization.
        receipt = {
            "status": "abstain",
            "reason": "bare_term_format_unproven",
            "count_markers": count_markers,
            "count_markers_by_lane": count_markers_by_lane,
        }
        deficient = sum(1 for markers in count_markers_by_lane if markers < MIN_LANE_COUNT_MARKERS)
        if (
            native_words is not None
            and count_markers >= MIN_COUNT_MARKERS
            and deficient <= NATIVE_MAX_DEFICIENT_MARKER_LANES
        ):
            _native_table, native_receipt = _rebuild_from_native_words(
                native_words,
                "\n".join(
                    item.text for item in fragments if item.citations >= CONSUME_MIN_CITATIONS
                ),
            )
            if _native_table is not None and int(native_receipt["columns"]) == len(lanes):
                # Native geometry proves the short lane is real; it does not
                # replace model text. Continue through the existing
                # token-conserving model-fragment serializer so PDF text-layer
                # OCR/order differences cannot leak into output.
                native_deficient_lane_proven = True
            else:
                return raw, receipt
        else:
            return raw, receipt

    lane_centers = [
        sum(item.x * item.citations for item in lane) / sum(item.citations for item in lane)
        for lane in lanes
    ]
    if any(
        b - a < MIN_LANE_SEPARATION for a, b in zip(lane_centers, lane_centers[1:], strict=False)
    ):
        return raw, {"status": "abstain", "reason": "lanes_not_separated"}

    body_y0 = min(item.bbox[1] for item in anchors)
    body_y1 = max(item.bbox[3] for item in anchors)
    assigned: list[list[Fragment]] = [[] for _ in lanes]
    consumed: set[int] = set()
    for item in fragments:
        width = item.bbox[2] - item.bbox[0]
        if width > MAX_FURNITURE_WIDTH or item.bbox[3] < body_y0 or item.bbox[1] > body_y1:
            continue
        lane_index = min(range(len(lanes)), key=lambda i: abs(item.x - lane_centers[i]))
        # A fragment must be plausibly anchored to its selected printed lane.
        if abs(item.x - lane_centers[lane_index]) >= MIN_LANE_SEPARATION:
            continue
        assigned[lane_index].append(item)
        consumed.add(item.record_index)

    if not consumed or any(not lane for lane in assigned):
        return raw, {"status": "abstain", "reason": "empty_assigned_lane"}

    column_fragments: list[list[str]] = []
    hints: list[list[bool | None]] = []
    kept_by_record: dict[int, list[str]] = {}
    duplicate_lines = 0
    for lane_index, lane in enumerate(assigned):
        ordered = sorted(lane, key=lambda item: (item.yc, item.x, item.record_index))
        texts: list[str] = []
        lane_hints: list[bool | None] = []
        previous_item: Fragment | None = None
        previous_lines: list[str] = []
        for item in ordered:
            lines = [line.strip() for line in item.text.splitlines() if line.strip()]
            if not lines:
                continue
            # The decoder sometimes emits two vertically adjacent tiles with
            # an exact suffix/prefix overlap.  Remove only the longest exact
            # line sequence when geometry independently proves the tile seam.
            if previous_item is not None and abs(
                item.bbox[1] - previous_item.bbox[3]
            ) <= _tile_seam_tolerance(previous_item, previous_lines, item, lines):
                overlap = 0
                for size in range(1, min(len(previous_lines), len(lines)) + 1):
                    if previous_lines[-size:] == lines[:size]:
                        overlap = size
                if overlap:
                    lines = lines[overlap:]
                    duplicate_lines += overlap
            kept_by_record[item.record_index] = lines
            for line in lines:
                texts.append(line)
                lane_hints.append(_line_hint(line, item.x, lane_centers[lane_index]))
            previous_item = item
            previous_lines = [line.strip() for line in item.text.splitlines() if line.strip()]
        column_fragments.append(texts)
        hints.append(lane_hints)

    grouped = entry_major.regroup_columns(column_fragments, hints=hints, column_bounded=True)
    if grouped is None:
        return raw, {"status": "abstain", "reason": "unresolved_entry_boundaries"}
    grouped = [[_escape_cell(cell) for cell in column] for column in grouped]
    rendered = entry_major.render(grouped)
    if rendered is None:
        return raw, {"status": "abstain", "reason": "empty_grouped_table"}

    before = Counter()
    for index in consumed:
        before.update(entry_major.bag("\n".join(kept_by_record.get(index, []))))
    after = entry_major.bag(rendered)
    if before != after:
        return raw, {"status": "abstain", "reason": "token_conservation_failure"}

    box = [
        min(item.bbox[0] for lane in assigned for item in lane),
        min(item.bbox[1] for lane in assigned for item in lane),
        max(item.bbox[2] for lane in assigned for item in lane),
        max(item.bbox[3] for lane in assigned for item in lane),
    ]
    table = {"bbox_2d": box, "text_content": rendered}
    insert_at = min(consumed)
    output: list[dict] = []
    for index, record in enumerate(records):
        if index == insert_at:
            output.append(table)
        if index not in consumed:
            output.append(record)
    transformed = json.dumps(output, ensure_ascii=False)
    return transformed, {
        "status": "transformed",
        "reason": "dense_citations_separated_lanes",
        "route": "concordance",
        "citations": total_citations,
        "count_markers": count_markers,
        "count_markers_by_lane": count_markers_by_lane,
        "columns": len(lanes),
        "entries_by_column": [len(column) for column in grouped],
        "absorbed_records": len(consumed),
        "proven_duplicate_lines_removed": duplicate_lines,
        **({"native_deficient_lane_proven": True} if native_deficient_lane_proven else {}),
        "table_bbox": box,
    }


CONSTANTS = {
    "min_total_citations": MIN_TOTAL_CITATIONS,
    "min_lane_citations": MIN_LANE_CITATIONS,
    "min_lane_separation": MIN_LANE_SEPARATION,
    "max_furniture_width": MAX_FURNITURE_WIDTH,
    "max_lanes": MAX_LANES,
    "min_count_markers": MIN_COUNT_MARKERS,
    "min_lane_count_markers": MIN_LANE_COUNT_MARKERS,
    "tile_seam_pitch_multiplier": TILE_SEAM_PITCH_MULTIPLIER,
    "max_tile_seam_gap": MAX_TILE_SEAM_GAP,
    "native_segment_gap": NATIVE_SEGMENT_GAP,
    "native_line_y_tolerance": NATIVE_LINE_Y_TOLERANCE,
    "native_line_height_tolerance": NATIVE_LINE_HEIGHT_TOLERANCE,
    "native_max_deficient_marker_lanes": NATIVE_MAX_DEFICIENT_MARKER_LANES,
    "min_native_agreement": MIN_NATIVE_AGREEMENT,
    "consume_min_citations": CONSUME_MIN_CITATIONS,
    "min_duplicate_bbox_overlap": MIN_DUPLICATE_BBOX_OVERLAP,
}
