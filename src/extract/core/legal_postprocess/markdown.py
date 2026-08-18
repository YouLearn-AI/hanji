"""Renderer-faithful GFM helpers shared by the production legal cascade.

These are the minimal production-owned subset of the eval line-table parser. The
contract test pins them against ``evals2.core.metrics.line_table`` so routing and
scoring cannot silently diverge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markdown_it.token import Token

_MD = None
_WS = re.compile(r"\s+")
_TEXT_CONTENT_OPEN_RE = re.compile(r'"text_content"\s*:\s*"')
_BBOX_2D_INNER_RE = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]*)\]')


def _md():
    global _MD
    if _MD is None:
        from markdown_it import MarkdownIt

        _MD = MarkdownIt("commonmark").enable("table")
    return _MD


def _plain(tok: Token) -> str:
    if tok.children is None:
        return tok.content
    parts: list[str] = []
    for child in tok.children:
        if child.type in ("text", "code_inline", "html_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif child.children:
            parts.append(_plain(child))
    return "".join(parts)


def _escaped_split(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    previous = ""
    for char in line:
        if char == "|" and previous != "\\":
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
        previous = char
    cells.append("".join(current))
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


@dataclass
class _RTable:
    header: list[str]
    rows: list[list[str]]
    overflow_rows: int
    chunk_index: int

    @property
    def first_cells(self) -> list[str]:
        return [row[0] if row else "" for row in self.rows]

    def body_text(self) -> str:
        return " ".join(" ".join(cell for cell in row[1:] if cell) for row in self.rows)


def _record_tables(text: str, chunk_index: int) -> tuple[list[_RTable], bool]:
    lines = text.splitlines()
    tokens = _md().parse(text)
    tables: list[_RTable] = []
    outside_content = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "table_open":
            cursor = index + 1
            depth = 1
            header: list[str] = []
            rows: list[list[str]] = []
            overflow = 0
            in_head = False
            row_cells: list[str] | None = None
            row_map: tuple[int, int] | None = None
            while cursor < len(tokens) and depth:
                current = tokens[cursor]
                if current.type == "table_open":
                    depth += 1
                elif current.type == "table_close":
                    depth -= 1
                elif current.type == "thead_open":
                    in_head = True
                elif current.type == "thead_close":
                    in_head = False
                elif current.type == "tr_open":
                    row_cells = []
                    row_map = tuple(current.map) if current.map else None
                elif current.type == "tr_close" and row_cells is not None:
                    if in_head:
                        header = row_cells
                    else:
                        rows.append(row_cells)
                        if row_map is not None and 0 <= row_map[0] < len(lines):
                            source_cells = len(_escaped_split(lines[row_map[0]]))
                            if header and source_cells > len(header):
                                overflow += 1
                    row_cells = None
                elif current.type == "inline" and row_cells is not None:
                    row_cells.append(_WS.sub(" ", _plain(current)).strip())
                cursor += 1
            tables.append(_RTable(header, rows, overflow, chunk_index))
            index = cursor
            continue
        if token.type in ("fence", "code_block", "html_block", "hr") or (
            token.type == "inline" and _plain(token).strip()
        ):
            outside_content = True
        index += 1
    return tables, outside_content


def repair_truncated_json(raw: str) -> str:
    if "[" not in raw:
        return raw
    start = raw.find("[")
    body = raw[start:]
    depth = 0
    in_string = False
    escaped = False
    last = -1
    for index, char in enumerate(body):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if char == "}" and depth == 1:
                last = index
    if last < 0:
        return raw
    return raw[:start] + body[: last + 1] + "]"


def _string_is_unterminated(body: str) -> bool:
    escaped = False
    for char in body:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return False
    return True


def _salvage_truncated_table_chunk(raw: str) -> tuple[int, int, int, int, str] | None:
    depth = 0
    in_string = False
    escaped = False
    last_complete_end = -1
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                last_complete_end = index
    open_index = raw.rfind("{")
    if open_index <= last_complete_end:
        return None
    fragment = raw[open_index:]
    bbox_match = _BBOX_2D_INNER_RE.search(fragment)
    text_match = _TEXT_CONTENT_OPEN_RE.search(fragment)
    if not bbox_match or not text_match:
        return None
    body = fragment[text_match.end() :]
    if not _string_is_unterminated(body) or (body.count("|") < 2 and "\\n" not in body):
        return None
    pipe = body.rfind("|")
    newline = body.rfind("\\n")
    cut = max(pipe + 1 if pipe != -1 else -1, newline + 2 if newline != -1 else -1)
    if cut <= 0:
        return None
    rebuilt = (
        '{"bbox_2d":['
        + bbox_match.group(1)
        + '], "text_content":"'
        + body[:cut].rstrip("\\")
        + '"}'
    )
    try:
        record = json.loads(rebuilt)
        bbox = record["bbox_2d"]
        if len(bbox) != 4:
            return None
        x0, y0, x1, y1 = (int(value) for value in bbox)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return x0, y0, x1, y1, str(record.get("text_content") or "")
