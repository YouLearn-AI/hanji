"""Regression tests for the Qwen-LoRA bbox_2d JSON parser.

Locks in the fixes from the Codex adversarial review: the parser must never drop
a record the valid-array path would keep (truncated arrays, key aliases/order,
float coords, escaped strings, trailing prose), must not crash on non-finite
coords, must split GFM rows correctly (code spans), must not mis-route prose into
tables, and the list-input path must round-trip the parser's own output.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract.core.ocr.qwen_lora import (  # noqa: E402
    _looks_like_markdown_table,
    _split_markdown_row,
    parse_bbox_2d_json,
    parse_bbox_2d_records,
    parse_qwen_lora_response,
)


def test_valid_array():
    assert parse_bbox_2d_json('[{"bbox_2d": [0, 0, 100, 100], "text_content": "hi"}]') == [
        (0, 0, 100, 100, "hi")
    ]


def test_code_fence_and_trailing_special_tokens():
    raw = '```json\n[{"bbox_2d":[1,2,3,4],"text_content":"x"}]\n```<|im_end|>'
    assert parse_bbox_2d_json(raw) == [(1, 2, 3, 4, "x")]


def test_truncated_array_keeps_complete_records():
    # array never closes; the one complete record must survive
    assert parse_bbox_2d_json('[{"bbox_2d":[0,0,9,9],"text_content":"a"},{"bbox_2d":[0,0,9,9]') == [
        (0, 0, 9, 9, "a")
    ]


def test_truncated_array_with_aliases_reorder_floats():
    # Codex #1: the old regex fallback dropped all of these.
    assert parse_bbox_2d_json('[{"bbox": [0, 0, 100, 100], "text": "alias"}') == [(0, 0, 100, 100, "alias")]
    assert parse_bbox_2d_json('[{"text_content": "rev", "bbox_2d": [0,0,100,100]}') == [(0, 0, 100, 100, "rev")]
    assert parse_bbox_2d_json('[{"bbox_2d": [0.5,0,100.5,100], "text_content": "f"}') == [(0, 0, 100, 100, "f")]


def test_trailing_bracket_prose_does_not_force_drop():
    # Codex #1: rfind("]") used to include trailing "[done]" and drop the record.
    assert parse_bbox_2d_json('[{"bbox":[0,0,100,100],"text":"a"}] trailing [done]') == [(0, 0, 100, 100, "a")]


def test_backslash_terminated_text_in_truncated_array():
    # Codex #2: a legal JSON string ending in an escaped backslash.
    assert parse_bbox_2d_json('[{"bbox_2d":[0,0,100,100],"text_content":"C:\\\\"}') == [(0, 0, 100, 100, "C:\\")]


def test_brace_inside_text_does_not_break_object_scan():
    assert parse_bbox_2d_json('[{"bbox_2d":[0,0,9,9],"text_content":"a}{b"},{"bbox_2d":[0,0,9,9]') == [
        (0, 0, 9, 9, "a}{b")
    ]


def test_non_finite_coords_skipped_not_crash():
    # Codex #3: json accepts Infinity/NaN; must skip, not raise OverflowError.
    out = parse_bbox_2d_json('[{"bbox_2d":[Infinity,0,9,9],"text_content":"bad"},{"bbox_2d":[0,0,9,9],"text_content":"ok"}]')
    assert out == [(0, 0, 9, 9, "ok")]


def test_missing_or_wrong_shapes_skipped():
    assert parse_bbox_2d_json('[{"text_content":"no bbox"},{"bbox_2d":[0,0,9],"text_content":"short"},42,"str"]') == []


def test_split_markdown_row_codespan_and_escape():
    # Codex #4: pipes inside backtick code spans / escaped are literal cell content.
    assert _split_markdown_row("| `a|b` | 1 |") == ["`a|b`", "1"]
    assert _split_markdown_row(r"| a\|b | 2 |") == ["a|b", "2"]


def test_prose_with_dashes_is_not_a_table():
    # Codex #5: a stray dashed line + one pipe is prose, not a table.
    assert _looks_like_markdown_table("Claim A | Claim B\n---\nThis is prose.") is False
    assert _looks_like_markdown_table("| A | B |\n| --- | --- |\n| 1 | 2 |") is True


def test_ragged_body_rows_are_still_a_table():
    # The model's real tables have ragged rows (merged/empty cells, embedded-newline
    # wraps). A header + valid delimiter is the table signal; ragged body rows must
    # NOT demote the whole table back to prose (a production "we have no tables" bug).
    ragged = (
        "| Type | Item Code | Quantity | Unit | Unit Cost | Extension |\n"
        "|---|---|---|---|---|---|\n"
        "| 1) (s ) | | | | ( | 303.60 ) |\n"      # ragged: fewer real cells
        "| 13617 HIGDON |  | 246581 | METHOWVLY GEN WAGES | 2,064.21 |"  # different width
    )
    assert _looks_like_markdown_table(ragged) is True


def test_header_delimiter_then_pure_prose_is_not_a_table():
    # The one shape still rejected: a header + delimiter followed by a body that
    # contains no pipe at all is prose, not a table.
    assert _looks_like_markdown_table("| A | B |\n| --- | --- |\njust a sentence.") is False


def test_real_table_routes_to_table_not_block():
    res = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d":[0,0,1000,1000],"text_content":"| A | B |\\n| --- | --- |\\n| 1 | 2 |"}]'},
        page_width=100,
        page_height=100,
    )
    assert len(res.tables) == 1 and res.tables[0].n_cols == 2
    assert not res.blocks


def test_list_input_roundtrips_parser_output():
    # Codex #6: parser-native tuples must round-trip through the list-input path.
    recs = parse_bbox_2d_json('[{"bbox_2d":[0,0,100,100],"text_content":"z"}]')
    res = parse_qwen_lora_response(recs, page_width=10, page_height=10)
    assert [b.text for b in res.blocks] == ["z"]


def test_image_sentinel_becomes_figure():
    res = parse_qwen_lora_response(
        {"raw": '[{"bbox_2d":[0,0,1000,1000],"text_content":"<image>"}]'},
        page_width=100,
        page_height=100,
    )
    assert len(res.figures) == 1 and not res.blocks and not res.tables


def test_oversized_int_coord_skipped_not_crash():
    # Codex r2 #1: float(huge_int) raises OverflowError before round(); must skip.
    big = "9" * 309
    out = parse_bbox_2d_json(
        '[{"bbox_2d":[' + big + ',0,9,9],"text_content":"bad"},{"bbox_2d":[0,0,1,1],"text_content":"ok"}]'
    )
    assert out == [(0, 0, 1, 1, "ok")]


def test_boolean_coords_rejected():
    # Codex r2 #2: bool is an int subclass; true/false must not become 1/0 coords.
    assert parse_bbox_2d_json('[{"bbox_2d":[true,false,9,9],"text_content":"b"}]') == []


def test_delimiter_row_with_empty_internal_cell_is_not_a_table():
    # Codex r2 #3: "|  | --- |" is not a valid delimiter row.
    assert _looks_like_markdown_table("| A | B |\n|  | --- |\n| x | y |") is False


def test_extract_output_tokens_only_for_dict_envelope():
    # Codex r2 #4: non-dict responses (which the parser tolerates) carry no count.
    from extract.core.ocr.qwen_lora import _extract_output_tokens
    assert _extract_output_tokens({"n_output_tokens": 5}) == 5
    assert _extract_output_tokens({"raw": "[]"}) is None
    assert _extract_output_tokens(["x"]) is None
    assert _extract_output_tokens("x") is None


def test_multibacktick_codespan_pipe_not_split():
    # Codex r3 #1: GFM code spans use runs of backticks; a `|` inside `` ``...`` `` is literal.
    assert _split_markdown_row("| ``a|b`` | 1 |") == ["``a|b``", "1"]


def test_ragged_body_is_a_table_but_pure_prose_body_is_not():
    # Updated: a ragged body row (wrong width but still pipe-delimited) IS a table —
    # the header+delimiter is the signal and _table_from_markdown normalizes ragged
    # rows. The old all-rows-same-width rule dropped ~45% of real tables to prose
    # (a production "we have no tables" bug). Trailing PURE prose (no pipe) is still
    # not a table.
    assert _looks_like_markdown_table("| A | B |\n| --- | --- |\n| 1 | 2 | 3 |") is True
    assert _looks_like_markdown_table("| A | B |\n| --- | --- |\nThis is prose after table.") is False
    assert _looks_like_markdown_table("| A | B |\n| --- | --- |\n| 1 | 2 |") is True  # clean still works


def test_bool_token_count_rejected():
    # Codex r3 #4: bool is an int subclass; a bool n_output_tokens is not a count.
    from extract.core.ocr.qwen_lora import _extract_output_tokens
    assert _extract_output_tokens({"n_output_tokens": True}) is None
    assert _extract_output_tokens({"n_output_tokens": 12}) == 12


def test_unescaped_backslash_and_newline_in_text_recovered():
    # Codex r4: models emit unescaped LaTeX backslashes / literal newlines in
    # text_content; the record must be recovered, not silently dropped.
    out = parse_bbox_2d_json(
        '[{"bbox_2d":[0,0,9,9],"text_content":"$\\alpha$"},{"bbox_2d":[1,1,2,2],"text_content":"ok"}]'
    )
    assert out == [(0, 0, 9, 9, "$\\alpha$"), (1, 1, 2, 2, "ok")]
    nl = parse_bbox_2d_json(
        '[{"bbox_2d":[0,0,9,9],"text_content":"line1\nline2"},{"bbox_2d":[1,1,2,2],"text_content":"ok"}]'
    )
    assert nl == [(0, 0, 9, 9, "line1\nline2"), (1, 1, 2, 2, "ok")]


def test_latex_backslash_u_command_recovered_but_valid_unicode_kept():
    # Codex r5 #2: \u must be 4 hex digits; LaTeX \underline (\u + non-hex) recovers.
    out = parse_bbox_2d_json(r'[{"bbox_2d":[0,0,9,9],"text_content":"$\underline{x}$"},{"bbox_2d":[1,1,2,2],"text_content":"ok"}]')
    assert out == [(0, 0, 9, 9, r"$\underline{x}$"), (1, 1, 2, 2, "ok")]
    # a valid \uXXXX escape is still honored
    assert parse_bbox_2d_json(r'[{"bbox_2d":[0,0,9,9],"text_content":"AB"}]') == [(0, 0, 9, 9, "AB")]


def test_truncated_table_blob_salvages_complete_cells():
    # The p0186 class: a whole dense page is ONE table record whose text_content
    # string is cut off mid-stream at the token cap, so the trailing object never
    # closes and the per-object scan drops it WHOLE — losing every amount. The
    # complete pipe cells emitted before the cut must be salvaged.
    raw = (
        '[{"bbox_2d":[0,0,9,9],"text_content":"header"}, '
        '{"bbox_2d":[10,20,900,800],"text_content":'
        '"| a | 115.00 |\\n| b | 245.00 | partial 999'
    )
    out = parse_bbox_2d_json(raw)
    # the header survives, and the table chunk is recovered (not dropped whole)
    assert (0, 0, 9, 9, "header") in out
    salvaged = [r for r in out if r[:4] == (10, 20, 900, 800)]
    assert len(salvaged) == 1
    text = salvaged[0][4]
    # complete cells (closed by | or \n) are kept; the partial trailing cell ("999",
    # no closing | or \n) is dropped — so no phantom amount is invented.
    assert "115.00" in text and "245.00" in text
    assert "999" not in text


def test_truncated_table_salvage_is_phantom_free_on_partial_amount():
    # An amount cut mid-digits (no trailing cell boundary) must NOT be recovered —
    # the kill criterion: never invent an amount absent as complete digits in raw.
    raw = (
        '[{"bbox_2d":[10,20,900,800],"text_content":'
        '"| x | 140.00 |\\n| y | 38'  # "38" is a truncated amount, no closing |
    )
    out = parse_bbox_2d_json(raw)
    salvaged = [r for r in out if r[:4] == (10, 20, 900, 800)]
    assert len(salvaged) == 1
    assert "140.00" in salvaged[0][4]
    # the partial row "| y | 38" lacks a trailing boundary, so 38 is dropped
    assert "38" not in salvaged[0][4]


def test_salvage_no_op_on_complete_array():
    # A non-truncated array must be untouched — the salvage only fires on an
    # unterminated trailing pipe-table string.
    raw = '[{"bbox_2d":[0,0,9,9],"text_content":"| a | 1.00 |\\n| b | 2.00 |"}]'
    out = parse_bbox_2d_json(raw)
    assert out == [(0, 0, 9, 9, "| a | 1.00 |\n| b | 2.00 |")]


def test_salvage_no_op_on_truncated_bbox():
    # The other truncation shape (cut at the START of a new small object, before its
    # text_content) has nothing to salvage and must be left alone.
    raw = '[{"bbox_2d":[0,0,9,9],"text_content":"ok"}, {"bbox_2d":[10,20,30'
    assert parse_bbox_2d_json(raw) == [(0, 0, 9, 9, "ok")]


def test_salvage_no_op_on_truncated_prose():
    # A truncated PROSE string (not a pipe table) must not be mangled into cells.
    raw = '[{"bbox_2d":[0,0,9,9],"text_content":"ok"}, {"bbox_2d":[10,20,30,40],"text_content":"some prose cut off here'
    assert parse_bbox_2d_json(raw) == [(0, 0, 9, 9, "ok")]


# --- KV region discriminator (plan 061) ------------------------------------


def test_records_carry_kv_kind():
    # parse_bbox_2d_records surfaces the "kv" discriminator; None for champion records.
    raw = (
        '[{"bbox_2d":[0,0,9,9],"text_content":"prose"},'
        '{"type":"kv","bbox_2d":[10,10,90,90],"text_content":"Name: Ada\\nDOB: <empty>"}]'
    )
    recs = parse_bbox_2d_records(raw)
    assert recs == [
        (0, 0, 9, 9, "prose", None),
        (10, 10, 90, 90, "Name: Ada\nDOB: <empty>", "kv"),
    ]


def test_kv_type_is_normalized_strip_and_lower():
    # The discriminator is normalized (strip + lower) so a tolerant "  KV " still
    # routes; the 5-tuple contract (tests + bench parity) still drops it entirely.
    raw = '[{"type":" KV ","bbox_2d":[1,2,3,4],"text_content":"K: V"}]'
    assert parse_bbox_2d_json(raw) == [(1, 2, 3, 4, "K: V")]
    assert parse_bbox_2d_records(raw) == [(1, 2, 3, 4, "K: V", "kv")]
    page = parse_qwen_lora_response(raw, page_width=1000.0, page_height=1000.0)
    assert len(page.key_values) == 1


def test_prelist_input_roundtrips_6tuple_kv_records():
    # A caller feeding parse_bbox_2d_records output (6-tuples) back into the
    # response parser must not silently drop KV records (the list-tolerance path).
    recs = parse_bbox_2d_records('[{"type":"kv","bbox_2d":[0,0,90,90],"text_content":"K: V"}]')
    page = parse_qwen_lora_response(recs, page_width=1000.0, page_height=1000.0)
    assert len(page.key_values) == 1 and page.key_values[0].text == "K: V"


def test_unknown_type_value_ignored_by_router():
    # A stray/unknown type must not crash; it is kept as its normalized kind and
    # falls through to the text/table inference (router only special-cases "kv").
    recs = parse_bbox_2d_records('[{"type":"weird","bbox_2d":[0,0,9,9],"text_content":"x"}]')
    assert recs == [(0, 0, 9, 9, "x", "weird")]


def test_response_routes_kv_records_to_key_values():
    raw = (
        '[{"type":"kv","bbox_2d":[100,100,900,300],'
        '"text_content":"Next of Kin\\nName: Karen Nolan\\nWork Phone: <empty>\\n[ ] POA"},'
        '{"bbox_2d":[0,0,1000,50],"text_content":"# Header"}]'
    )
    page = parse_qwen_lora_response(raw, page_width=1000.0, page_height=1000.0)
    assert len(page.key_values) == 1
    kv = page.key_values[0]
    assert kv.text.startswith("Next of Kin")
    assert "Work Phone: <empty>" in kv.text and "[ ] POA" in kv.text
    # The non-KV record stays a normal text block; the KV text is NOT a block.
    assert len(page.blocks) == 1
    assert page.blocks[0].text == "# Header"


def test_kv_discriminator_wins_over_table_inference():
    # A KV region whose text happens to contain pipes must NOT be parsed as a table.
    raw = '[{"type":"kv","bbox_2d":[0,0,900,200],"text_content":"Route: A | B | C\\nStatus: open"}]'
    page = parse_qwen_lora_response(raw, page_width=1000.0, page_height=1000.0)
    assert len(page.key_values) == 1
    assert page.tables == []


def test_kv_discriminator_wins_over_image_sentinel():
    # The explicit discriminator wins over ALL content inference, including the
    # image sentinel — a record tagged type:"kv" is a KV region regardless of text.
    raw = '[{"type":"kv","bbox_2d":[0,0,900,200],"text_content":"<image>"}]'
    page = parse_qwen_lora_response(raw, page_width=1000.0, page_height=1000.0)
    assert len(page.key_values) == 1
    assert page.figures == []


def test_empty_kv_text_is_dropped():
    raw = '[{"type":"kv","bbox_2d":[0,0,9,9],"text_content":"   "}]'
    page = parse_qwen_lora_response(raw, page_width=1000.0, page_height=1000.0)
    assert page.key_values == []
