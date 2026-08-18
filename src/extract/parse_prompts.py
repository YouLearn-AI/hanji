"""Shared parse prompt policy and output-format renderers.

The parsing policy is the behavior contract: grouping, table rendering, image
handling, signatures, and bbox coverage. The renderer wraps that policy in the
schema/coordinate envelope required by a caller. That keeps GT labeling,
training/eval rows, and production serving aligned without forcing every caller
to emit the same JSON shape.
"""

from __future__ import annotations

_PARSE_PROMPT_POLICY_TEMPLATE = """\
Your DEFAULT is to GROUP. A normal page is 5-30 records - almost never more. A
"block" is a semantic SECTION (a panel, a heading + its content, a key-value group,
or a whole table), NOT a single line, cell, or field. If you are emitting one record
per line, per cell, or per form field, STOP - that is WRONG. When a region is not a
clean table, you must STILL group it into section blocks; never fall back to
one-record-per-element.

Block categories:
- {text_record}: ONE record per block - a heading TOGETHER WITH the lines beneath it,
  a paragraph, a list, or a key-value field group. text_content = the block's text,
  with "\\n" between its lines. DO NOT emit one record per line.
- {table_record}: a table or dense grid of cells, rendered as GitHub-Flavored Markdown
  (| col | col |\\n|---|---|\\n| cell | cell |). Never emit one record per cell or
  per row - the markdown carries every cell. A repeated item|amount list (receipt
  lines, menu items) IS a table.
  HEADERLESS TABLES: if a table has no visible column headings, DO NOT invent any.
  Render only the visible rows/cells in their observed order. If Markdown syntax needs
  a separator row, use empty header cells rather than synthetic names like "Column 1".
  TABLE CELL TEXT: cell contents must be plain visible text. Do NOT add Markdown
  emphasis or formatting inside cells (no **bold**, _italics_, backticks, or headings)
  unless that formatting is the only way to preserve information that is visible on
  the page.
  TALL TABLES: a table of up to ~15 rows is ONE block. If it is taller, SPLIT it into
  consecutive table blocks of ~10-15 rows each, stacked top-to-bottom. Each block's
  bbox_2d MUST tightly enclose ONLY its own rows - from the top of its first row to the
  bottom of its last row. NEVER emit a 50-row table as one block whose box covers only
  the first row: the box must span exactly the rows the block contains.
- {image_record}: ONE record per photo, figure, chart, scan, or non-text graphic.
  text_content = "<image>". Handwritten signatures, cursive e-signatures, initials,
  signature scribbles, and signature marks are ALWAYS images - do NOT transcribe or
  guess them, even if partly readable. Printed labels such as "Signature:" remain text.
  Do NOT emit for logos < 40 px wide.

CRITICAL - transcribe MEANING, not layout glyphs:
- Fill-in / blank lines: emit ONLY the label, NOT the blank. Write "Name:" - never
  "Name:________________". For signature fields, keep the printed label as text and
  {signature_image_instruction}.
- NEVER reproduce decorative rules or separators - rows of *, -, _, =, ., or any
  repeated glyph. Omit them entirely; they are not content.
- Checkboxes / Y-N / selection fields: write the field and its options on one line with the marks inline - "<row label> Y [x] N [ ]", "<label>: [x] Yes [ ] No"; "[x]" filled, "[ ]" empty; one record per field, not per option.
- text_content must equal the VISIBLE text of the block - never pad, repeat, or
  continue a character run. No single text block exceeds ~20 lines; split a longer
  section at its sub-headings.

Grouping rules:
- A section heading and the content beneath it (down to the next heading) form ONE block.
- A field label and its value are ONE block - EVEN in a dense report header. Write
  "Visit Date: 08/12/2025" as one record; NEVER split the label from its value
  ("Visit Date:" + "08/12/2025" as two records is WRONG).
- On forms, when a key has an associated value, merge the key and value into the same
  block and bbox so the association is explicit. If several related key-value fields
  are visually grouped, emit the group as one block with one "Label: value" line per field.
- A bordered or visually-grouped PANEL (e.g. a "PRESCRIBER INFORMATION" box with all
  its fields) is ONE block - join its label:value pairs with "\\n".
- A row of related cells that is NOT a clean table (a lab-result line, a transaction
  line) is ONE block - join the cells into one line; NEVER one record per cell.
- NEVER emit a bare value, a single cell, or a lone field as its own record.
- Keep COLUMNS separate: two side-by-side panels (e.g. Patient | Ordering Provider,
  Bill To | Ship To) are TWO blocks - never merge across the gutter.
- DO NOT merge unrelated neighboring panels or sections.
- Group by semantic relationship, NOT by bbox size. Small unrelated regions must stay
  separate. Example: a page title at the top-left and a page number at the top-right
  are TWO blocks, even if both boxes are small and on the same horizontal band. Only
  group items that belong to the same section, panel, list, table, or key-value group.
- Prefer CORRECT grouping over tight boxes: a block's bbox may be wide and may lightly
  touch a neighbor - do NOT over-split a section just to keep boxes small or separate.
- Page-edge content: if visible text touches or sits near the page boundary, inspect the
  full edge carefully and make the bbox include the entire visible glyphs/block, even if
  the box must start at 0 or end at 1000. Do NOT shrink edge boxes inward.
- Bbox coverage is strict: every transcribed character in text_content MUST be inside
  that record's bbox_2d, with no clipped letters. This matters most for small regions,
  rotated/non-horizontal text, page-edge text, headers/footers, stamps, and fax strips.
  Use a tight box around the actual region, but never make it so tight that any visible
  character you transcribed falls outside the box.
"""


def _policy_for_records(
    *,
    text_record: str,
    table_record: str,
    image_record: str,
    signature_image_instruction: str,
) -> str:
    return _PARSE_PROMPT_POLICY_TEMPLATE.format(
        text_record=text_record,
        table_record=table_record,
        image_record=image_record,
        signature_image_instruction=signature_image_instruction,
    )


PARSE_PROMPT_POLICY = _policy_for_records(
    text_record="Text records",
    table_record="Table records",
    image_record="Image records",
    signature_image_instruction=(
        'emit the actual signature mark itself as an image record with text_content="<image>"'
    ),
)


def render_parse_prompt(
    *,
    task_line: str,
    schema: str,
    coordinate_spec: str,
    text_record: str,
    table_record: str,
    image_record: str,
    signature_image_instruction: str,
    include_image_prefix: bool = False,
) -> str:
    """Render the shared parse policy inside a caller-specific output envelope."""

    prefix = "<image>\n" if include_image_prefix else ""
    policy = _policy_for_records(
        text_record=text_record,
        table_record=table_record,
        image_record=image_record,
        signature_image_instruction=signature_image_instruction,
    )
    return (
        f"{prefix}{task_line}\n\n"
        f"Schema: {schema}\n"
        f"Coordinates: {coordinate_spec}\n\n"
        f"{policy}\n"
        "Output JSON only.\n"
    )


GEMINI_PARSE_GT_NATIVE_YXYX_PROMPT = render_parse_prompt(
    task_line="You are annotating a document page. Detect every BLOCK and return a JSON array.",
    schema='[{"bbox_2d":[y_min,x_min,y_max,x_max], '
    '"type":"text|table|image", "text_content":"..."}]',
    coordinate_spec="normalized 0-1000 (Gemini's standard object-detection format).",
    text_record='type="text"',
    table_record='type="table"',
    image_record='type="image"',
    signature_image_instruction=(
        'emit the actual signature mark itself as type="image" with text_content="<image>"'
    ),
)

# The production serving prompt: the exact body the model was fine-tuned and
# gated on. The model is prompt-coupled — serve exactly this string (with the
# leading "<image>\n" prefix below).
PRODUCTION_BBOX_2D_JSON_PROMPT = r"""Detect every BLOCK in this document and return a JSON array.

Schema: [{"bbox_2d":[x1,y1,x2,y2], "text_content":"..."}]
Coordinates: normalized 0-1000 page coordinates; [x1,y1,x2,y2] = [left,top,right,bottom].

EMPTY PAGE: If the page has no legible content, return exactly []. Otherwise, transcribe every legible content block; a page with only one legible item is not empty.

Your DEFAULT is to GROUP. Most such pages form 5-30 records; sparse pages may form only 1-4. More than 30 remains unusual. A
"block" is a semantic SECTION (a panel, a heading + its content, a key-value group,
or a whole table), NOT a single line, cell, or field. If you are emitting one record
per line, per cell, or per form field, STOP - that is WRONG. When a region is not a
clean table, you must STILL group it into section blocks; never fall back to
one-record-per-element.

Block categories:
- Text records: ONE record per block - a heading TOGETHER WITH the lines beneath it,
  a paragraph, a list, or a key-value field group. text_content = the block's text,
  with "\n" between its lines. DO NOT emit one record per line.
- Table records: a table or dense grid of cells, rendered as GitHub-Flavored Markdown
  (| col | col |\n|---|---|\n| cell | cell |). Never emit one record per cell or
  per row - the markdown carries every cell. A repeated item|amount list (receipt
  lines, menu items) IS a table.
  HEADERLESS TABLES: if a table has no visible column headings, DO NOT invent any.
  Render only the visible rows/cells in their observed order. If Markdown syntax needs
  a separator row, use empty header cells rather than synthetic names like "Column 1".
  TABLE CELL TEXT: cell contents must be plain visible text. Do NOT add Markdown
  emphasis or formatting inside cells (no **bold**, _italics_, backticks, or headings)
  unless that formatting is the only way to preserve information that is visible on
  the page.
  TALL TABLES: A logical table on one page is ONE block regardless of row count. Include
  every visible row in one GFM table. Its bbox_2d must tightly enclose the full table
  from the first row through the last row.
- Image records: ONE record per photo, figure, chart, scan, or non-text graphic.
  text_content = "<image>". Handwritten signatures, cursive e-signatures, initials,
  signature scribbles, and signature marks are ALWAYS images - do NOT transcribe or
  guess them, even if partly readable. Printed labels such as "Signature:" remain text.
  Do NOT emit for logos < 40 px wide.

CRITICAL - transcribe MEANING, not layout glyphs:
- Fill-in / blank lines: emit ONLY the label, NOT the blank. Write "Name:" - never
  "Name:________________". For signature fields, keep the printed label as text and
  emit the actual signature mark itself as an image record with text_content="<image>".
- NEVER reproduce decorative rules or separators - rows of *, -, _, =, ., or any
  repeated glyph. Omit them entirely; they are not content.
- Checkboxes / Y-N / selection fields: write the field and its options on one line with the marks inline - "<row label> Y [x] N [ ]", "<label>: [x] Yes [ ] No"; "[x]" filled, "[ ]" empty; one record per field, not per option.
- text_content must equal the VISIBLE text of the block - never pad, repeat, or
  continue a character run. No single text block exceeds ~20 lines; split a longer
  section at its sub-headings.

Grouping rules:
- A section heading and the content beneath it (down to the next heading) form ONE block.
- A field label and its value are ONE block - EVEN in a dense report header. Write
  "Visit Date: 08/12/2025" as one record; NEVER split the label from its value
  ("Visit Date:" + "08/12/2025" as two records is WRONG).
- On forms, when a key has an associated value, merge the key and value into the same
  block and bbox so the association is explicit. If several related key-value fields
  are visually grouped, emit the group as one block with one "Label: value" line per field.
- A bordered or visually-grouped PANEL (e.g. a "PRESCRIBER INFORMATION" box with all
  its fields) is ONE block - join its label:value pairs with "\n".
- A row of related cells that is NOT a clean table (a lab-result line, a transaction
  line) is ONE block - join the cells into one line; NEVER one record per cell.
- NEVER emit a bare value, a single cell, or a lone field as its own record.
- Keep COLUMNS separate: two side-by-side panels (e.g. Patient | Ordering Provider,
  Bill To | Ship To) are TWO blocks - never merge across the gutter.
- DO NOT merge unrelated neighboring panels or sections.
- Group by semantic relationship, NOT by bbox size. Small unrelated regions must stay
  separate. Example: a page title at the top-left and a page number at the top-right
  are TWO blocks, even if both boxes are small and on the same horizontal band. Only
  group items that belong to the same section, panel, list, table, or key-value group.
- Prefer CORRECT grouping over tight boxes: a block's bbox may be wide and may lightly
  touch a neighbor - do NOT over-split a section just to keep boxes small or separate.
- Page-edge content: if visible text touches or sits near the page boundary, inspect the
  full edge carefully and make the bbox include the entire visible glyphs/block, even if
  the box must start at 0 or end at 1000. Do NOT shrink edge boxes inward.
- Bbox coverage is strict: every transcribed character in text_content MUST be inside
  that record's bbox_2d, with no clipped letters. This matters most for small regions,
  rotated/non-horizontal text, page-edge text, headers/footers, stamps, and fax strips.
  Use a tight box around the actual region, but never make it so tight that any visible
  character you transcribed falls outside the box.

Output JSON only.
Checkboxes: transcribe every checkbox as [x] if marked or [ ] if unmarked, placed before its label (e.g. "[x] Allergies reviewed"; Y/N pairs as "Y [x] N [ ]"). Include every checkbox, including checkbox grids and Y/N option pairs."""


PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE = (
    "<image>\n" + PRODUCTION_BBOX_2D_JSON_PROMPT
)
