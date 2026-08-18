"""LAYOUT extraction recipe — the default schema-extraction pass.

Multi-column forms (referrals, intake packets, two-column registration forms)
cause FIELD-BINDING errors in a plain row-major extraction: a patient's and an
emergency contact's address/phone get crossed, "LAST, FIRST" names are
mis-split, and values are pulled from a summary page instead of the structured
form. No single text pass fixes all of it.

The LAYOUT pass fixes binding without dropping fields: each chunk is prefixed
with its (x%,y%) page position plus a ``[SECTION]`` tag on recognized headings
(``layout_annotate``), and an entity-first / "FAMILY, GIVEN" instruction suffix
(``LAYOUT_INSTRUCTIONS``) is appended to the extraction prompt. One pass, the
same latency/cost as a plain extract.

Gated by ``settings.EXTRACT_LAYOUT_RECIPE_ENABLED`` (default ON — every schema
extraction uses it; set False to revert instantly to the plain pass).
"""

from __future__ import annotations

import re
from typing import Any

from extract.config import settings
from extract.core.schema_extract import (
    ChunkTransform,
    SchemaExtractResult,
    SchemaModel,
    _IndexedChunk,
)

#: Section/entity headings that anchor a region (first line of a chunk → [SECTION]).
_SECTION_RE = re.compile(
    r"(patient information|emergency contact|next of kin|next-of-kin|guarantor|"
    r"insurance|referral information|primary care|attending physician|referring "
    r"physician|facility|demographics|subscriber|imaging and procedures|messages)",
    re.I,
)

#: Entity-first / "FAMILY, GIVEN" layout instructions, appended to the extraction
#: prompt for the layout pass. Tells the model to group chunks into entities by the
#: (x%,y%)+[SECTION] tags ``layout_annotate`` injects and bind fields within a region
#: — the fix for 2-column patient-vs-next-of-kin binding on referral forms.
LAYOUT_INSTRUCTIONS: str | None = (
    "[LAYOUT] Each chunk is prefixed with its position as (x%,y%) of the page; "
    "section/entity headings are marked [SECTION]. This is a multi-column form. Use "
    "the coordinates and [SECTION] headings to group chunks into ENTITIES (patient, "
    "emergency contact / next-of-kin, guarantor, attending/referring/primary-care "
    "physician, facility, insurance). Binding rules (general):\n"
    "1. For each object, take its fields ONLY from chunks in the SAME section/region "
    "(near in x,y, under the same heading).\n"
    "2. When the SAME label (e.g. 'Home Phone', 'Address', 'Name') appears more than "
    "once, bind the value whose chunk is in the same section/region as the field's "
    "entity — never by reading order or label text alone.\n"
    "3. Never borrow a value from another entity's section; if a field is absent in "
    "its own section, use null.\n"
    "4. A name printed 'FAMILY, GIVEN MIDDLE' (with a comma): first_name = the "
    "token(s) AFTER the comma; last_name = the text BEFORE the comma. Never take the "
    "first printed token as first_name just because it appears first.\n"
    "5. Prefer the structured form over a summary page, "
    "but never drop a value that is genuinely present anywhere."
)


def layout_annotate(index: int, chunk: _IndexedChunk) -> str:
    """Prefix the chunk text with ``[SECTION] (x%,y%)`` so the model can group
    2-column fields by region (the binding fix). ``chunk.bbox`` is already normalized
    to 0–1000 page-relative, so (x%,y%) = bbox-center / 10. When bbox is absent (e.g.
    a caller sends ``page_sizes=[]``), the coordinate prefix is omitted — the
    [SECTION] tag and the prompt suffix still apply. Rewrites only the rendered
    text; chunk page/bbox geometry is untouched.
    """
    b = chunk.bbox
    pos = ""
    if b and len(b) >= 4:
        pos = f"(x{int((b[0] + b[2]) / 2 / 10)},y{int((b[1] + b[3]) / 2 / 10)}) "
    first_line = (chunk.text or "").split("\n", 1)[0]
    sec = "[SECTION] " if _SECTION_RE.search(first_line) else ""
    if "\n" in (chunk.text or ""):
        # Multi-line chunk (a markdown table, a wrapped paragraph): the annotation
        # goes on its OWN line so it cannot be read as part of the table's header
        # row. Single-line chunks keep the historical inline prefix byte-identical.
        return f"{sec}{pos}".rstrip() + "\n" + chunk.text if (sec or pos) else chunk.text
    return f"{sec}{pos}{chunk.text}"


# ChunkTransform is satisfied by layout_annotate; the alias documents intent.
_LAYOUT_TRANSFORM: ChunkTransform = layout_annotate


def should_use_layout_recipe() -> bool:
    """Whether schema extraction defaults to the single LAYOUT pass.

    Global toggle (``settings.EXTRACT_LAYOUT_RECIPE_ENABLED``). When ON, EVERY
    extraction uses the layout pass; set the flag False to revert instantly.
    """
    return settings.EXTRACT_LAYOUT_RECIPE_ENABLED


async def aextract_layout(
    chunks: list[Any],
    page_sizes: list[tuple[float, float]],
    user_schema: dict[str, Any],
    *,
    model: SchemaModel,
    strict: bool = True,
    xref_images: dict[int, bytes] | None = None,
    xref_cache: Any = None,
    xref_second_reader: bool = True,
    page_images: dict[int, bytes] | None = None,
    xref_prefetched: list | None = None,
    exempt_unquotable: bool = False,
    bare_scalars: bool | None = None,
    keep_repeated_rows: bool = False,
    chunk_ids: bool = True,
    context_first: bool = False,
    aux_arrays_inline: bool | None = None,
    omit_null: bool | None = None,
    short_row_keys: bool | None = None,
    cache_stagger_seconds: float | None = None,
) -> SchemaExtractResult:
    """The single LAYOUT pass: the plain schema extract + (x%,y%)+[SECTION] chunk
    annotation + the entity-binding prompt suffix. One pass, same latency/cost as a
    normal extract, fixing 2-column field binding.

    EVERY per-request option is forwarded rather than dropped, and that is the whole
    point of the parameter list above. This recipe is the global default
    (``EXTRACT_LAYOUT_RECIPE_ENABLED``), so an option this wrapper does not name is
    an option that silently does nothing for almost every caller while its flag
    reads as ON. Adding an option to ``run_schema_pass`` means adding it here in the
    same change.
    """
    from extract.core.schema_extract import aextract_schema

    return await aextract_schema(
        chunks,
        page_sizes,
        user_schema,
        model=model,
        strict=strict,
        chunk_transform=_LAYOUT_TRANSFORM,
        extra_instructions=LAYOUT_INSTRUCTIONS,
        xref_images=xref_images,
        xref_cache=xref_cache,
        xref_second_reader=xref_second_reader,
        xref_prefetched=xref_prefetched,
        page_images=page_images,
        exempt_unquotable=exempt_unquotable,
        bare_scalars=bare_scalars,
        keep_repeated_rows=keep_repeated_rows,
        chunk_ids=chunk_ids,
        context_first=context_first,
        aux_arrays_inline=aux_arrays_inline,
        omit_null=omit_null,
        short_row_keys=short_row_keys,
        cache_stagger_seconds=cache_stagger_seconds,
    )
