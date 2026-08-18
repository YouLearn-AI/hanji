"""Schema extraction over parse chunks (plan 050).

Given a document's parse chunks (text + page + bbox) and an arbitrary user JSON
schema, fill the schema from the document with a grounded LLM pass and attach a
per-field citation (page + bbox + verbatim source span).

The anti-hallucination guarantee: the model never emits coordinates. Every leaf
comes back as ``{value, chunks: [int], quote}`` — the model returns cheap integer
chunk indices and a verbatim quote; this module remaps each index to the chunk's
``{page, bbox, text}`` and verifies the quote is a real substring of a cited
chunk. A non-null value whose quote can't be found anywhere is flagged
``UNGROUNDED`` (suspected fabrication). Coordinates come 100% from the parser, so
they can't be hallucinated.

The gate has one principled boundary: a value the SCHEMA itself makes unquotable —
an enum option no page prints verbatim, a normalized number or composed date, a
zero default — is verified against what it CAN be checked against (enum membership,
its digits on the page) and reported in ``verified_by`` instead of being nulled
(:func:`_unquotable_exemption`; EXP-A measured 577 correct cells destroyed per 30
fabrications caught). Free text keeps the quote-or-death rule unchanged.

This is the consolidation of three prior implementations into ``core``:
``scripts/schema_extract_prototype.py`` (the blueprint: wrap/convert/verify),
``web/src/server/schema-extract.ts`` (per-chunk-not-union boxes, text-match
augmentation, containment fallback), and earlier key-concepts machinery
(token-budget batching + offset re-basing so long docs work in one request).

``core`` stays pure: the LLM call is injected as a :class:`SchemaModel`. The
Vertex structured-output client lives in ``extract.clients.gemini_extract``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from extract.config import settings
from extract.core.models import Chunk, ChunkType, FieldEvidence

# --- Tunables ---------------------------------------------------------------


log = logging.getLogger(__name__)


def _bounded_env_int(name: str, *, default: int, low: int, high: int) -> int:
    """An environment override read inside its supported range, or the default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer — using %d", name, raw, default)
        return default
    if not (low <= value <= high):
        log.warning("%s=%d is outside [%d, %d] — using %d", name, value, low, high, default)
        return default
    return value


#: Hard nesting cap, enforced at validation time (plan 050 D4). Matches Extend;
#: well within Gemini 3.5 Flash. A pathological/cyclic schema fails here, before
#: any model call.
MAX_SCHEMA_DEPTH = 5
#: Output volume — not input length or nesting — is the binding failure mode for
#: wide schemas (ExtractBench: 0 valid fields on a 369-field schema). Field-group
#: splitting (below) handles breadth, so this is now only a sanity ceiling against
#: absurd schemas; beyond it, ask the caller to split.
MAX_LEAF_FIELDS = 1000
#: Field-group splitting (plan 050 M4, validated 2026-06-18): a schema with more
#: than this many leaves is partitioned into disjoint field-groups of ≤ this size,
#: each extracted concurrently against the same chunks and merged. The
#: {value,chunks,quote} wrapper triples the property count, so a 369-field schema
#: (~1,100 wrapped props) overflows Gemini's response-schema limit → 400; split
#: into groups, each fits. (ExtractBench 10kq: 0.000 → 0.226, 0 failures.)
#: Capped at 40 (not 50): on a real customer schema (70 leaves) Vertex
#: was measured to 400 between ~51 and ~59 leaves, so 50 sits on the edge; 40 keeps
#: a safety margin while staying concurrent (no wall-clock cost — see aextract_schema).
#: The override is READ THROUGH BOUNDS, not raw: this constant decides call
#: fan-out and the response-schema ceiling, so a typo in the environment used to
#: be dangerous in three different directions — a non-integer took the whole API
#: module down at import, ``0`` or a negative turned a valid 1,000-leaf schema
#: into 1,000 concurrent model calls, and anything above 40 walked back through
#: the 400 that this number exists to avoid. Out-of-range or unparseable values
#: fall back to the measured default and say so, because a silently ignored
#: setting is its own failure mode.
MAX_GROUP_LEAVES = _bounded_env_int("EXTRACT_MAX_GROUP_LEAVES", default=40, low=1, high=40)
#: Per-batch context budget in characters (~4 chars/token → ~40k tokens of
#: chunk text per call). A document larger than this is split into batches
#: extracted concurrently, each against the full schema, then merged.
#: 160k (vs the older 40k) keeps cross-reference fields that straddle a batch
#: boundary in the same call (payer↔member_id, facility↔point_of_origin, NPI),
#: which measurably improves field accuracy (+~0.03) and insurance pairing
#: (+0.03) at ~free latency (decode-bound; batches fan out concurrently).
#: 80–160k is a plateau; 300k over-stuffs the context and regresses.
MAX_BATCH_CHARS = 160_000
#: Per-field evidence is per-chunk, never a union: a union of scattered supports
#: becomes a page-sized rectangle on dense forms. Cap the supporting chunks so
#: repeated boilerplate can't flood a field with boxes.
SUPPORT_MAX_CHUNKS = 16
#: A chunk shorter than this is too generic to text-match-augment a long value
#: (dates/units match all over a page).
MATCH_MIN_CHUNK_CHARS = 25
#: Coordinate scale for normalized evidence bboxes (0–1000, page-relative).
COORD_SCALE = 1000.0

_WRAPPER_KEYS = frozenset({"value", "chunks", "quote"})
#: The ``chunk_ids=False`` wrapper is exactly ``{value, quote}`` (its response
#: schema requires precisely these two keys, so exact match is safe).
_WRAPPER_KEYS_NO_IDS = frozenset({"value", "quote"})
_SUPPORTED_LEAF_TYPES = frozenset({"string", "number", "integer", "boolean"})
_REF_KEYS = frozenset({"$ref", "$dynamicRef", "$recursiveRef"})


#: Prepended ONLY when page images are attached (multimodal). Keeps the field_value
#: win of reading values off the image while preserving text grounding: the image is
#: for READING a value the OCR text dropped or garbled, never for choosing the
#: citation. Without this clause the original experiment saw values recovered but
#: citations disrupted (address fields grounded -> ungrounded).
_IMAGE_READ_GUIDANCE = (
    "\n\nPage images are attached for reference. Use them ONLY to read or correct a value the "
    "parsed text may have dropped, garbled, or mis-OCR'd — output the value as it appears on the "
    "page. Citations and grounding are UNCHANGED by the images: always cite the value's occurrence "
    "in the provided TEXT chunks (its source-of-record page and a literal text quote), exactly as "
    "you would without the images. Do not let an image change which page or quote you cite.\n\n"
)


EXTRACTION_PROMPT = """\
You extract structured fields from a parsed document. Below is the full document \
as numbered chunks. A chunk starts with `[chunk_id | page N]` and its text \
follows; a chunk tagged `[chunk_id | page N | table]` is a markdown table whose \
rows follow one per line — read it as a grid (a value belongs to its own row AND \
column; never carry a value across rows).

Fill every field of the response schema. For each leaf field return:
- "value": the field's value, using ONLY what is literally written in the document. \
Convert numbers to numbers (e.g. "65" -> 65; "100,000 steps" -> 100000) but never \
compute, infer, or guess a value that is not printed. For a CATEGORICAL/ENUM field whose \
allowed value is not printed verbatim, choose the option best supported by the page and set \
chunks/quote to the exact text you based that choice on (e.g. quote "Private Home" when \
selecting community_residential_setting); return null if nothing on the page supports any option.
- "chunks": the chunk_id(s) the value was read from.
- "quote": an EXACT substring (<= 200 chars, verbatim, contiguous) copied from one \
of those chunks, containing the value as printed PLUS enough surrounding text to be \
unique on the page. For a table cell, quote the whole row. Never quote the bare value.

If a field is NOT present in the document: value = null, chunks = [], quote = "". \
Never fabricate. An empty answer is correct when the document does not contain the field. \
A null is ALWAYS better than a plausible guess, even when the schema seems to expect a value.

If a value is MASKED or REDACTED on the page (e.g. an SSN shown as "***-**-1234", \
"XXX-XX-1234", or with hidden/asterisked characters), the true value is not legible: \
return value = null. NEVER reconstruct or fabricate masked or hidden characters — output \
only characters that are actually printed and legible.

For arrays: one entry per row/record, in document order, and include EVERY row — \
never skip, sample, or summarize rows (a 40-entry reference list means 40 entries). \
In sparse tables, a blank cell is null — NEVER fill it by inheriting a value from a \
neighboring row or column, and never copy a value from an adjacent record.

DOCUMENT:
"""

#: ``{value, quote}`` first-pass variant (plan 057 follow-up): drop the chunk-id
#: citation from the model output entirely. Grounding still works because
#: :func:`_verify_leaf` anchors by QUOTE text-match (cited ids were only ever a
#: lookup preference, never trusted). Same rules as :data:`EXTRACTION_PROMPT`
#: minus every mention of chunk ids.
EXTRACTION_PROMPT_NO_IDS = """\
You extract structured fields from a parsed document. Below is the full document \
as numbered chunks. A chunk starts with `[chunk_id | page N]` and its text \
follows; a chunk tagged `[chunk_id | page N | table]` is a markdown table whose \
rows follow one per line — read it as a grid (a value belongs to its own row AND \
column; never carry a value across rows).

Fill every field of the response schema. For each leaf field return:
- "value": the field's value, using ONLY what is literally written in the document. \
Convert numbers to numbers (e.g. "65" -> 65; "100,000 steps" -> 100000) but never \
compute, infer, or guess a value that is not printed. For a CATEGORICAL/ENUM field whose \
allowed value is not printed verbatim, choose the option best supported by the page and set \
quote to the exact text you based that choice on (e.g. quote "Private Home" when \
selecting community_residential_setting); return null if nothing on the page supports any option.
- "quote": an EXACT substring (<= 200 chars, verbatim, contiguous) copied from the \
document, containing the value as printed PLUS enough surrounding text to be \
unique on the page. For a table cell, quote the whole row. Never quote the bare value.

If a field is NOT present in the document: value = null, quote = "". \
Never fabricate. An empty answer is correct when the document does not contain the field. \
A null is ALWAYS better than a plausible guess, even when the schema seems to expect a value.

If a value is MASKED or REDACTED on the page (e.g. an SSN shown as "***-**-1234", \
"XXX-XX-1234", or with hidden/asterisked characters), the true value is not legible: \
return value = null. NEVER reconstruct or fabricate masked or hidden characters — output \
only characters that are actually printed and legible.

For arrays: one entry per row/record, in document order, and include EVERY row — \
never skip, sample, or summarize rows (a 40-entry reference list means 40 entries). \
In sparse tables, a blank cell is null — NEVER fill it by inheriting a value from a \
neighboring row or column, and never copy a value from an adjacent record.

DOCUMENT:
"""


#: BARE first pass (2026-07-01 experiment): no wrapper at all — the model fills
#: the user schema directly. No quotes requested, no grounding, no unwrapping
#: beyond JSON parsing. Same value rules as the other prompts, minus every
#: chunks/quote mention.
EXTRACTION_PROMPT_BARE = """\
You extract structured fields from a parsed document. Below is the full document \
as numbered chunks. A chunk starts with `[chunk_id | page N]` and its text \
follows; a chunk tagged `[chunk_id | page N | table]` is a markdown table whose \
rows follow one per line — read it as a grid (a value belongs to its own row AND \
column; never carry a value across rows).

Fill every field of the response schema with the field's value, using ONLY what is \
literally written in the document. Convert numbers to numbers (e.g. "65" -> 65; \
"100,000 steps" -> 100000) but never compute, infer, or guess a value that is not \
printed. For a CATEGORICAL/ENUM field whose allowed value is not printed verbatim, \
choose the option best supported by the page; return null if nothing on the page \
supports any option.

If a field is NOT present in the document: value = null. Never fabricate. An empty \
answer is correct when the document does not contain the field. A null is ALWAYS \
better than a plausible guess, even when the schema seems to expect a value.

If a value is MASKED or REDACTED on the page (e.g. an SSN shown as "***-**-1234", \
"XXX-XX-1234", or with hidden/asterisked characters), the true value is not legible: \
return value = null. NEVER reconstruct or fabricate masked or hidden characters — output \
only characters that are actually printed and legible.

For arrays: one entry per row/record, in document order, and include EVERY row — \
never skip, sample, or summarize rows (a 40-entry reference list means 40 entries). \
In sparse tables, a blank cell is null — NEVER fill it by inheriting a value from a \
neighboring row or column, and never copy a value from an adjacent record.

DOCUMENT:
"""


#: OUTPUT DIET (``EXTRACT_OMIT_NULL_LEAVES``). Appended to whichever extraction
#: prompt the call uses — wrapped or bare. A field the page does not print costs
#: the same emitted structure as one it does; letting the model skip it is the
#: one instruction that removes those tokens. The schema shape comes back in
#: :func:`_rehydrate`, so what the caller receives is unchanged.
_OMIT_NULL_INSTRUCTION = (
    "\nOMIT any field the document prints no value for — leave it out of the JSON "
    "entirely instead of emitting it with a null value.\n"
)

#: Auto-schema (plan 050 M4): cap the designed field count. Well under
#: :data:`MAX_LEAF_FIELDS` so a generated schema always passes validation.
MAX_GENERATED_FIELDS = 40
#: Design-pass context budget. Larger than one extract batch — the design output
#: is tiny (field names), so we can afford to show the model more of the document
#: so fields that only appear on later pages still get designed.
DESIGN_CONTEXT_CHARS = 120_000

SCHEMA_GEN_PROMPT = """\
You are designing a JSON extraction schema for the document below, shown as \
numbered chunks. Identify the structured facts a downstream user would most want \
to pull out of THIS document, and return them as a flat list of fields.

For each field return:
- "name": a snake_case identifier (e.g. "invoice_number", "patient_date_of_birth").
- "type": one of "string", "integer", "number", "boolean".
- "description": a short instruction naming exactly what to extract.

Return the 5-40 most useful, unambiguous fields that are ACTUALLY PRESENT in this \
document. Prefer concrete printed facts (ids, names, dates, amounts, totals) over \
derived or subjective ones. Use "string" for dates, phone numbers, and ids — do \
NOT invent a "date" type. Return only the field definitions, never their values.

DOCUMENT:
"""

#: Gemini structured-output schema for the design pass (OpenAPI dialect — types
#: uppercased, like :func:`to_gemini_schema` output).
_SCHEMA_GEN_RESPONSE = {
    "type": "OBJECT",
    "properties": {
        "fields": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "type": {
                        "type": "STRING",
                        "enum": ["string", "integer", "number", "boolean"],
                    },
                    "description": {"type": "STRING"},
                },
                "required": ["name", "type", "description"],
            },
        },
    },
    "required": ["fields"],
}


# --- Errors -----------------------------------------------------------------


class SchemaValidationError(ValueError):
    """User schema rejected at the convert-then-trust boundary → HTTP 422.

    Raised before any model call. The route layer maps every subclass to 422
    (``v1._map_schema_error``) so a bad schema fails fast and loud.
    """


class SchemaTooDeep(SchemaValidationError):
    pass


class SchemaCycle(SchemaValidationError):
    pass


class UnsupportedFieldType(SchemaValidationError):
    pass


class SchemaTooWide(SchemaValidationError):
    pass


class EmptySchema(SchemaValidationError):
    pass


# --- Model seam -------------------------------------------------------------


class SchemaModel(Protocol):
    """The injected structured-output LLM (Gemini-on-Vertex in production).

    Returns the parsed JSON object the model emitted for ``response_schema``.
    Implementations must raise on timeout / empty / non-JSON output.
    """

    async def generate_json(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]: ...


# --- Result -----------------------------------------------------------------


@dataclass
class SchemaExtractResult:
    values: dict[str, Any]
    evidence: dict[str, list[FieldEvidence]]
    ungrounded_fields: list[str] = field(default_factory=list)
    #: Diagnostics for values kept WITHOUT a quote anchor because the leaf is
    #: structurally unquotable: ``{field path: "enum" | "digits" | "zero-default" |
    #: "date-digits"}``. These paths are deliberately NOT in ``ungrounded_fields``,
    #: which keeps its meaning (a free-text value nothing on the page supports).
    verified_by: dict[str, str] = field(default_factory=dict)
    #: Reconciliation-pass accounting when that pass ran (``None`` = it did not).
    #: ``{"leaves_changed", "leaves_kept"}``, or ``{"skipped": reason}`` when the
    #: pass declined. Diagnostics only — the route stamps it onto the stage timer.
    reconcile: dict[str, Any] | None = None


@dataclass
class _IndexedChunk:
    """A non-empty parse chunk, carrying its resolved page-relative geometry."""

    text: str
    page: int
    bbox: list[float] | None  # already normalized to 0–1000 page-relative
    confidence: float | None = None  # parse-side chunk confidence, propagated to FieldEvidence
    #: The parser's own element kind. Carried so the rendered context can tell the
    #: model a table from prose (audit P0-3); never used for grounding.
    chunk_type: ChunkType = ChunkType.TEXT

    @property
    def is_table(self) -> bool:
        """Whether this chunk should be presented to the model AS a table.

        Typed ``TABLE`` chunks, plus the lenient GFM-shaped TEXT chunk — the same
        rule :func:`extract.core.assemble._gfm_text_view` already uses to treat a
        markdown-table TEXT chunk as a table. The shape check is load-bearing, not
        belt-and-braces: the pre-parsed chunks route (``/v1/extract/schema/chunks``)
        and every cached-chunk eval row send no ``chunk_type`` at all, so a real
        table arrives typed ``TEXT`` (audit P1-13).
        """
        if self.chunk_type == ChunkType.TABLE:
            return True
        # Local import: ``core`` stays free of import-time coupling between the
        # assembly and schema lanes (assemble only imports ``models``).
        from extract.core.assemble import _is_gfm_delimiter

        lines = self.text.split("\n", 2)
        return (
            len(lines) >= 2
            and lines[0].lstrip().startswith("|")
            and _is_gfm_delimiter(lines[1])
        )


@dataclass
class _Leaf:
    """A merge cell: one extracted leaf with its grounding outcome."""

    value: Any
    evidence: list[FieldEvidence]
    ungrounded: bool
    #: Set only for a value the quote ladder could not anchor but that the
    #: structurally-unquotable exemption kept anyway ("enum" / "digits" /
    #: "zero-default" / "date-digits" — see :func:`_unquotable_exemption`).
    #: Such a leaf is NOT ungrounded and carries no evidence.
    verified_by: str | None = None


@dataclass
class _VPLeaf:
    """A merge cell for the ``{value,pages}`` first pass (plan 057): the extracted
    value plus the one-based page numbers the model says it appears on. ``pages``
    are routing hints for the image localizer, never final evidence."""

    value: Any
    pages: list[int]


# --- Schema validation (the safety boundary — plan 050 D4/D5) ---------------


def validate_schema(user_schema: dict[str, Any]) -> None:
    """Reject pathological schemas before they reach the model (→ 422).

    Enforces: non-empty; nesting ≤ :data:`MAX_SCHEMA_DEPTH`; no ``$ref`` /
    cycles; only supported leaf types; ≤ :data:`MAX_LEAF_FIELDS` leaves.
    """
    if not isinstance(user_schema, dict) or not user_schema:
        raise EmptySchema("Schema must be a non-empty JSON object of fields.")
    leaves = 0
    # The user schema is a bare mapping of field name → node (our dialect), so
    # the fields themselves sit at depth 1.
    for name, node in user_schema.items():
        leaves += _validate_node(node, depth=1, path=str(name))
    if leaves == 0:
        raise EmptySchema("Schema declares no extractable leaf fields.")
    if leaves > MAX_LEAF_FIELDS:
        raise SchemaTooWide(
            f"Schema has {leaves} leaf fields, exceeds the {MAX_LEAF_FIELDS}-field cap. "
            "Split it into smaller schemas (field-group splitting is a follow-up)."
        )


def _validate_node(node: Any, *, depth: int, path: str) -> int:
    """Validate one schema node; return its leaf count."""
    if not isinstance(node, dict):
        raise UnsupportedFieldType(f"{path}: schema node must be an object, got {type(node).__name__}.")
    if _REF_KEYS & node.keys():
        raise SchemaCycle(f"{path}: $ref / recursive schemas are not supported.")
    if depth > MAX_SCHEMA_DEPTH:
        raise SchemaTooDeep(
            f"{path}: schema nesting exceeds the {MAX_SCHEMA_DEPTH}-level limit."
        )
    t = _scalar_type(node.get("type", "string"))
    if t == "object":
        props = node.get("properties")
        if not isinstance(props, dict) or not props:
            raise SchemaValidationError(f"{path}: object node needs a non-empty 'properties'.")
        return sum(
            _validate_node(v, depth=depth + 1, path=f"{path}.{k}") for k, v in props.items()
        )
    if t == "array":
        items = node.get("items")
        if not isinstance(items, dict):
            raise SchemaValidationError(f"{path}: array node needs an 'items' schema.")
        return _validate_node(items, depth=depth + 1, path=f"{path}[]")
    if t not in _SUPPORTED_LEAF_TYPES:
        raise UnsupportedFieldType(
            f"{path}: unsupported type {t!r}. Supported: "
            f"{', '.join(sorted(_SUPPORTED_LEAF_TYPES))}, object, array."
        )
    return 1


# --- Schema wrapping / conversion (ported from the prototype) ---------------


def _scalar_type(t: Any) -> Any:
    """Collapse a JSON-Schema nullable union (``["string", "null"]`` — common in
    Pydantic/Extend exports) to its base scalar. Every leaf is nullable via the
    wrapper, so the ``"null"`` member is redundant; left in, a list ``type`` would
    crash the validator (unhashable) and produce a junk Gemini type."""
    if isinstance(t, list):
        return next((x for x in t if x != "null"), "string")
    return t


def _clean_enum(enum: Any) -> list[Any] | None:
    """Drop ``null`` members from an enum. Vertex rejects a null in a string enum
    (400 INVALID_ARGUMENT); nullability is carried by the wrapper instead. Returns
    ``None`` when nothing usable remains."""
    if not isinstance(enum, list):
        return None
    vals = [e for e in enum if e is not None]
    return vals or None


def wrap_schema(node: dict[str, Any], *, chunk_ids: bool = True,
                omit_null: bool = False) -> dict[str, Any]:
    """Wrap every leaf as ``{value, chunks, quote}`` so each value carries its
    evidence indices + grounding quote. With ``chunk_ids=False`` the leaf is
    ``{value, quote}`` — the quote alone grounds the value (plan 057 follow-up).
    ``omit_null=True`` marks every OBJECT's properties explicitly optional
    (``required: []``) so a leaf with no printed value may be left out entirely
    (the output diet — the shape is rehydrated after parsing). The leaf wrapper
    itself keeps its own ``required``: an emitted leaf is always complete.
    Assumes a validated node."""
    t = _scalar_type(node.get("type", "string"))
    if t == "object":
        obj: dict[str, Any] = {
            "type": "object",
            "properties": {k: wrap_schema(v, chunk_ids=chunk_ids, omit_null=omit_null)
                           for k, v in node.get("properties", {}).items()},
        }
        if omit_null:
            obj["required"] = []
        return obj
    if t == "array":
        return {"type": "array",
                "items": wrap_schema(node.get("items", {"type": "string"}),
                                     chunk_ids=chunk_ids, omit_null=omit_null)}
    leaf: dict[str, Any] = {"type": t, "nullable": True}
    enum = _clean_enum(node.get("enum"))
    if enum is not None:
        leaf["enum"] = enum
    props: dict[str, Any] = {"value": leaf}
    if chunk_ids:
        props["chunks"] = {"type": "array", "items": {"type": "integer"}}
    props["quote"] = {"type": "string"}
    return {
        "type": "object",
        "description": node.get("description", ""),
        "properties": props,
        "required": list(props),
    }


def to_gemini_schema(node: dict[str, Any]) -> dict[str, Any]:
    """Uppercase types to the OpenAPI dialect Gemini structured output wants."""
    out: dict[str, Any] = {}
    for key, val in node.items():
        if key == "type":
            out["type"] = str(_scalar_type(val)).upper()
        elif key == "properties":
            out["properties"] = {k: to_gemini_schema(v) for k, v in val.items()}
        elif key == "items":
            out["items"] = to_gemini_schema(val)
        elif key == "enum":
            cleaned = _clean_enum(val)
            if cleaned is not None:
                out["enum"] = cleaned
        elif key in ("required", "description", "nullable"):
            out[key] = val
    return out


def bare_schema(node: dict[str, Any], *, omit_null: bool = False) -> dict[str, Any]:
    """No-wrapper leaf (2026-07-01 bare experiment): the model returns the value
    directly — nullable leaf, no ``{value, quote}`` envelope anywhere.
    ``omit_null=True`` makes every object's properties explicitly optional, as in
    :func:`wrap_schema`."""
    t = _scalar_type(node.get("type", "string"))
    if t == "object":
        obj: dict[str, Any] = {
            "type": "object",
            "properties": {k: bare_schema(v, omit_null=omit_null)
                           for k, v in node.get("properties", {}).items()},
        }
        if omit_null:
            obj["required"] = []
        return obj
    if t == "array":
        return {"type": "array",
                "items": bare_schema(node.get("items", {"type": "string"}), omit_null=omit_null)}
    leaf: dict[str, Any] = {"type": t, "nullable": True,
                            "description": node.get("description", "")}
    enum = _clean_enum(node.get("enum"))
    if enum is not None:
        leaf["enum"] = enum
    return leaf


def build_response_schema(user_schema: dict[str, Any], *, chunk_ids: bool = True,
                          wrapped: bool = True,
                          omit_null: bool = False) -> dict[str, Any]:
    """User schema → wrapped → Gemini response_schema (the full pipeline).
    ``wrapped=False`` skips the ``{value, quote}`` envelope entirely (bare mode).
    ``omit_null=True`` (the output diet) makes every field explicitly optional so
    an absent field may be omitted from the response instead of emitted as null."""
    builder = (
        (lambda v: wrap_schema(v, chunk_ids=chunk_ids, omit_null=omit_null))
        if wrapped
        else (lambda v: bare_schema(v, omit_null=omit_null))
    )
    tree: dict[str, Any] = {
        "type": "object",
        "properties": {k: builder(v) for k, v in user_schema.items()},
    }
    if omit_null:
        tree["required"] = []
    return to_gemini_schema(tree)


# --- Standard-JSON-Schema flattener (plan 050 M4) ---------------------------
# Accept the schemas real callers actually generate — Pydantic
# ``model_json_schema()`` and zod-to-json-schema emit ``$ref`` + ``$defs`` +
# ``anyOf`` (for Optional) + ``allOf`` + validation metadata that our bare
# ``{name: node}`` dialect doesn't use. Normalize them in before validation,
# resolving non-recursive refs and still rejecting genuine cycles.

#: JSON-Schema keys we drop when normalizing to our dialect: validation-only
#: constraints and provenance metadata that don't affect extraction.
_DROPPED_SCHEMA_KEYS = frozenset(
    {
        "title", "default", "examples", "example", "$schema", "$id", "$comment",
        "additionalProperties", "unevaluatedProperties", "readOnly", "writeOnly",
        "deprecated", "pattern", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems",
        "uniqueItems", "minProperties", "maxProperties", "multipleOf", "const",
        "discriminator", "required", "propertyNames", "patternProperties",
    }
)


def _is_null_schema(node: Any) -> bool:
    """A union member that just expresses 'null' (Pydantic Optional → anyOf[T, null])."""
    return isinstance(node, dict) and node.get("type") == "null"


def _clean_node_keys(node: dict[str, Any]) -> dict[str, Any]:
    """Drop metadata/validation keys, keep only what our dialect/wrap consumes.

    ``format`` is KEPT (it used to be dropped with the other validation-only keys):
    a leaf declaring ``{"type": "string", "format": "date"}`` is declaring that its
    value is a NORMALIZED date, which is exactly the contract the verification-gate
    exemption ladder reads (:func:`_unquotable_exemption`). It reaches no model —
    :func:`wrap_schema` and :func:`to_gemini_schema` both build leaves from a fixed
    key whitelist — so keeping it changes only what the verifier can see.
    """
    return {k: v for k, v in node.items() if k not in _DROPPED_SCHEMA_KEYS}


def _merge_resolved(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two resolved nodes (allOf members). Properties union; b wins scalars."""
    out = dict(a)
    for k, v in b.items():
        if k == "properties" and isinstance(out.get("properties"), dict) and isinstance(v, dict):
            out["properties"] = {**out["properties"], **v}
        else:
            out[k] = v
    return out


def _resolve_node(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> Any:
    """Normalize one schema node: inline $ref (cycle-checked), unwrap nullable
    anyOf/oneOf, merge allOf, drop metadata, recurse into object/array."""
    if not isinstance(node, dict):
        return node  # let validate_schema reject non-object nodes with a clear path

    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        if name in seen:
            raise SchemaCycle(f"$ref cycle through {name!r} is not supported.")
        target = defs.get(name)
        if target is None:
            raise SchemaValidationError(f"Unresolved $ref {ref!r} (no matching $defs entry).")
        resolved = _resolve_node(target, defs, seen + (name,))
        siblings = _clean_node_keys({k: v for k, v in node.items() if k != "$ref"})
        return {**resolved, **siblings} if isinstance(resolved, dict) else resolved

    if "allOf" in node:
        merged: dict[str, Any] = {}
        for member in node["allOf"]:
            part = _resolve_node(member, defs, seen)
            if isinstance(part, dict):
                merged = _merge_resolved(merged, part)
        rest = _clean_node_keys({k: v for k, v in node.items() if k != "allOf"})
        return _merge_resolved(merged, _resolve_node(rest, defs, seen) if rest else {})

    for union_key in ("anyOf", "oneOf"):
        if union_key in node:
            members = [m for m in node[union_key] if not _is_null_schema(m)]
            if not members:  # only null(s) — degenerate; treat as a free string
                return {"type": "string"}
            chosen = _resolve_node(members[0], defs, seen)  # first non-null branch
            if isinstance(chosen, dict) and "description" in node:
                chosen = {**chosen, "description": node["description"]}
            return chosen

    out = _clean_node_keys(node)
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out["properties"] = {k: _resolve_node(v, defs, seen) for k, v in out["properties"].items()}
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _resolve_node(out["items"], defs, seen)
    return out


def flatten_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a standard JSON Schema into our bare ``{name: node}`` dialect.

    Handles Pydantic ``model_json_schema()`` / zod output: resolves non-recursive
    ``$ref`` against ``$defs``/``definitions``, unwraps a nullable ``anyOf``/``oneOf``
    (``Optional[T]`` → ``T``; every leaf is nullable in our pipeline anyway), merges
    ``allOf``, and drops validation-only metadata. A self-referential schema (a
    ``$ref`` cycle) still raises :class:`SchemaCycle`. A schema already in our
    dialect passes through unchanged (idempotent).
    """
    if not isinstance(raw, dict) or not raw:
        raise EmptySchema("Schema must be a non-empty JSON object of fields.")
    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        d = raw.get(key)
        if isinstance(d, dict):
            defs.update(d)

    root = raw
    if isinstance(root.get("$ref"), str):  # top-level is itself a ref to the model
        root = _resolve_node({"$ref": root["$ref"]}, defs, ())
        if not isinstance(root, dict):
            raise SchemaValidationError("Top-level $ref did not resolve to an object schema.")

    # Standard object schema → its properties are our field mapping; otherwise the
    # mapping IS our bare dialect (minus any sibling $defs/definitions).
    if isinstance(root.get("properties"), dict) and root.get("type", "object") == "object":
        fields = root["properties"]
    else:
        fields = {k: v for k, v in root.items() if k not in ("$defs", "definitions")}
    if not fields:
        raise EmptySchema("Schema declares no fields.")
    return {name: _resolve_node(node, defs, ()) for name, node in fields.items()}


# --- Context rendering + batching -------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_bbox(
    bbox: list[float] | None, page_size: tuple[float, float] | None
) -> list[float] | None:
    """Absolute PDF points (top-left origin) → 0–1000 page-relative, clipped."""
    if bbox is None or page_size is None:
        return None
    w, h = page_size
    if not w or not h:
        return None
    x0, y0, x1, y1 = bbox
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    scale = COORD_SCALE
    return [
        max(0.0, min(scale, x0 / w * scale)),
        max(0.0, min(scale, y0 / h * scale)),
        max(0.0, min(scale, x1 / w * scale)),
        max(0.0, min(scale, y1 / h * scale)),
    ]


#: A run of HORIZONTAL whitespace (spaces/tabs/CR — never the newline). Chunk text
#: is run-collapsed per LINE so structure survives; see :func:`normalize_chunk_text`.
_HSPACE_RUN = re.compile(r"[^\S\n]+")
#: 3+ consecutive newlines → one blank line (paragraph separation, not padding).
_BLANK_RUN = re.compile(r"\n{3,}")


def normalize_chunk_text(raw: str | None) -> str:
    """Collapse horizontal whitespace runs; PRESERVE line structure.

    This was ``re.sub(r"\\s+", " ", ...)`` until 2026-07-30 — a whole-text
    whitespace collapse that folded every TABLE chunk's GFM markdown into a
    single line of pipe soup before the extraction model ever saw it, on every
    schema-extract request and on all three schema paths (post-model pipeline
    audit, finding P0-3). A newline in a parse chunk is STRUCTURE — a table row
    boundary, a paragraph break — not whitespace, so it is kept. Horizontal runs
    (the OCR's column padding) still collapse, per line, and a line's leading /
    trailing spaces are stripped, so single-line prose is byte-identical to the
    old behaviour.
    """
    lines = [_HSPACE_RUN.sub(" ", ln).strip() for ln in (raw or "").split("\n")]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def _index_chunks(
    chunks: list[Chunk], page_sizes: list[tuple[float, float]]
) -> list[_IndexedChunk]:
    """Drop empty-text chunks and resolve each to page-relative geometry.

    Original list position is not preserved as the model-facing id: indices are
    re-based per batch (the model only ever sees a contiguous 0..n slice), so a
    chunk's identity is its position within its batch.

    ``chunk_type`` rides along (audit P0-3) so :func:`render_context` can tell the
    model which chunks are tables — a table's text is rendered as the markdown
    grid it is, rows intact.
    """
    indexed: list[_IndexedChunk] = []
    for c in chunks:
        text = normalize_chunk_text(c.page_content)
        if not text:
            continue
        page = c.page_no
        page_size = page_sizes[page - 1] if 0 < page <= len(page_sizes) else None
        indexed.append(_IndexedChunk(text=text, page=page, bbox=normalize_bbox(c.bbox, page_size),
                                     confidence=c.confidence, chunk_type=c.chunk_type))
    return indexed


def _batch(indexed: list[_IndexedChunk]) -> list[list[_IndexedChunk]]:
    """Split chunks into context-budget batches, never breaking a chunk."""
    batches: list[list[_IndexedChunk]] = []
    current: list[_IndexedChunk] = []
    size = 0
    for c in indexed:
        # +24 ≈ the "[i | page N] " framing per line.
        cost = len(c.text) + 24
        if current and size + cost > MAX_BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(c)
        size += cost
    if current:
        batches.append(current)
    return batches


#: A chunk-transform hook (additive, default unused). Given a batch's
#: ``_IndexedChunk`` and its in-batch index, it returns the text the model should
#: read for that chunk — used by the layout recipe to prefix each chunk with its
#: (x%,y%) position + [SECTION] tag so the model can disambiguate 2-column fields.
#: It rewrites ONLY the rendered text; the chunk's page/bbox geometry (and thus all
#: downstream citations) are untouched. ``None`` → byte-identical default rendering.
ChunkTransform = Callable[[int, "_IndexedChunk"], str]


def render_context(
    batch: list[_IndexedChunk], chunk_transform: ChunkTransform | None = None
) -> str:
    """Indexed-chunk serialization the model reads: ``[i | page N] text``.

    A TABLE chunk is tagged ``[i | page N | table]`` and its markdown grid follows
    on the NEXT line, one row per line — that is the same text the ``content`` /
    ``segments`` surfaces ship (``chunking._render_content`` concatenates a table
    chunk's markdown verbatim), so the model reads the table as a table. Any
    multi-line chunk gets the same header-then-body framing; a single-line chunk
    is byte-identical to the historical ``[i | page N] text``.

    ``chunk_transform`` (additive, default ``None``) rewrites only the per-chunk
    TEXT (e.g. to inject the layout (x%,y%) + [SECTION] prefixes); the framing and
    the chunk's page/bbox geometry are unchanged.
    """
    def _text(i: int, c: _IndexedChunk) -> str:
        return chunk_transform(i, c) if chunk_transform is not None else c.text

    lines: list[str] = []
    for i, c in enumerate(batch):
        head = f"[{i} | page {c.page} | table]" if c.is_table else f"[{i} | page {c.page}]"
        body = _text(i, c)
        lines.append(f"{head}\n{body}" if "\n" in body else f"{head} {body}")
    return "\n".join(lines)


# --- Grounding verification (prototype + TS refinements) --------------------


#: A value at least this long is allowed to ground by verbatim containment when
#: the model's quote doesn't match a chunk (the TS short-value salvage). Below
#: it, a bare value substring-matches too much to prove anything, so we require
#: a real quote match — and treat a no-match as a suspected hallucination.
_MIN_CONTAINMENT_CHARS = 8


def _find_chunk(batch: list[_IndexedChunk], cited: list[int], pred) -> int | None:
    """First chunk index satisfying ``pred`` — cited ids first, then any chunk."""
    for i in cited:
        if pred(batch[i]):
            return i
    for i in range(len(batch)):
        if pred(batch[i]):
            return i
    return None


def _page_window_anchor(batch: list[_IndexedChunk], q: str, nv: str) -> int | None:
    """Page-window quote salvage (2026-07-01, plan 057 retrial).

    On line-level parses (raw OCR blocks, median ~20 chars) a quote that follows
    the prompt's "value PLUS surrounding text" rule spans several chunks and can
    never substring-match any single one — which nulled ~27%% of the both-failed
    fields despite correct extraction. Match the normalized quote against each
    page's concatenated chunk text instead; anchor to the value-bearing chunk
    inside the matched span (else the span's first chunk). The quote is still
    required to be real verbatim page text, so the anti-hallucination property
    is unchanged, and it pins the occurrence so even short values ground safely.
    """
    by_page: dict[int, list[int]] = {}
    for i, c in enumerate(batch):
        by_page.setdefault(c.page, []).append(i)
    for _, idxs in sorted(by_page.items()):
        spans: list[tuple[int, int, int]] = []
        parts: list[str] = []
        off = 0
        for i in idxs:
            t = _norm(batch[i].text)
            spans.append((off, off + len(t), i))
            parts.append(t)
            off += len(t) + 1  # single-space joiner, mirrors _norm whitespace collapse
        start = " ".join(parts).find(q)
        if start < 0:
            continue
        end = start + len(q)
        overlapped = [i for a, b, i in spans if a < end and b > start]
        if not overlapped:
            continue
        for i in overlapped:
            if nv and nv in _norm(batch[i].text):
                return i
        return overlapped[0]
    return None


def _digits(s: str) -> str:
    """Every digit of ``s``, in order, separators and letters dropped."""
    return re.sub(r"\D", "", s)


def _canonical_number_str(value: Any) -> str:
    """A number's MINIMAL printed form: ``59900.0`` → "59900", ``42.50`` → "42.5".

    ``str()`` on a float carries a repr artifact — ``str(59900.0)`` is "59900.0",
    whose trailing "0" the anchor rungs below then demand exist on the page. A
    receipt prints "59.900" and the money value was strict-nulled for the artifact
    (bench: 7 printed money values lost this way, together with the digit-only
    string class). Fixed-point is used only when it round-trips, so an exponent-
    range float still falls back to ``str``.
    """
    if not isinstance(value, float):
        return str(value)
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    s = f"{value:.10f}".rstrip("0").rstrip(".")
    try:
        return s if s and float(s) == value else str(value)
    except ValueError:  # pragma: no cover — defensive
        return str(value)


def _value_text(value: Any) -> str:
    """The value as text for matching against the page: numbers canonicalized
    (see :func:`_canonical_number_str`), everything else verbatim."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return _canonical_number_str(value)


def _canonical_digits(value: Any) -> str:
    """Digits of a numeric value with separators and trailing decimal zeros dropped:
    ``19``/``19.0`` → "19", ``1.000`` → "1", ``2.50`` → "25"."""
    return _digits(_canonical_number_str(value))


def _boundary_run_pattern(text: str) -> re.Pattern[str] | None:
    """``text``'s alphanumerics, in order, tolerant of printed separators and
    anchored on both sides so a match can never sit INSIDE a longer alnum run.

    That anchoring is the digit-drop protection: an id with a dropped digit
    (11001151763) is a substring of the printed run (110011517633) and must never
    ground against it. Shared by the separator-tolerant ladder rung and the
    digit-only-string exemption, which is the same match without the length floor.
    """
    toks = re.findall(r"[A-Za-z0-9]+", text)
    if not toks:
        return None
    # tight inside a token (id chars may be OCR-spaced), roomier between tokens
    # (cell borders / pipes / slashes sit between printed words)
    tok_pats = [r"[\W_]{0,2}".join(re.escape(c) for c in t) for t in toks]
    return re.compile(
        r"(?<![A-Za-z0-9])" + r"[\W_]{0,8}".join(tok_pats) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _sep_tolerant_anchor(
    batch: list[_IndexedChunk],
    cited: list[int],
    value: Any,
    *,
    canonical_numbers: bool = False,
) -> int | None:
    """Separator-tolerant containment salvage (last rung).

    The model reformats separators of values it read correctly — ``A / B`` for a
    printed ``A/B``, spaced or hyphenated ids — and the verbatim rungs above then
    strict-null a value that IS on the page (measured: coverage names slash-joined
    from adjacent cells). Match the value's alphanumeric characters in order,
    allowing up to 3 non-alnum chars between them, anchored on both sides so a
    match can never extend a longer alnum run (no digit-run false grounding).

    ``canonical_numbers`` tokenizes the value from its CANONICAL text
    (:func:`_value_text`) so a float's repr artifact — ``str(59900.0)`` is
    "59900.0" — does not become a token the page has to carry. It is PER
    REQUEST, and off by default, because it is not a repair with no downside:
    it decides whether a leaf grounds, so turning it on changes which values
    survive strict mode. On the receipt lane it recovers 7 printed money values
    the artifact was nulling; for a customer who has not screened it, today's
    answers are today's answers. It travels with the gate exemptions
    (``exempt_unquotable``) because it is the same decision — this customer's
    documents need the deployment's strict gate read less literally.
    """
    text = _value_text(value) if canonical_numbers else str(value)
    toks = re.findall(r"[A-Za-z0-9]+", text)
    if not (_MIN_CONTAINMENT_CHARS <= sum(len(t) for t in toks) <= 200):
        return None
    pat = _boundary_run_pattern(text)
    if pat is None:
        return None
    anchor = _find_chunk(batch, cited, lambda c: bool(pat.search(c.text)))
    if anchor is not None:
        return anchor
    # Cross-cell composites ("Payer / Plan" joined from ADJACENT table cells)
    # span chunk boundaries, so no single chunk matches. Search each page's
    # joined chunk text (same construction as _page_window_anchor); anchor to
    # the first chunk the match overlaps. Still boundary-anchored + in-order —
    # a fabricated value cannot match.
    by_page: dict[int, list[int]] = {}
    for i, c in enumerate(batch):
        by_page.setdefault(c.page, []).append(i)
    for _, idxs in sorted(by_page.items()):
        spans: list[tuple[int, int, int]] = []
        parts: list[str] = []
        off = 0
        for i in idxs:
            t = batch[i].text
            spans.append((off, off + len(t), i))
            parts.append(t)
            off += len(t) + 1
        m = pat.search(" ".join(parts))
        if not m:
            continue
        for a, b, i in spans:
            if a < m.end() and b > m.start():
                return i
    return None


#: Checkbox token vocabularies (wedge post-mortem S1, 2026-07-16), mirroring
#: the FROZEN eval ruler (evals2/core/metrics/checkboxes.py GLYPH_CHECKED /
#: _MARK_STRIP, cbx-v3.1) — the ruler is frozen, so it is the reference, not
#: the consumer; keep in sync when the parse vocabulary grows. Two tiers:
#: the QUOTE tier is liberal (model quotes may use paren variants for a
#: printed box), the SOURCE tier is strict (what counts as a checkbox token in
#: parse text: guarded brackets + ballot glyphs — prose "(X)"/"( )" is NOT a
#: checkbox). Bracket markers carry word-boundary guards ("X 4"/"DX"/"buf[ ]"
#: are not markers).
_CHECKBOX_MARKED_Q = re.compile(r"(?<!\w)(?:\[\s*[xX✓✔]\s*\]|\(\s*[xX✓✔]\s*\))(?!\w)|[☑☒✓✔]")
_CHECKBOX_MARKED_SRC = re.compile(r"(?<!\w)\[\s*[xX✓✔]\s*\](?!\w)|[☑☒✓✔]")
_CHECKBOX_EMPTY = re.compile(r"(?<!\w)\[\s*\](?!\w)|☐")


def _strip_vs(s: str) -> str:
    """Drop emoji variation selectors so ☑️ (U+2611 U+FE0F) matches ☑."""
    return s.replace("︎", "").replace("️", "")


def _norm_checkbox(s: str) -> str:
    s = _strip_vs(s)
    return _norm(_CHECKBOX_EMPTY.sub("[ ]", _CHECKBOX_MARKED_Q.sub("[x]", s)))


#: Rung-2 ceiling: a cited chunk may anchor a quote-less boolean only when it
#: carries at most this many checkbox tokens. Per-option chunks ("[x] Single")
#: and self-contained pairs ("[x] Yes [ ] No") qualify; a section mega-chunk
#: (dozens of boxes) proves nothing about any single field and is exactly where
#: fabricated/mis-associated states would false-ground, so it stays ungrounded.
_CHECKBOX_ANCHOR_MAX_TOKENS = 2


def _checkbox_quote_anchor(
    batch: list[_IndexedChunk], cited: list[int], quote: str, value: bool
) -> int | None:
    """Checkbox rung 1 — glyph-normalized, polarity-checked quote match (S1).

    ``str(True/False)`` is shorter than ``_MIN_CONTAINMENT_CHARS``, so every
    value-salvage rung is skipped for booleans: grounding was quote-match-or-
    death, and a checkbox state whose quote used a ☑/(x) glyph variant of the
    chunk's printed [x] strict-nulled even when the parse read the box
    correctly. Retry the quote match with checkbox tokens normalized on both
    sides, requiring:

    - the normalized quote is at least ``_MIN_CONTAINMENT_CHARS`` long — a
      whitespace or bare-token quote ("☑" → "[x]", 3 chars) substring-matches
      any marked box anywhere and proves nothing;
    - the quote's own tokens do not CONTRADICT the value: a ``true`` needs at
      least one marked token when any token is present, a ``false`` at least
      one empty token (a token-free or mixed-row quote is polarity-neutral,
      matching the pre-existing quote-rung semantics for strings).

    The matched chunk's own text is emitted as evidence (via_quote stays
    False at the call site) so evidence remains source-verbatim even when the
    model's quote used a different glyph form.
    """
    qq = _norm_checkbox(quote) if quote else ""
    if len(qq) < _MIN_CONTAINMENT_CHARS:
        return None
    marked, empty = qq.count("[x]"), qq.count("[ ]")
    if (marked or empty) and not (marked if value else empty):
        return None
    return _find_chunk(batch, cited, lambda c: qq in _norm_checkbox(c.text))


def _checkbox_token_anchor(batch: list[_IndexedChunk], cited: list[int], value: bool) -> int | None:
    """Checkbox rung 2 — no-quote anchor to a small, uniformly-agreeing cited chunk.

    One measured packet lost its entire 40-field demographics section to
    quote-less strict nulls even though the parse emitted every ``[x]`` token
    correctly (post-mortem S1). Recover by anchoring to the first CITED chunk
    whose SOURCE checkbox tokens (strict vocabulary — prose "(X)" is not a
    token) are (a) at most ``_CHECKBOX_ANCHOR_MAX_TOKENS`` and (b) UNIFORMLY
    consistent with the value: marked-only for ``true``, empty-only for
    ``false``. Per-option chunks ("[x] Single") qualify; mixed pairs
    ("[x] Yes [ ] No") do not — token presence there cannot distinguish the
    polarities, so both would ground from the same evidence.

    Deliberately NOT recovered: any non-empty quote (a failed real quote or a
    fabricated one must not fall through to token grounding — enforced at the
    call site), uncited chunks, checkbox-free prose, contradicting or mixed
    tokens, and section mega-chunks — where fabricated and mis-associated
    states live (post-mortem F1a/F4/S2). Known limit: this proves the cited
    chunk shows a box in the claimed state, not that the box belongs to THIS
    field — option-level proof needs the field label threaded into the verify
    layer (follow-up), and the pre-existing quote rung shares the limit.
    """
    for i in cited:
        text = _strip_vs(batch[i].text)
        marked = len(_CHECKBOX_MARKED_SRC.findall(text))
        empty = len(_CHECKBOX_EMPTY.findall(text))
        if not (marked + empty) or marked + empty > _CHECKBOX_ANCHOR_MAX_TOKENS:
            continue
        if (marked and not empty) if value else (empty and not marked):
            return i
    return None


# --- Structurally-unquotable leaves (EXP-A, 2026-08-05) ---------------------
# The ladder above asks one question: does this value appear on the page? For whole
# CLASSES of leaf that question has no answer, because the schema itself asks for
# something the page never prints:
#
#   - ENUM leaves. The extraction prompt already tells the model, for a categorical
#     field whose allowed value is not printed verbatim, to choose the option best
#     supported by the page — and then the gate nulled it for not being quotable.
#     "electronic_invoice" is on no receipt ever printed.
#   - NORMALIZED / COMPOSED values: an ISO ``issue_date`` composed from a printed
#     "26/04/29", a 24h ``issue_time`` from an am/pm clock, a numeric ``rate_pct``
#     of 19 from a printed "19%", a quantity 1 from a printed "1.000".
#   - ZERO DEFAULTS: 0 on a line that simply carries no discount.
#   - DIGIT-ONLY STRINGS (sku / PLU / EAN): quotable in principle, but every
#     salvage rung excludes them (the ladder's own digit-run false-grounding guard
#     plus the 8-char containment floor), so in practice they are unquotable too.
#
# Measured (EXP-A, offline, receipts): of 607 strict-mode rejections only 30 were
# real fabrications — the other 577 destroyed CORRECT answers. The serving pipeline
# trace agrees: 311 of 323 nulls destroyed correct values, including 7 printed money
# values lost to the two mechanisms fixed alongside this ladder (the digit-only
# string class above, and the float repr artifact — see _canonical_number_str).
#
# So these classes are exempt from nulling. The exemption is a correctness fix to
# verification policy, not a customer flag, but it is conservative in three ways:
#
#   1. it is consulted LAST, only after the whole ladder has failed, so every value
#      that grounds today still grounds today, with the same evidence;
#   2. every exemption but the enum one still checks the value against the printed
#      text (its digits must be there) — the enum one checks membership in the
#      declared enum, which is what makes it schema-valid rather than free text;
#   3. an exempted leaf carries NO evidence and is reported in its own diagnostics
#      map (``verified_by``), never laundered into ``ungrounded_fields``.
#
# Known limit, stated rather than hidden: a digit check is weaker than a quote. A
# one- or two-digit value ("19", "1") verifies against any line whose digits contain
# it, so this rung proves the digits are printed on the page, not that they were
# printed for THIS field — the same limit the checkbox token rung carries. It is
# scoped to leaves the SCHEMA declares numeric or date/time for that reason.


def _is_subsequence(needle: str, hay: str) -> bool:
    """``needle``'s characters appear in ``hay`` in order (not necessarily adjacent)."""
    it = iter(hay)
    return all(c in it for c in needle)


#: JSON-Schema ``format`` values that declare a normalized date/time contract.
_DATETIME_FORMATS = frozenset({"date", "date-time", "datetime", "time"})
#: Fallback for the configured key set when settings are unavailable (core stays
#: importable without the app config).
_DEFAULT_NORMALIZED_DATE_KEYS = "issue_date,issue_time"


def _normalized_date_keys() -> frozenset[str]:
    try:
        from extract.config import settings

        raw = settings.EXTRACT_NORMALIZED_DATE_KEYS
    except Exception:  # noqa: BLE001 — core must not require the app config
        raw = _DEFAULT_NORMALIZED_DATE_KEYS
    return frozenset(k.strip().lower() for k in (raw or "").split(",") if k.strip())


def _declares_normalized_datetime(schema: Any, key: str | None) -> bool:
    """Whether this leaf declares a date/time NORMALIZATION contract — by its
    JSON-Schema ``format``, or by being one of the configured key names
    (``EXTRACT_NORMALIZED_DATE_KEYS``)."""
    if isinstance(schema, dict) and str(schema.get("format", "")).lower() in _DATETIME_FORMATS:
        return True
    return bool(key) and key.lower() in _normalized_date_keys()


def _datetime_digit_forms(value: str) -> list[str]:
    """The digit strings a source line could legitimately print for this normalized
    date/time: the value's own digits, plus the two-digit-year form of an ISO date
    ("2026-04-29" → 260429) and the 12-hour form of a 24-hour time ("14:30" → 230).

    These are the only two transforms in the value space — a century the schema adds
    and a clock convention it fixes — so the set stays small and enumerable.
    """
    forms = {_digits(value)}
    if re.match(r"\s*\d{4}-\d{2}-\d{2}", value):
        forms.add(_digits(value)[2:])
    m = re.match(r"\s*(\d{1,2}):(\d{2})", value)
    if m and int(m.group(1)) <= 23:
        h12 = int(m.group(1)) % 12 or 12
        forms.add(_digits(f"{h12}:{value.split(':', 1)[1]}"))
    return [f for f in forms if f]


def _enum_member(value: Any, schema: Any) -> bool:
    """The leaf declares an enum AND this value is one of its members. Membership is
    re-checked here rather than assumed: the response schema enforces it at the
    provider, but the exemption should stand on its own evidence."""
    enum = _clean_enum(schema.get("enum")) if isinstance(schema, dict) else None
    return enum is not None and value in enum


def _declared_numeric(value: Any, schema: Any) -> bool:
    """The SCHEMA declares this leaf number/integer and the value really is one.
    ``bool`` is excluded (it subclasses int, and booleans have their own rungs)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(schema, dict) and _scalar_type(schema.get("type", "")) in (
        "number",
        "integer",
    )


def _unquotable_exemption(
    value: Any, schema: Any, key: str | None, batch: list[_IndexedChunk]
) -> str | None:
    """Why an un-anchorable value is nonetheless kept — or ``None`` to null it.

    Consulted only after every rung of the grounding ladder has failed. Without a
    schema node (callers that verify without one) nothing is exempt, so behaviour
    there is byte-identical to before.
    """
    if _enum_member(value, schema):
        return "enum"
    if _declared_numeric(value, schema):
        if value == 0:
            # A no-discount line prints no "0" to quote; the zero IS the absence.
            return "zero-default"
        d = _canonical_digits(value)
        if d and any(d in _digits(c.text) for c in batch):
            return "digits"
    elif isinstance(value, str) and re.fullmatch(r"\d+", value):
        # DIGIT-ONLY STRINGS (sku / PLU / EAN class: "9240", "090465"). Every rung of
        # the ladder excludes them BY CONSTRUCTION: they are usually shorter than
        # _MIN_CONTAINMENT_CHARS, the plain-containment rung explicitly skips
        # digit-only values, and the separator-tolerant rung applies the same floor —
        # so a correctly-read sku could only survive on a verbatim quote.
        #
        # This is that rung with the length floor lifted and NOTHING else relaxed:
        # the same boundary-anchored, separator-tolerant match, so the digit-drop
        # protection still holds (11001151763 does not ground on a printed
        # 110011517633). That is why it is stricter than the numeric rung above,
        # where trailing-zero formatting ("42.50" for 42.5) rules out a right anchor.
        pat = _boundary_run_pattern(value)
        if pat is not None and any(pat.search(c.text) for c in batch):
            return "digits"
    if _declares_normalized_datetime(schema, key) and isinstance(value, str):
        forms = _datetime_digit_forms(value)
        for c in batch:
            for line in c.text.splitlines():
                ld = _digits(line)
                if ld and any(_is_subsequence(f, ld) for f in forms):
                    # A single printed LINE carries the value's digits in order —
                    # the composition ("26/04/29" → 2026-04-29) is visible in one
                    # place, not assembled from scattered numbers.
                    return "date-digits"
    return None


def _schema_child(schema: Any, key: str) -> Any:
    """Child node of a user-schema object node (``None`` when unknown)."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    return props.get(key) if isinstance(props, dict) else None


def _schema_items(schema: Any) -> Any:
    """Item node of a user-schema array node (``None`` when unknown)."""
    items = schema.get("items") if isinstance(schema, dict) else None
    return items if isinstance(items, dict) else None


def _verify_leaf(
    node: dict[str, Any],
    batch: list[_IndexedChunk],
    strict: bool,
    *,
    schema: Any = None,
    key: str | None = None,
    canonical_numbers: bool = False,
) -> _Leaf:
    """Turn one ``{value, chunks, quote}`` wrapper into a merge cell.

    Grounding is decided by the QUOTE (or, for long-enough values, verbatim
    containment) actually matching a chunk — never by the model merely citing a
    chunk id, with ONE scoped exception: checkbox-state booleans, whose value
    ("true"/"false") is unquotable, may anchor to a cited chunk carrying at
    most two checkbox tokens that agree with the value (see
    :func:`_checkbox_anchor`). Any other non-null value with neither a quote
    match nor a containment match is a suspected fabrication: nulled under
    ``strict``, kept-but-flagged otherwise — UNLESS the leaf's own schema node
    (``schema``/``key``, threaded from the user schema) says the value is
    structurally unquotable: a schema enum, a normalized number, a zero default,
    a composed date/time (see :func:`_unquotable_exemption`). Those are kept and
    reported via ``_Leaf.verified_by`` instead. Evidence boxes come only from
    chunks that genuinely contain the value/quote, plus per-line text-match
    augmentation for long prose.
    """
    value = node.get("value")
    quote = node.get("quote") or ""
    if value is None or value == "":
        return _Leaf(value=None, evidence=[], ungrounded=False)

    n = len(batch)
    cited = [i for i in (node.get("chunks") or []) if type(i) is int and 0 <= i < n]
    q = _norm(quote)
    nv = _norm(str(value))

    anchor: int | None = None
    anchor_via_quote = False
    if q:  # prototype: the quote must be a real substring of a chunk
        anchor = _find_chunk(batch, cited, lambda c: q in _norm(c.text))
        anchor_via_quote = anchor is not None
    if anchor is None and q:  # page-window salvage: multi-line quote on a line-level parse
        anchor = _page_window_anchor(batch, q, nv)
        anchor_via_quote = anchor is not None
    if anchor is None and len(nv) >= _MIN_CONTAINMENT_CHARS and not nv.isdigit():
        # TS short-value salvage. Digit-only values are excluded from PLAIN
        # substring containment: an id with a dropped digit is a substring of the
        # real printed run (e.g. 11001151763 inside 110011517633) and would ground
        # a wrong money value — they fall through to the boundary-anchored rung.
        anchor = _find_chunk(batch, cited, lambda c: nv in _norm(c.text))
    if anchor is None:  # separator-tolerant salvage (model reformatted separators)
        anchor = _sep_tolerant_anchor(
            batch, cited, value, canonical_numbers=canonical_numbers
        )
    if anchor is None and isinstance(value, bool):
        # checkbox-state salvage rung 1: glyph-normalized quote (S1);
        # via_quote stays False so evidence text is the source chunk, not the
        # model's (possibly different-glyph) quote
        anchor = _checkbox_quote_anchor(batch, cited, quote, value)
    if anchor is None and isinstance(value, bool) and not quote.strip():
        # checkbox-state salvage rung 2: no-quote only — a failed non-empty
        # quote (drifted OR fabricated) must not fall through to token
        # grounding (S1)
        anchor = _checkbox_token_anchor(batch, cited, value)
    if anchor is None:
        # Last: is this leaf unquotable BY CONSTRUCTION (enum / normalized number /
        # zero default / composed date)? Only then is a quote-less value kept.
        why = _unquotable_exemption(value, schema, key, batch)
        if why is not None:
            return _Leaf(value=value, evidence=[], ungrounded=False, verified_by=why)
        return _Leaf(value=(None if strict else value), evidence=[], ungrounded=True)

    support = [anchor]
    if isinstance(value, str) and len(value) >= 80:
        # Long prose spans many line chunks but the model often cites only the
        # first — add every chunk whose text is contained in the value so each
        # line gets its own box (never a page-sized union).
        nvv = _norm(value)
        for i, c in enumerate(batch):
            if len(support) >= SUPPORT_MAX_CHUNKS:
                break
            if i not in support and len(c.text) >= MATCH_MIN_CHUNK_CHARS and _norm(c.text) in nvv:
                support.append(i)
    support = sorted(set(support))[:SUPPORT_MAX_CHUNKS]
    evidence = []
    for i in support:
        # The anchor's source span is the model's verified quote (≤200 chars, and
        # it provably contains the value) — not the raw chunk text, which on a
        # paragraph-sized chunk can be truncated *before* the value and make the
        # citation un-self-verifying (plan 050 §6: text = the grounding quote).
        # Augmentation chunks each contribute their own line text.
        text = quote.strip() if i == anchor and anchor_via_quote else batch[i].text
        evidence.append(FieldEvidence(page=batch[i].page, bbox=batch[i].bbox, text=text[:300],
                                      confidence=batch[i].confidence))
    return _Leaf(value=value, evidence=evidence, ungrounded=False)


def _verify_passthrough(node: Any) -> Any:
    """No-verify mode (2026-07-01 experiment): unwrap wrappers into _Leaf cells
    WITHOUT any grounding check — every model value is trusted verbatim, no
    strict nulling, and no first-pass evidence (citations must come entirely
    from the second pass, routed by value-occurrence pages). Tree shape matches
    :func:`_verify` so merge/flatten work unchanged."""
    if isinstance(node, dict) and (
        node.keys() >= _WRAPPER_KEYS or node.keys() == _WRAPPER_KEYS_NO_IDS
    ):
        v = node.get("value")
        return _Leaf(value=(None if v == "" else v), evidence=[], ungrounded=False)
    if isinstance(node, dict):
        return {k: _verify_passthrough(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_verify_passthrough(v) for v in node]
    return _Leaf(value=node, evidence=[], ungrounded=False)


def _verify(
    node: Any,
    batch: list[_IndexedChunk],
    strict: bool,
    schema: Any = None,
    key: str | None = None,
    *,
    canonical_numbers: bool = False,
) -> Any:
    """Walk the wrapped model result into a tree of ``_Leaf`` cells / dicts /
    lists, mirroring the value shape (used for lockstep merge + flatten).

    ``schema`` is the USER-schema node for this position (the root is passed as
    ``{"type": "object", "properties": <user schema>}``) and ``key`` the leaf's own
    field name; both are walked in lockstep with the value tree purely so
    :func:`_verify_leaf` can see what the leaf DECLARES (enum, numeric type, date
    format). Omitted → no leaf is exempt and verification is exactly as before.
    """
    if isinstance(node, dict) and (
        node.keys() >= _WRAPPER_KEYS or node.keys() == _WRAPPER_KEYS_NO_IDS
    ):
        return _verify_leaf(
            node, batch, strict, schema=schema, key=key, canonical_numbers=canonical_numbers
        )
    if isinstance(node, dict):
        return {
            k: _verify(
                v, batch, strict, _schema_child(schema, k), k,
                canonical_numbers=canonical_numbers,
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [
            _verify(
                v, batch, strict, _schema_items(schema), key,
                canonical_numbers=canonical_numbers,
            )
            for v in node
        ]
    # A bare scalar (model emitted a value without the wrapper) — keep it,
    # ungrounded by construction.
    return _Leaf(value=node, evidence=[], ungrounded=node is not None)


# --- Cross-batch merge ------------------------------------------------------


def _record_key(rec: Any) -> dict[str, Any] | None:
    """Normalized scalar-leaf signature of an array record, for dedup. ``None`` if
    the record isn't a uniform ``{field: _Leaf}`` dict (then we don't dedup it)."""
    if not isinstance(rec, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in rec.items():
        val = v.value if isinstance(v, (_Leaf, _VPLeaf)) else v  # tree cells or already-flattened values
        if isinstance(val, (dict, list)):
            return None  # nested object/array record — leave it alone
        out[k] = re.sub(r"[^a-z0-9]", "", str(val).lower()) if val not in (None, "") else None
    return out


def _field_covers(bv: str | None, sv: str | None) -> bool:
    """Field-level compatibility for subsumption: exact match, or one normalized
    value is a prefix/suffix of the other (min 6 chars). Candidates restate the
    same record with a name variant — 'DevotedHealth' vs 'DevotedHealthPlans',
    'SELF PAY/EMERGENT ONLY' vs 'SELF PAY / SELF PAY/EMERGENT ONLY' — and exact
    equality shipped both as separate coverages. Short values never fuzzy-match
    (a 2-char state code or initials must not glue two records)."""
    if bv == sv:
        return True
    if not bv or not sv:
        return False
    if min(len(bv), len(sv)) < 6:
        return False
    return bv.startswith(sv) or sv.startswith(bv) or bv.endswith(sv) or sv.endswith(bv)


def _subsumes(big: dict[str, Any], small: dict[str, Any]) -> bool:
    """``big`` covers ``small``: every non-null field of ``small`` matches ``big``
    (exactly, or as a prefix/suffix variant of the same printed value) and ``big``
    carries at least as much information."""
    for k, sv in small.items():
        if sv is not None and not _field_covers(big.get(k), sv):
            return False
    return sum(1 for v in big.values() if v) >= sum(1 for v in small.values() if v)


def _dedup_records(records: list) -> list:
    """Drop array records that another record fully covers — the same coverage/
    contact restated on a later page (cross-batch), or a name-only echo of a
    fully-identified one. Conservative: a record survives unless some OTHER record
    is strictly more complete and consistent with it (equal records keep the
    first). Genuinely distinct records (e.g. different member_ids) are untouched."""
    keys = [_record_key(r) for r in records]
    if len(records) < 2 or any(k is None for k in keys):
        return records
    keep: list[int] = []
    for i, ki in enumerate(keys):
        drop = False
        for j, kj in enumerate(keys):
            if i == j or not _subsumes(kj, ki):
                continue
            if _subsumes(ki, kj):  # equal information → keep the earliest
                if j < i:
                    drop = True
                    break
            else:  # kj strictly more complete → ki is a redundant echo
                drop = True
                break
        if not drop:
            keep.append(i)
    return [records[i] for i in keep]


def _dedup_tree(node: Any) -> Any:
    """Apply :func:`_dedup_records` to every array in a verified tree."""
    if isinstance(node, dict):
        return {k: _dedup_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return _dedup_records([_dedup_tree(v) for v in node])
    return node


def _merge(a: Any, b: Any) -> Any:
    """Merge two verified trees of identical schema shape.

    Scalars take the first non-null cell (a grounded value beats a strict-nulled
    one); arrays concatenate in document order (deduping records the same coverage
    restated across pages produces); objects merge per key.
    """
    if isinstance(a, _Leaf) and isinstance(b, _Leaf):
        return a if a.value is not None else b
    if isinstance(a, list) and isinstance(b, list):
        return _dedup_records(a + b)
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge(a[k], b[k]) if k in b else a[k] for k in a} | {
            k: v for k, v in b.items() if k not in a
        }
    # Shape mismatch across batches (rare): prefer the side with content.
    return a if _has_content(a) else b


def _has_content(node: Any) -> bool:
    if isinstance(node, (_Leaf, _VPLeaf)):
        return node.value is not None
    if isinstance(node, dict):
        return any(_has_content(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_content(v) for v in node)
    return node is not None


# --- Flatten tree → (values, evidence, ungrounded) --------------------------


def _flatten(
    node: Any,
    path: str,
    values_out: Any,
    evidence: dict[str, list[FieldEvidence]],
    ungrounded: list[str],
    verified_by: dict[str, str] | None = None,
) -> Any:
    if isinstance(node, _Leaf):
        if node.evidence:
            evidence[path] = node.evidence
        if node.ungrounded:
            ungrounded.append(path)
        if node.verified_by and verified_by is not None:
            verified_by[path] = node.verified_by
        return node.value
    if isinstance(node, dict):
        return {
            k: _flatten(
                v, f"{path}.{k}" if path else k, values_out, evidence, ungrounded, verified_by
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [
            _flatten(v, f"{path}[{i}]", values_out, evidence, ungrounded, verified_by)
            for i, v in enumerate(node)
        ]
    return node


# --- Orchestration ----------------------------------------------------------


# --- Field-group splitting (plan 050 M4) -----------------------------------


def _leaf_count(node: Any) -> int:
    if not isinstance(node, dict):
        return 1
    t = node.get("type")
    if t == "object":
        return sum(_leaf_count(v) for v in node.get("properties", {}).values()) or 1
    if t == "array":
        return _leaf_count(node.get("items", {}))
    return 1


def split_schema(fields: dict[str, Any], max_leaves: int = MAX_GROUP_LEAVES) -> list[dict[str, Any]]:
    """Partition a field-map into disjoint groups, each ≤ ``max_leaves`` leaves.

    Small schemas return a single group (no behaviour change). A field whose own
    leaf count exceeds the budget is split by recursing into its object
    properties, so a 161-leaf ``balance_sheet`` becomes several
    ``{balance_sheet: {subset}}`` groups that the merge re-unites.
    """
    groups: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    cur_n = 0

    def flush() -> None:
        nonlocal cur, cur_n
        if cur:
            groups.append(cur)
            cur, cur_n = {}, 0

    for k, node in fields.items():
        nl = _leaf_count(node)
        if nl <= max_leaves:
            if cur and cur_n + nl > max_leaves:
                flush()
            cur[k] = node
            cur_n += nl
        else:
            flush()
            if isinstance(node, dict) and node.get("type") == "object":
                for sg in split_schema(node.get("properties", {}), max_leaves):
                    groups.append({k: {"type": "object", "properties": sg}})
            else:  # an array (or other) too big to split — keep whole, best effort
                groups.append({k: node})
    flush()
    return groups or [dict(fields)]


# --- Output diet (flag-gated) ----------------------------------------------
# Decode is the cost and latency floor of an extraction call, and most of a wide
# receipt's output tokens are structure the SCHEMA already carries: leaves the
# page never printed, and the same column names respelled on every row. Both
# levers below shrink only what the model must EMIT. Neither is post-processing
# of a value: the omitted shape is rebuilt from the user schema (a key, never a
# value) and the row aliases are a bijection undone before verification, so what
# reaches the grounding gate — and the caller — is byte-for-byte the same
# contract as with the flags off.


def _absent(node: Any, *, wrapped: bool, chunk_ids: bool) -> Any:
    """What a diet-omitted schema node would have carried had the model emitted
    it: nulls all the way down (an object keeps its keys, an array is empty)."""
    t = _scalar_type(node.get("type", "string")) if isinstance(node, dict) else "string"
    if t == "object":
        return {k: _absent(v, wrapped=wrapped, chunk_ids=chunk_ids)
                for k, v in node.get("properties", {}).items()}
    if t == "array":
        return []
    if not wrapped:
        return None
    return {"value": None, "chunks": [], "quote": ""} if chunk_ids else {"value": None, "quote": ""}


def _rehydrate(raw: Any, node: Any, *, wrapped: bool, chunk_ids: bool) -> Any:
    """Restore the user schema's shape onto a diet response, before anything reads
    it: every declared leaf the model left out comes back as a null cell with no
    evidence. Keys only — a value is never invented — so verification, merge,
    dedup and flatten downstream walk exactly the tree they always walked. Keys
    the model volunteered that the schema does not declare are left alone."""
    if not isinstance(node, dict):
        return raw
    t = _scalar_type(node.get("type", "string"))
    if t == "object" and isinstance(raw, dict):
        restored = {
            k: (_rehydrate(raw[k], v, wrapped=wrapped, chunk_ids=chunk_ids) if k in raw
                else _absent(v, wrapped=wrapped, chunk_ids=chunk_ids))
            for k, v in node.get("properties", {}).items()
        }
        return {**raw, **restored}
    if t == "array" and isinstance(raw, list):
        items = node.get("items", {"type": "string"})
        return [_rehydrate(r, items, wrapped=wrapped, chunk_ids=chunk_ids) for r in raw]
    return raw


_ALIAS_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _alias(i: int) -> str:
    """Positional short key: ``0→a … 25→z, 26→aa … 701→zz, 702→aaa``.

    Bijective base-26, unbounded: the two-character form runs out at 702 and the
    arithmetic that produced it raised ``IndexError`` there. That is inside the
    module's own 1,000-leaf ceiling, so a wide-enough array schema could fail the
    request before a single model call — an alias generator must cover every
    schema the validator admits.
    """
    out = ""
    n = i
    while True:
        out = _ALIAS_LETTERS[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            return out


def _row_alias_maps(group_schema: dict[str, Any]) -> dict[str, dict[str, str]]:
    """``{array_field: {alias: real_key}}`` for each top-level array-of-objects in
    this group — aliases assigned in schema property order."""
    maps: dict[str, dict[str, str]] = {}
    for name, node in group_schema.items():
        if not isinstance(node, dict) or _scalar_type(node.get("type")) != "array":
            continue
        items = node.get("items")
        if not isinstance(items, dict) or _scalar_type(items.get("type")) != "object":
            continue
        props = items.get("properties") or {}
        if props:
            maps[name] = {_alias(i): k for i, k in enumerate(props)}
    return maps


def _apply_row_aliases(
    response_schema: dict[str, Any], maps: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Rename each aliased array's item properties in the response schema. The
    per-column descriptions ride along untouched — they are input tokens, and the
    prompt legend ties each alias back to the name the caller asked for."""
    props = dict(response_schema.get("properties") or {})
    for name, amap in maps.items():
        arr = props.get(name)
        if not isinstance(arr, dict) or not isinstance(arr.get("items"), dict):
            continue
        items = dict(arr["items"])
        item_props = items.get("properties") or {}
        items["properties"] = {a: item_props[k] for a, k in amap.items() if k in item_props}
        if isinstance(items.get("required"), list):
            req = set(items["required"])
            items["required"] = [a for a, k in amap.items() if k in req]
        props[name] = {**arr, "items": items}
    return {**response_schema, "properties": props}


def _unalias_rows(raw: Any, maps: dict[str, dict[str, str]]) -> Any:
    """Alias keys → the caller's field names, before any value is read."""
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    for name, amap in maps.items():
        rows = out.get(name)
        if isinstance(rows, list):
            out[name] = [
                {amap.get(k, k): v for k, v in r.items()} if isinstance(r, dict) else r
                for r in rows
            ]
    return out


def _alias_legend(maps: dict[str, dict[str, str]]) -> str:
    """The prompt line that makes the aliasing readable to the model."""
    return "".join(
        f"\nItem keys in {name} are abbreviated — keys: "
        + ", ".join(f"{a}={k}" for a, k in amap.items())
        + "\n"
        for name, amap in maps.items()
    )


def _merge_values(a: Any, b: Any) -> Any:
    """Deep-union two disjoint field-group value trees (objects split across
    groups deep-merge; arrays concatenate)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge_values(a[k], v) if k in a else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        return _dedup_records(a + b)
    return a if a is not None else b


def _model_pins(
    service_tier: str | None,
    model_name: str | None,
    candidate_count: int | None,
) -> dict[str, Any]:
    """The per-request model arguments, as kwargs — and ONLY the ones that are set.

    Every one of these overrides a deployment-wide setting for a single request,
    and every one of them must be invisible when unset: a ``SchemaModel`` is only
    required to accept ``(prompt, response_schema[, images])``, so a stub or a
    non-Vertex implementation has to keep being called with exactly the signature
    it has. ``candidate_count`` is compared to ``None`` rather than truth-tested —
    ``1`` is a meaningful pin (candcount OFF) on a deployment whose default is 2.
    """
    pins: dict[str, Any] = {}
    if service_tier:
        pins["service_tier"] = service_tier
    if model_name:
        pins["model"] = model_name
    if candidate_count is not None:
        pins["candidate_count"] = candidate_count
    return pins


async def _gather_groups(tasks: list) -> list:
    """Every group's result, in order, or the first exception raised."""
    return await asyncio.gather(*tasks)


async def _drain_groups(tasks: list) -> None:
    """Leave no group call running after this request has stopped needing it.

    ``asyncio.gather`` does NOT cancel its other children when one raises, so a
    single failed group used to return an error to the caller while every other
    Vertex call carried on — still running, still billing, its own exception
    surfacing later as "never retrieved". Cancelling a finished task is a no-op,
    so the success path is untouched; the second pass retrieves each outcome so
    nothing is left unobserved. Also covers the request itself being cancelled.
    """
    for t in tasks:
        if not t.done():
            t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t


async def _aextract_group(
    batches: list[list[_IndexedChunk]],
    group_schema: dict[str, Any],
    model: SchemaModel,
    strict: bool,
    chunk_ids: bool = True,
    verify_grounding: bool = True,
    wrapped: bool = True,
    *,
    exempt_unquotable: bool = False,
    keep_repeated_rows: bool = False,
    context_first: bool = False,
    page_images: dict[int, bytes] | None = None,
    chunk_transform: ChunkTransform | None = None,
    extra_instructions: str | None = None,
    omit_null: bool = False,
    short_row_keys: bool = False,
    service_tier: str | None = None,
    model_name: str | None = None,
    candidate_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[FieldEvidence]], list[str], dict[str, str]]:
    """Extract one field-group across the document batches; return
    ``(values, evidence, ungrounded, verified_by)``.

    ``wrapped=False`` = bare mode: the model fills the user schema directly (no
    envelope, no quotes, passthrough only). ``chunk_transform`` /
    ``extra_instructions`` (additive, both default ``None``) are threaded through
    from :func:`aextract_schema`; with every option at its default the prompt is
    byte-identical to the historical ``EXTRACTION_PROMPT + render_context(b)``.

    ``omit_null`` / ``short_row_keys`` are the output diet (deployment flags,
    both default off → schema and prompt unchanged). Row aliasing applies to BARE
    array groups only: a wrapped leaf's keys are the evidence envelope, not the
    caller's field names.

    ``service_tier`` (per request, ``None`` = the deployment's
    ``EXTRACT_SERVICE_TIER``) is passed to the model only when it is set, so a
    ``SchemaModel`` that does not accept the argument — every test stub, and any
    non-Vertex implementation — is called with exactly the signature it had.
    ``model_name`` (→ the model's ``model`` argument) and ``candidate_count``
    ride the SAME guard, for the same reason: they are per-request pins of the
    two settings a measured stack names (``GEMINI_EXTRACT_MODEL`` /
    ``GEMINI_EXTRACT_CANDIDATE_COUNT``), and unset means "whatever the model was
    constructed with", byte-identical to before they existed.
    """
    response_schema = build_response_schema(
        group_schema, chunk_ids=chunk_ids, wrapped=wrapped, omit_null=omit_null
    )
    alias_maps = _row_alias_maps(group_schema) if (short_row_keys and not wrapped) else {}
    if alias_maps:
        response_schema = _apply_row_aliases(response_schema, alias_maps)
    if not wrapped:
        base_prompt = EXTRACTION_PROMPT_BARE
    else:
        base_prompt = EXTRACTION_PROMPT if chunk_ids else EXTRACTION_PROMPT_NO_IDS
    if omit_null:
        base_prompt = base_prompt + _OMIT_NULL_INSTRUCTION
    suffix = f"\n{extra_instructions}" if extra_instructions else ""
    if alias_maps:
        suffix = suffix + _alias_legend(alias_maps)
    if page_images:
        base_prompt = base_prompt + _IMAGE_READ_GUIDANCE

    def _imgs_for(batch):
        """Only the images for pages this batch actually contains — attaching the
        whole document to every batch would multiply prefill for no benefit."""
        if not page_images:
            return None
        out = [page_images[pg] for pg in sorted({c.page for c in batch}) if pg in page_images]
        return out or None

    # Per-request model arguments, sent ONLY when set — see the docstring.
    pins = _model_pins(service_tier, model_name, candidate_count)

    def _call(b):
        """Two-arg call when there are no images. `SchemaModel` implementations are
        not required to accept an images parameter (the test stubs do not), so the
        text-only path must remain byte-identical AND signature-identical."""
        if context_first:
            # Cache-friendly layout: shared context leads, group-specific
            # instructions trail, images go first at the client — the identical
            # image+context prefix across a request's group calls hits Vertex
            # implicit caching (see EXTRACT_CONTEXT_FIRST_CUSTOMERS).
            prompt = render_context(b, chunk_transform) + base_prompt + suffix
        else:
            prompt = base_prompt + render_context(b, chunk_transform) + suffix
        imgs = _imgs_for(b)
        if imgs is None:
            return model.generate_json(prompt, response_schema, **pins)
        if context_first:
            return model.generate_json(prompt, response_schema, imgs, images_first=True, **pins)
        return model.generate_json(prompt, response_schema, imgs, **pins)

    raws = await asyncio.gather(*(_call(b) for b in batches))
    # OUTPUT-DIET REHYDRATION, before anything reads a value: alias keys become the
    # caller's field names again, and every leaf the model was allowed to omit comes
    # back null. Both are shape restoration off the user schema, so the verified /
    # merged / flattened tree is the one the un-dieted call would have produced.
    if alias_maps:
        raws = [_unalias_rows(raw, alias_maps) for raw in raws]
    if omit_null:
        diet_root = {"type": "object", "properties": group_schema}
        raws = [
            _rehydrate(raw, diet_root, wrapped=wrapped, chunk_ids=chunk_ids) for raw in raws
        ]
    # The user schema travels with the values so each leaf is verified knowing what it
    # DECLARES (enum / numeric / date) — see :func:`_unquotable_exemption`. Per-customer
    # (EXTRACT_GATE_EXEMPT_CUSTOMERS → ``exempt_unquotable``): without it the schema is
    # withheld from _verify, so no leaf is exempt and the gate behaves exactly as it
    # always has for every other customer.
    root_schema = {"type": "object", "properties": group_schema} if exempt_unquotable else None
    trees = [
        _verify(raw, batch, strict, root_schema, canonical_numbers=exempt_unquotable)
        if (wrapped and verify_grounding)
        else _verify_passthrough(raw)
        for raw, batch in zip(raws, batches, strict=True)
    ]
    merged = trees[0]
    for tree in trees[1:]:
        merged = _merge(merged, tree)
    # Post-verify record dedup for EVERY doc. Multi-batch merges dedup inside
    # _merge, but a single-batch doc never passed through it — so a candcount
    # union echo (same coverage restated, or a strict-nulled-name twin of a
    # fully-identified record) survived to the output. Idempotent on merged trees.
    if not keep_repeated_rows:
        merged = _dedup_tree(merged)
    evidence: dict[str, list[FieldEvidence]] = {}
    ungrounded: list[str] = []
    verified_by: dict[str, str] = {}
    values = _flatten(merged, "", None, evidence, ungrounded, verified_by)
    return (values if isinstance(values, dict) else {}), evidence, ungrounded, verified_by


async def agenerate_schema(
    chunks: list[Chunk],
    page_sizes: list[tuple[float, float]],
    model: SchemaModel,
) -> dict[str, Any]:
    """Design a flat extraction schema from the document (plan 050 M4 auto_schema).

    Returns a user_schema (``{name: {type, description}}``) that is guaranteed to
    pass :func:`validate_schema` (flat, supported leaf types, ≤
    :data:`MAX_GENERATED_FIELDS`). The design pass reads the FIRST context batch —
    a representative slice — so wide docs don't pay for the whole text twice just
    to pick fields; the subsequent extract pass still sees every chunk.

    Raises :class:`EmptySchema` when the document has no text, or the model
    returns no usable field — both map to 422 (nothing to extract).
    """
    indexed = _index_chunks(chunks, page_sizes)
    if not indexed:
        raise EmptySchema("Document has no extractable text to design a schema from.")
    raw = await model.generate_json(
        SCHEMA_GEN_PROMPT + render_context(_design_sample(indexed)), _SCHEMA_GEN_RESPONSE
    )
    fields = raw.get("fields") if isinstance(raw, dict) else None
    schema: dict[str, Any] = {}
    for f in fields or []:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        # Field names become evidence/source path keys (``a.b``, ``a[0]``); a name
        # carrying those separators would corrupt the namespace — drop it.
        if not name or name in schema or any(c in name for c in ".[]"):
            continue
        ftype = f.get("type")
        if ftype not in _SUPPORTED_LEAF_TYPES:
            ftype = "string"
        schema[name] = {"type": ftype, "description": str(f.get("description") or "")}
        if len(schema) >= MAX_GENERATED_FIELDS:
            break
    if not schema:
        raise EmptySchema("Schema generation produced no usable fields.")
    return schema


def _design_sample(indexed: list[_IndexedChunk]) -> list[_IndexedChunk]:
    """Chunks to show the schema-design pass: the whole document if it fits
    :data:`DESIGN_CONTEXT_CHARS`, else an evenly-strided sample spanning every
    section so fields on later pages still get designed (not just the head)."""
    costs = [len(c.text) + 24 for c in indexed]
    if sum(costs) <= DESIGN_CONTEXT_CHARS:
        return indexed
    keep = max(1, int(len(indexed) * DESIGN_CONTEXT_CHARS / sum(costs)))
    step = len(indexed) / keep
    idxs = sorted({min(len(indexed) - 1, int(i * step)) for i in range(keep)})
    return [indexed[i] for i in idxs]


# --- Reconciliation pass (best-of-N with a multimodal judge) ----------------
# The first pass reads ONE serialization of the document (the parse text, plus any
# labelled secondary chunks) and answers in one shot. When a document carries the same
# fact several ways — printed text, an independent recognizer's read, a machine-decoded
# QR payload, and the pixels themselves — a single pass has to commit to a reading with
# no opportunity to weigh the alternatives against a concrete answer.
#
# The reconciliation pass gives it that opportunity: the SAME model, over the SAME
# sources, is shown the first pass's answer as a labelled CANDIDATE and asked to produce
# the final typed schema from scratch. This is best-of-N (N=2) with the model as its own
# judge — no deterministic stage reads the document, decides a field, or edits a string.
#
# What the code does with the two answers is bookkeeping, not content transformation:
# the reconciler's leaves go through the IDENTICAL grounding ladder (:func:`_verify`),
# and a leaf is only substituted when the reconciler produced a non-null value that
# grounds. An ungrounded reconciler value is dropped and the first pass keeps the field.

#: Appended AFTER the document context (same position as ``extra_instructions``), so the
#: reconciler sees every source first and the task last. It restates the citation
#: contract because the grounding gate is real: an ungrounded leaf is discarded.
RECONCILE_INSTRUCTIONS = """
You are now producing the FINAL extraction for this document. Above are all of the \
sources available for it: the parsed text chunks, any clearly-labelled secondary \
sources (an independent recognizer's read of the page, a machine-decoded QR payload), \
and — when they are attached to this request — the page images themselves.

Below is a CANDIDATE extraction produced by an earlier pass over the same document. \
It is a hint, not an answer. Any field in it may be wrong, may be null where the \
document does state a value, or may state a value the document does not support.

Fill the response schema from scratch, reconciling every source:
- Where the sources agree, return the agreed value.
- Where they disagree, decide which source actually read the page correctly and return \
THAT reading. Never average, splice or blend two readings of the same value.
- For an IDENTIFIER (an id, code, reference, authorization or hash-like string) return \
the characters EXACTLY as the source you trust prints them — never normalize \
separators, capitalization, padding or length, and never shorten or extend the run.
- A source that was MACHINE-DECODED (a QR payload) carries error correction that \
printed text does not, so for a hash-like identifier prefer it over a text read of the \
same field when the two disagree.
- A null is still better than a guess. If the candidate holds a value the document does \
not support, return null instead of repeating it.

Every field still needs its own citation into the TEXT chunks above under exactly the \
rules given at the top of this prompt. A value you cannot ground with a literal quote \
is discarded and the candidate's value is kept in its place.
"""

#: The first pass's answer, labelled as fallible. Its own header, so the model can never
#: read it as part of the document.
_CANDIDATE_LABEL = "\n[CANDIDATE EXTRACTION — may contain errors]\n"


_RECON_MISSING = object()  # absent-key sentinel: absence is not a retraction


def _row_scalars(node: Any, path: str = "") -> dict[str, Any]:
    """Every scalar leaf under one record, keyed by its path inside that record."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            out.update(_row_scalars(v, f"{path}.{k}" if path else k))
        return out
    if isinstance(node, list):
        out = {}
        for i, v in enumerate(node):
            out.update(_row_scalars(v, f"{path}[{i}]"))
        return out
    return {path: node}


def _identity_cells(rows: list[Any]) -> list[dict[str, Any]]:
    """Per row, the filled cells that DISTINGUISH it from every sibling row.

    A receipt's rows share plenty of cells — ``quantity=1``, ``unit="UN"``, a
    repeated unit price — so agreement on *a* cell says nothing about identity. A
    cell earns identity only if its (path, value) pair occurs in exactly one row
    of the array, which is a counting fact about this array, not a threshold.
    """
    filled = [
        {k: v for k, v in _row_scalars(r).items() if v is not None}
        if isinstance(r, dict)
        else {}
        for r in rows
    ]
    seen: dict[tuple[str, Any], int] = {}
    for cells in filled:
        for kv in cells.items():
            try:
                seen[kv] = seen.get(kv, 0) + 1
            except TypeError:  # unhashable value: never an identity cell
                continue
    out: list[dict[str, Any]] = []
    for cells in filled:
        out.append({k: v for k, v in cells.items() if seen.get((k, v)) == 1})
    return out


def _rows_correspond(
    base_row: Any, recon_row: Any, identity: dict[str, Any], filled: dict[str, Any]
) -> bool:
    """Is ``recon_row`` the SAME RECORD as ``base_row``, or a different row at the
    same index?

    The test is agreement on an IDENTITY-BEARING cell — one whose value occurs in
    no other row of the first pass's array (:func:`_identity_cells`). Agreement on
    just any filled cell is not enough: distinct receipt lines routinely share
    ``quantity=1`` or a unit, and matching on those would let row B be merged into
    row A and ship A's quantity with B's description, sku and amount. No tuned
    constant anywhere — "occurs in exactly one row" is a count over this array.

    Three deliberate ``True``s, each a case where nothing can be hybridized:
    a non-record pair (scalar array elements); a base row with no filled cell at
    all (the empty row a second pass should be free to complete); and a row with
    no identity cell of its own — every value it carries also appears in a sibling
    — which is accepted only when the counterpart repeats ALL of its filled cells,
    i.e. when the two are the same content anyway (the genuinely repeated receipt
    line: five identical waters). Wrong answers cost a correction, never a
    corruption: the fallback is always "keep the first pass".
    """
    if not isinstance(base_row, dict) or not isinstance(recon_row, dict):
        return True
    if not filled:
        return True
    recon_cells = _row_scalars(recon_row)
    if identity:
        return any(recon_cells.get(k) == v for k, v in identity.items())
    return all(recon_cells.get(k) == v for k, v in filled.items())


def _reconcile_merge(
    base: Any,
    recon: Any,
    path: str,
    *,
    recon_ungrounded: set[str],
    recon_evidence: dict[str, list[FieldEvidence]],
    out_evidence: dict[str, list[FieldEvidence]],
    replaced: list[str],
    counts: dict[str, int],
) -> Any:
    """Per-leaf merge of two model answers: grounded-reconciler-wins, else first-pass.

    Mechanical bookkeeping over two already-verified outputs — it never inspects the
    document, and never composes a value out of parts of both answers. A leaf is
    substituted only when the reconciler's value is non-null AND grounded (its path is
    not in ``recon_ungrounded``); the citation travels with the value it belongs to, so
    a substituted field is never shipped under the other pass's quote.

    Containers merge in lockstep. Arrays merge element-wise over the common prefix only:
    changing the ROW SET is not per-leaf bookkeeping, so the first pass's records are
    the record set (a reconciler row with no counterpart is dropped, and a first-pass
    row the reconciler omitted is kept untouched).

    Element-wise is POSITIONAL, which is only sound while the two passes agree on row
    order — and nothing makes the second call preserve it. A reordered array would put
    row B opposite row A and, since each leaf is taken independently, could ship A's
    description with B's sku and amount: a record that neither pass ever produced,
    which is precisely what "never composes a value out of parts of both answers"
    exists to forbid. So a record pair is merged only once :func:`_rows_correspond`
    says the two are the same row — agreement on a cell that no OTHER row carries —
    otherwise the first pass's row stands whole.
    """
    if isinstance(base, dict) and isinstance(recon, dict):
        return {
            k: _reconcile_merge(
                v, recon.get(k, _RECON_MISSING), f"{path}.{k}" if path else k,
                recon_ungrounded=recon_ungrounded, recon_evidence=recon_evidence,
                out_evidence=out_evidence,
                replaced=replaced, counts=counts,
            )
            for k, v in base.items()
        }
    if isinstance(base, list) and isinstance(recon, list):
        out_rows = []
        identities = _identity_cells(base)
        for i, v in enumerate(base):
            counterpart = recon[i] if i < len(recon) else _RECON_MISSING
            filled_cells = (
                {k: c for k, c in _row_scalars(v).items() if c is not None}
                if isinstance(v, dict)
                else {}
            )
            if counterpart is not _RECON_MISSING and not _rows_correspond(
                v, counterpart, identities[i], filled_cells
            ):
                # Same index, different record: keep the first pass's row whole
                # rather than mix two rows into one that neither pass emitted.
                counts["rows_unmatched"] = counts.get("rows_unmatched", 0) + 1
                counts["leaves_kept"] += _leaf_cells(v)
                out_rows.append(v)
                continue
            out_rows.append(
                _reconcile_merge(
                    v, counterpart, f"{path}[{i}]",
                    recon_ungrounded=recon_ungrounded, recon_evidence=recon_evidence,
                    out_evidence=out_evidence,
                    replaced=replaced, counts=counts,
                )
            )
        return out_rows
    if isinstance(base, (dict, list)):  # shape mismatch — keep the first pass whole
        counts["leaves_kept"] += _leaf_cells(base)
        return base
    # --- leaf ---------------------------------------------------------------
    # An EXPLICIT reconciler null is a retraction: the second pass, seeing every source,
    # judged the candidate value unsupported. Measured (dev prototype): letting the null
    # through keeps the lift and returns hallucination cells to baseline. A merely
    # ABSENT key (sentinel) never retracts.
    if recon is None and base is not None and path not in recon_ungrounded:
        counts["leaves_nulled"] = counts.get("leaves_nulled", 0) + 1
        replaced.append(path)
        out_evidence.pop(path, None)
        return None
    take = (
        recon is not None
        and recon is not _RECON_MISSING
        and not isinstance(recon, (dict, list))
        and path not in recon_ungrounded
    )
    if not take:
        counts["leaves_kept"] += 1
        return base
    if recon != base:
        counts["leaves_changed"] += 1
        replaced.append(path)
        if path in recon_evidence:
            out_evidence[path] = recon_evidence[path]
        else:  # grounded leaves always carry evidence; drop a stale citation regardless
            out_evidence.pop(path, None)
        return recon
    counts["leaves_kept"] += 1
    return base


def _leaf_cells(node: Any) -> int:
    """Scalar cells under ``node`` (for the kept/changed accounting only)."""
    if isinstance(node, dict):
        return sum(_leaf_cells(v) for v in node.values())
    if isinstance(node, list):
        return sum(_leaf_cells(v) for v in node)
    return 1


async def _areconcile(
    batch: list[_IndexedChunk],
    user_schema: dict[str, Any],
    candidate: dict[str, Any],
    model: SchemaModel,
    *,
    strict: bool,
    chunk_ids: bool,
    images: dict[int, bytes] | None,
    chunk_transform: ChunkTransform | None,
    extra_instructions: str | None,
    exempt_unquotable: bool = False,
    service_tier: str | None = None,
    model_name: str | None = None,
    candidate_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[FieldEvidence]], list[str], dict[str, str]]:
    """ONE extra model call over every source plus the candidate; verified identically.

    Returns the reconciler's ``(values, evidence, ungrounded, verified_by)`` in the same
    flat-path shape the first pass produces, so the merge is a lockstep walk of two like
    trees.

    "Verified identically" is load-bearing and includes the EXEMPTIONS: the schema
    reaches ``_verify`` here only when ``exempt_unquotable`` is on, exactly as in
    the first pass. Passing it unconditionally would let the second pass keep an
    unquotable enum that the first pass would have nulled — and because a
    grounded reconciler leaf REPLACES the candidate, a customer who enabled
    reconciliation without enabling exemptions would get exempted values through
    a back door they never turned on."""
    response_schema = build_response_schema(user_schema, chunk_ids=chunk_ids, wrapped=True)
    base_prompt = EXTRACTION_PROMPT if chunk_ids else EXTRACTION_PROMPT_NO_IDS
    if images:
        base_prompt = base_prompt + _IMAGE_READ_GUIDANCE
    prompt = (
        base_prompt
        + render_context(batch, chunk_transform)
        + (f"\n{extra_instructions}" if extra_instructions else "")
        + RECONCILE_INSTRUCTIONS
        + _CANDIDATE_LABEL
        + json.dumps(candidate, indent=2, ensure_ascii=False, default=str)
    )
    imgs = (
        [images[pg] for pg in sorted({c.page for c in batch}) if pg in images] or None
        if images
        else None
    )
    # The reconciler answers on the SAME model, tier and candidate count as the
    # pass it is second-guessing — see _model_pins.
    pins = _model_pins(service_tier, model_name, candidate_count)
    raw = (
        await model.generate_json(prompt, response_schema, **pins)
        if imgs is None
        else await model.generate_json(prompt, response_schema, imgs, **pins)
    )
    recon_root = {"type": "object", "properties": user_schema} if exempt_unquotable else None
    tree = _dedup_tree(
        _verify(raw, batch, strict, recon_root, canonical_numbers=exempt_unquotable)
    )
    evidence: dict[str, list[FieldEvidence]] = {}
    ungrounded: list[str] = []
    verified_by: dict[str, str] = {}
    values = _flatten(tree, "", None, evidence, ungrounded, verified_by)
    return (values if isinstance(values, dict) else {}), evidence, ungrounded, verified_by


async def aextract_schema(
    chunks: list[Chunk],
    page_sizes: list[tuple[float, float]],
    user_schema: dict[str, Any],
    *,
    model: SchemaModel,
    strict: bool = True,
    chunk_ids: bool = True,
    verify_grounding: bool = True,
    wrapped: bool = True,
    exempt_unquotable: bool = False,
    chunk_transform: ChunkTransform | None = None,
    extra_instructions: str | None = None,
    page_images: dict[int, bytes] | None = None,
    xref_images: dict[int, bytes] | None = None,
    xref_cache=None,
    xref_always: bool = False,
    xref_second_reader: bool = True,
    xref_prefetched: list[Chunk] | None = None,
    lean_arrays: bool = False,
    bare_scalars: bool | None = None,
    keep_repeated_rows: bool = False,
    context_first: bool = False,
    aux_arrays_inline: bool | None = None,
    omit_null: bool | None = None,
    short_row_keys: bool | None = None,
    cache_stagger_seconds: float | None = None,
    reconcile: bool = False,
    reconcile_images: dict[int, bytes] | None = None,
    service_tier: str | None = None,
    model_name: str | None = None,
    candidate_count: int | None = None,
) -> SchemaExtractResult:
    """Fill ``user_schema`` from parse ``chunks`` with grounded citations.

    ``chunk_ids=False`` asks the model for ``{value, quote}`` only (no chunk-id
    citations); grounding is unchanged — the quote text-match anchors evidence.
    ``verify_grounding=False`` (experiment) skips the grounding gate entirely:
    values pass through untouched, with no strict nulling and no evidence.
    ``wrapped=False`` (experiment) drops the envelope altogether — the model fills
    the user schema directly; raw Gemini output, no quotes, no post-processing.

    Raises :class:`SchemaValidationError` (→ 422) for a bad schema before any
    model call. Two axes of splitting, both merged transparently:
    - **Document**: long docs are split into context-budget batches.
    - **Schema** (plan 050 M4): wide schemas are split into disjoint field-groups
      of ≤ :data:`MAX_GROUP_LEAVES` leaves so the wrapped response_schema stays
      under Gemini's complexity limit. A narrow schema is one group (unchanged).

    ``chunk_transform`` / ``extra_instructions`` are additive prompt-shaping hooks
    (both default ``None``), used by the layout recipe (:mod:`extract.core.
    layout_recipe`) to run alternate passes over the SAME chunks:
    - ``chunk_transform`` rewrites only the rendered per-chunk text (e.g. inject
      (x%,y%) + [SECTION] prefixes); the chunk geometry — hence every citation — is
      untouched.
    - ``extra_instructions`` appends a suffix to the extraction prompt (e.g. the
      entity-first / "form is the source of record" instructions).
    With both ``None`` (the default) every model call is byte-identical to today's
    production extraction — no behaviour change when unused.

    ``reconcile`` (per customer, ``EXTRACT_RECONCILE_CUSTOMERS``) adds ONE more call
    after the pass above: the same model, over the same sources plus ``reconcile_images``,
    is shown this pass's answer as a fallible candidate and re-answers the whole schema.
    Its leaves are verified by the SAME grounding ladder, and a leaf is replaced only
    where the reconciler's value is non-null and grounded.

    ``EXTRACT_OMIT_NULL_LEAVES`` / ``EXTRACT_SHORT_ROW_KEYS`` (deployment flags, both
    off by default) are the OUTPUT DIET — see the helpers above; they change only what
    the model emits, never a value. ``omit_null`` / ``short_row_keys`` /
    ``cache_stagger_seconds`` scope those three settings to THIS request (``None`` =
    read the deployment's): a lane that measured the diet on its own documents can run
    it without turning it on for every customer's schema.

    ``service_tier`` scopes the Vertex scheduling tier to THIS pass (``None`` = the
    deployment's ``EXTRACT_SERVICE_TIER``); ``model_name`` / ``candidate_count`` do
    the same for the model id and candcount (``None`` = ``GEMINI_EXTRACT_MODEL`` /
    ``GEMINI_EXTRACT_CANDIDATE_COUNT``, i.e. whatever the model was built with);
    ``aux_arrays_inline`` does the same for
    the 3-call shape (``None`` = ``EXTRACT_AUX_ARRAYS_INLINE``);
    ``xref_second_reader=False`` drops the
    Gemini recognizer from the fallback serial cross-read, matching what the caller
    asked its prefetch for. Both default to today's behaviour exactly.
    """
    # CROSS-ENGINE SECONDARY CONTEXT (opt-in). Appends an independent recognizer's read of
    # each supplied page as a labelled extra chunk. This is the only lever that can reach a
    # value our own parse GARBLED -- no schema or prompt change can recover characters that
    # are not in the text the model reads. Fail-open: a provider error yields fewer chunks.
    # ``xref_always`` runs both recognizers on every page instead of only the handwriting
    # pages (per customer; EXP-A: +0.035 dev micro on printed receipts, +~$0.004/page).
    # ``xref_prefetched`` is the same cross-read computed in the route's prefetch
    # thread, concurrent with parse — when present it saves the recognizer's
    # serial round-trip here (p50 ~7.5s). ``None`` (offline callers, prefetch
    # failure) falls back to the original serial call, byte-identical output.
    if xref_prefetched is not None:
        if xref_prefetched:
            chunks = list(chunks) + list(xref_prefetched)
    elif xref_images:
        from extract.core.xref_context import xref_chunks as _xref

        _extra = _xref(
            xref_images, cache=xref_cache, always=xref_always, second_reader=xref_second_reader
        )
        if _extra:
            chunks = list(chunks) + _extra

    user_schema = flatten_schema(user_schema)  # accept Pydantic/zod ($ref/anyOf/allOf)
    validate_schema(user_schema)
    indexed = _index_chunks(chunks, page_sizes)
    if not indexed:
        # Nothing to read from — every field is absent, no evidence.
        return SchemaExtractResult(
            values={k: None for k in user_schema}, evidence={}, ungrounded_fields=[]
        )

    batches = _batch(indexed)
    # The three OUTPUT-DIET/layout settings, resolved once: the caller's per-request
    # value when it gave one, the deployment's otherwise (which is what every caller
    # that predates the per-request form still gets).
    _omit_null = settings.EXTRACT_OMIT_NULL_LEAVES if omit_null is None else omit_null
    _short_row_keys = (
        settings.EXTRACT_SHORT_ROW_KEYS if short_row_keys is None else short_row_keys
    )
    _stagger = (
        settings.EXTRACT_CACHE_STAGGER_SECONDS
        if cache_stagger_seconds is None
        else cache_stagger_seconds
    )
    if lean_arrays:
        # Each top-level array field becomes its own BARE group (no evidence
        # envelope — the envelope is what multiplies a 50-row receipt's output
        # past the emission/response budget). Scalars keep the full envelope,
        # grounding gate included, split as always.
        _arr = {
            k: v
            for k, v in user_schema.items()
            if isinstance(v, dict) and v.get("type") == "array"
        }
        _rest = {k: v for k, v in user_schema.items() if k not in _arr}
        # One bare group PER top-level array. A single merged all-arrays call was
        # measured −1.1pt on line_items values at full scale (SEALED5 vs SEALED4
        # items 0.9551 vs 0.9687 — auxiliary arrays dilute the long table's call);
        # the split's context-duplication cost is reclaimed by the context-first
        # cache layout instead (EXTRACT_CONTEXT_FIRST_CUSTOMERS: identical
        # context+image prefix across group calls bills at the cached rate).
        # Per-request when the caller said so, the deployment default otherwise —
        # a lane whose sealed shape is the 3-call one asks for it directly instead
        # of the deployment turning it on for everybody.
        _inline = (
            settings.EXTRACT_AUX_ARRAYS_INLINE if aux_arrays_inline is None else aux_arrays_inline
        )
        if _inline and len(_arr) > 1:
            # 3-call shape: only the widest array (by schema size — the long
            # table) keeps its own bare call; small aux arrays ride in the
            # wrapped scalars group. Fewer queue draws, one less context copy,
            # and no aux dilution of the long table's call.
            _main = max(_arr, key=lambda k: len(json.dumps(_arr[k])))
            _rest = {**{k: v for k, v in _arr.items() if k != _main}, **_rest}
            _arr = {_main: _arr[_main]}
        groups = [{k: _arr[k]} for k in _arr] + (split_schema(_rest) if _rest else [])
        _bare = [True] * len(_arr) + [False] * (len(groups) - len(_arr))
    else:
        groups = split_schema(user_schema)
        _bare = [False] * len(groups)

    # THE QUOTE-DROP LEVER (screened 2026-08-06, full-126 paired vs SEALED9:
    # headers +2.11pt p=0.036, overall +0.18pt p=0.50 — inside the ±0.25pt
    # baseline noise measured in the same session — items −0.29pt ns, and
    # −$0.00154/receipt net, 11% of the bill). Read off the ALREADY-COMPUTED
    # group plan, so it can never change WHICH fields are asked for — only the
    # envelope the answer comes back in. The array groups are already bare under
    # lean_arrays; this is what makes the SCALARS+AUX call bare too.
    #
    # WHAT IT COSTS, stated where it happens rather than in a doc: a bare group
    # has no quotes, so the strict-grounding gate CANNOT RUN on those fields.
    # Their verification is SKIPPED, not passed. They come back with no evidence
    # and they never appear in `ungrounded_fields` (the screen measured 56 → 0),
    # which means an unsupported value survives here that strict mode would have
    # nulled — the screen's own `subtotal` −7.4pt is exactly that: the model
    # deriving a subtotal from the total on receipts that never print one. The
    # trade was taken with eyes open because the gate was also destroying its own
    # right answers (part of the +2.11pt headers is nulls that should never have
    # happened), and because this lane's consumer reads `values` + `ocr_text` and
    # never citations. A customer who reads citations must not be given this.
    if settings.EXTRACT_BARE_SCALARS if bare_scalars is None else bare_scalars:
        _bare = [True] * len(groups)

    def _group_task(g, bare):
        return _aextract_group(
            batches,
            g,
            model,
            strict,
            chunk_ids,
            verify_grounding and not bare,
            wrapped and not bare,
            exempt_unquotable=exempt_unquotable,
            keep_repeated_rows=keep_repeated_rows,
            context_first=context_first,
            page_images=page_images,
            chunk_transform=chunk_transform,
            extra_instructions=extra_instructions,
            omit_null=_omit_null,
            short_row_keys=_short_row_keys,
            service_tier=service_tier,
            model_name=model_name,
            candidate_count=candidate_count,
        )

    if context_first and len(groups) > 1:
        # CACHE-WARM STAGGER: under the context-first layout every group call
        # shares an identical context+image prefix, but simultaneous calls give
        # Vertex implicit caching no writer before its readers (measured: 23%
        # hit rate under plain gather). Give the first call a head start so its
        # prefill lands in the cache the rest bill against; costs the stagger
        # delay in wall-clock, saves 90% on the shared prefix per later call.
        tasks = [asyncio.create_task(_group_task(groups[0], _bare[0]))]
        try:
            await asyncio.sleep(_stagger)
            # OBSERVE THE LEADER AT THE BOUNDARY. The head start is also long
            # enough for the leader to have FAILED in — a rejected schema or bad
            # credentials fails fast, well inside the stagger. This request cannot
            # return a partial schema, so launching the siblings on a dead leader
            # bills every one of them for an answer that is already unusable and
            # delays the error by their full duration. Re-raise it here instead;
            # on the success path this is a no-op await of a finished task.
            if tasks[0].done():
                await tasks[0]
            tasks.extend(
                asyncio.create_task(_group_task(g, b))
                for g, b in zip(groups[1:], _bare[1:], strict=True)
            )
            results = list(await _gather_groups(tasks))
        finally:
            await _drain_groups(tasks)
    else:
        tasks = [
            asyncio.create_task(_group_task(g, b))
            for g, b in zip(groups, _bare, strict=True)
        ]
        try:
            results = list(await _gather_groups(tasks))
        finally:
            await _drain_groups(tasks)

    if len(results) == 1:
        values, evidence, ungrounded, verified_by = results[0]
    else:  # re-unite disjoint field-groups
        values, evidence, ungrounded, verified_by = {}, {}, [], {}
        for v, e, u, vb in results:
            values = _merge_values(values, v)
            evidence.update(e)
            ungrounded.extend(u)
            verified_by.update(vb)

    # RECONCILIATION PASS (opt-in, best-of-N with the model as its own judge). One extra
    # call over every source this document has — text chunks, labelled secondary sources,
    # the page pixels — plus the first pass's answer as a labelled fallible CANDIDATE.
    # The reconciler answers in the SAME wrapped schema, so its leaves go through the SAME
    # grounding ladder; only a non-null leaf that grounds may replace a first-pass value.
    # Fail-open: any error keeps the first pass exactly as it was.
    recon_meta: dict[str, Any] | None = None
    if reconcile:
        if not (wrapped and verify_grounding):
            # Nothing gates a reconciler value here, and an unverified second opinion
            # overwriting a first one is a coin flip, not a lever.
            recon_meta = {"skipped": "ungated"}
        elif len(batches) > 1 or len(results) > 1:
            # ONE call means ONE context and ONE response schema. A document that needs
            # several context batches, or a schema wide enough to be split into field
            # groups, cannot be reconciled in one call — and the documents this exists
            # for (receipts, invoices) are neither.
            recon_meta = {"skipped": "multi_batch" if len(batches) > 1 else "wide_schema"}
        else:
            try:
                r_values, r_evidence, r_ungrounded, r_verified_by = await _areconcile(
                    batches[0],
                    user_schema,
                    values,
                    model,
                    strict=strict,
                    chunk_ids=chunk_ids,
                    images=reconcile_images,
                    chunk_transform=chunk_transform,
                    extra_instructions=extra_instructions,
                    exempt_unquotable=exempt_unquotable,
                    service_tier=service_tier,
                    model_name=model_name,
                    candidate_count=candidate_count,
                )
                counts = {"leaves_changed": 0, "leaves_kept": 0}
                replaced: list[str] = []
                merged_evidence = dict(evidence)
                values = _reconcile_merge(
                    values, r_values, "",
                    recon_ungrounded=set(r_ungrounded),
                    recon_evidence=r_evidence,
                    out_evidence=merged_evidence,
                    replaced=replaced,
                    counts=counts,
                )
                evidence = merged_evidence
                # A field the first pass could not ground is no longer ungrounded once a
                # GROUNDED reconciler value has taken its place.
                taken = set(replaced)
                ungrounded = [p for p in ungrounded if p not in taken]
                # The exemption diagnostic travels with the value it explains: a
                # replaced path takes the reconciler's reason, or loses the first
                # pass's if the reconciler's own value was quote-grounded.
                for p in taken:
                    if p in r_verified_by:
                        verified_by[p] = r_verified_by[p]
                    else:
                        verified_by.pop(p, None)
                recon_meta = counts
            except Exception:  # noqa: BLE001 — a second opinion must never fail an extraction
                log.warning("reconciliation pass failed; first-pass values kept", exc_info=True)
                recon_meta = {"skipped": "error"}

    return SchemaExtractResult(
        values=values,
        evidence=evidence,
        ungrounded_fields=ungrounded,
        verified_by=verified_by,
        reconcile=recon_meta,
    )


# --- {value,pages} first pass (plan 057) ------------------------------------
# An alternate first pass that returns each leaf as ``{value, pages}`` instead of
# ``{value, chunks, quote}``. It reuses the exact same schema validation, field-
# group splitting, and cross-batch merge machinery, but (a) renders the document
# context grouped by page (page number printed once per page, not per chunk) and
# (b) returns page-routing hints for the downstream image localizer instead of
# parse-chunk evidence. Values are produced the same way; only the citation
# signal changes. Coordinates are NEVER emitted here — that is the second pass.


VALUE_PAGES_EXTRACTION_PROMPT = """\
Extract values for the requested schema from the document context.

Return exactly the requested schema shape, but every leaf field must be an
object with:
- value: the extracted value, or null when missing.
- pages: the one-based page numbers where the value visibly appears.

Do not return chunk ids, quotes, bounding boxes, explanations, or extra keys.

Rules:
- Use only literal information present in the document context.
- Do not infer, calculate, normalize beyond obvious formatting, or fill from outside knowledge.
- If a value is missing, unreadable, masked, redacted, or only partially visible, return value=null and pages=[].
- If a non-null value appears on more than one page, return all pages where that same value is visibly present and supports the same field.
- If a field's label appears on a page but the value does not, do not include that page.
- If the same value appears on multiple unrelated pages and only some occurrences support this field, include only the supporting pages.
- For arrays, return every visible record in document order.
- For table rows, do not inherit blank cells from previous rows; blank cells are null.
- Preserve identifiers, dates, phone numbers, names, currency, and leading zeros as they appear unless the schema explicitly requires a different primitive type.
- If the schema type is boolean or enum, choose only when the document explicitly supports that value.

Document context:
"""


def wrap_schema_value_pages(node: dict[str, Any]) -> dict[str, Any]:
    """Wrap every leaf as ``{value, pages}`` (the plan-057 first pass), parallel to
    :func:`wrap_schema`'s ``{value, chunks, quote}``. Assumes a validated node."""
    t = _scalar_type(node.get("type", "string"))
    if t == "object":
        return {
            "type": "object",
            "properties": {k: wrap_schema_value_pages(v) for k, v in node.get("properties", {}).items()},
        }
    if t == "array":
        return {"type": "array", "items": wrap_schema_value_pages(node.get("items", {"type": "string"}))}
    leaf: dict[str, Any] = {"type": t, "nullable": True}
    enum = _clean_enum(node.get("enum"))
    if enum is not None:
        leaf["enum"] = enum
    return {
        "type": "object",
        "description": node.get("description", ""),
        "properties": {
            "value": leaf,
            "pages": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["value", "pages"],
    }


def build_value_pages_response_schema(user_schema: dict[str, Any]) -> dict[str, Any]:
    """User schema → ``{value,pages}``-wrapped → Gemini response_schema."""
    wrapped = {
        "type": "object",
        "properties": {k: wrap_schema_value_pages(v) for k, v in user_schema.items()},
    }
    return to_gemini_schema(wrapped)


def _batch_by_page(indexed: list[_IndexedChunk]) -> list[list[_IndexedChunk]]:
    """Page-aware batching: keep whole pages together under :data:`MAX_BATCH_CHARS`
    so the page-grouped renderer prints each page header exactly once. A single
    page larger than the budget is split across batches (header still emitted once
    per batch by the renderer); chunks stay in document order within a page."""
    by_page: dict[int, list[_IndexedChunk]] = {}
    order: list[int] = []
    for c in indexed:
        if c.page not in by_page:
            by_page[c.page] = []
            order.append(c.page)
        by_page[c.page].append(c)

    batches: list[list[_IndexedChunk]] = []
    cur: list[_IndexedChunk] = []
    size = 0
    for page in order:
        chunks = by_page[page]
        page_cost = sum(len(c.text) + 24 for c in chunks) + 20  # +page header framing
        if page_cost > MAX_BATCH_CHARS:  # one page overflows: split it across batches
            if cur:
                batches.append(cur)
                cur, size = [], 0
            sub: list[_IndexedChunk] = []
            ssize = 0
            for c in chunks:
                cost = len(c.text) + 24
                if sub and ssize + cost > MAX_BATCH_CHARS:
                    batches.append(sub)
                    sub, ssize = [], 0
                sub.append(c)
                ssize += cost
            if sub:
                batches.append(sub)
            continue
        if cur and size + page_cost > MAX_BATCH_CHARS:
            batches.append(cur)
            cur, size = [], 0
        cur.extend(chunks)
        size += page_cost
    if cur:
        batches.append(cur)
    return batches


def render_context_by_page(batch: list[_IndexedChunk]) -> str:
    """Page-grouped serialization: ``<page number="N"> blocks </page>`` with the
    page number printed once per page (the batch is in document order, so pages
    are contiguous)."""
    from itertools import groupby

    parts: list[str] = []
    for page, chunks in groupby(batch, key=lambda c: c.page):
        body = "\n".join(c.text for c in chunks)
        parts.append(f'<page number="{page}">\n{body}\n</page>')
    return "\n\n".join(parts)


def _unwrap_vp(node: Any) -> Any:
    """Walk a ``{value,pages}`` model result into a tree of :class:`_VPLeaf` cells /
    dicts / lists, mirroring :func:`_verify` for the chunks/quote path."""
    if isinstance(node, dict) and {"value", "pages"} <= node.keys() and not isinstance(node.get("value"), (dict, list)):
        pages = [p for p in (node.get("pages") or []) if isinstance(p, int)]
        return _VPLeaf(value=node.get("value"), pages=pages)
    if isinstance(node, dict):
        return {k: _unwrap_vp(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_unwrap_vp(v) for v in node]
    return _VPLeaf(value=node, pages=[])  # bare scalar (no wrapper)


def _merge_vp(a: Any, b: Any, diag: dict[str, int]) -> Any:
    """Merge two ``{value,pages}`` trees across batches: scalars take the first
    non-null value and UNION pages when the two non-null values agree (a conflict
    keeps the first and bumps ``value_conflicts``); arrays concatenate+dedup;
    objects merge per key."""
    if isinstance(a, _VPLeaf) and isinstance(b, _VPLeaf):
        if a.value is None:
            return b
        if b.value is None:
            return a
        if _norm(str(a.value)) == _norm(str(b.value)):
            return _VPLeaf(value=a.value, pages=sorted(set(a.pages) | set(b.pages)))
        diag["value_conflicts"] = diag.get("value_conflicts", 0) + 1
        return a
    if isinstance(a, list) and isinstance(b, list):
        return _dedup_records(a + b)
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge_vp(a[k], b[k], diag) if k in b else a[k] for k in a} | {
            k: v for k, v in b.items() if k not in a
        }
    return a if _has_content(a) else b


def _flatten_vp(node: Any, path: str, page_hints: dict[str, list[int]]) -> Any:
    """Flatten a :class:`_VPLeaf` tree → value tree, filling ``page_hints`` keyed by
    concrete field path (``a.b`` / ``a[0]``, same convention as evidence keys)."""
    if isinstance(node, _VPLeaf):
        if node.value is not None and node.pages:
            page_hints[path] = sorted(dict.fromkeys(node.pages))
        return node.value
    if isinstance(node, dict):
        return {k: _flatten_vp(v, f"{path}.{k}" if path else str(k), page_hints) for k, v in node.items()}
    if isinstance(node, list):
        return [_flatten_vp(v, f"{path}[{i}]", page_hints) for i, v in enumerate(node)]
    return node


async def _aextract_group_vp(
    batches: list[list[_IndexedChunk]],
    group_schema: dict[str, Any],
    model: SchemaModel,
    diag: dict[str, int],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Extract one field-group across page-grouped batches → ``(values, page_hints)``."""
    response_schema = build_value_pages_response_schema(group_schema)
    raws = await asyncio.gather(
        *(model.generate_json(VALUE_PAGES_EXTRACTION_PROMPT + render_context_by_page(b), response_schema)
          for b in batches)
    )
    trees = [_unwrap_vp(raw) for raw in raws]
    merged = trees[0]
    for tree in trees[1:]:
        merged = _merge_vp(merged, tree, diag)
    page_hints: dict[str, list[int]] = {}
    values = _flatten_vp(merged, "", page_hints)
    return (values if isinstance(values, dict) else {}), page_hints


async def aextract_schema_values_and_pages(
    chunks: list[Chunk],
    page_sizes: list[tuple[float, float]],
    user_schema: dict[str, Any],
    *,
    model: SchemaModel,
) -> tuple[dict[str, Any], dict[str, list[int]], dict[str, int]]:
    """The plan-057 ``{value,pages}`` first pass: fill ``user_schema`` and return
    page hints (not final evidence) for the downstream image localizer.

    Same validation, field-group splitting, and cross-batch merge as
    :func:`aextract_schema`; the differences are the page-grouped context renderer,
    page-aware batching, and the ``{value,pages}`` response wrapper. Returns
    ``(values, page_hints, diagnostics)`` where ``page_hints`` maps a concrete field
    path to its one-based page numbers and ``diagnostics`` carries batch/group
    counts and the cross-batch value-conflict count.
    """
    user_schema = flatten_schema(user_schema)
    validate_schema(user_schema)
    diag: dict[str, int] = {"n_batches": 0, "n_groups": 0, "value_conflicts": 0}
    indexed = _index_chunks(chunks, page_sizes)
    if not indexed:
        return {k: None for k in user_schema}, {}, diag

    batches = _batch_by_page(indexed)
    groups = split_schema(user_schema)
    diag["n_batches"] = len(batches)
    diag["n_groups"] = len(groups)
    results = await asyncio.gather(
        *(_aextract_group_vp(batches, g, model, diag) for g in groups)
    )

    values: dict[str, Any] = {}
    page_hints: dict[str, list[int]] = {}
    for v, ph in results:
        values = _merge_values(values, v)
        page_hints.update(ph)
    return values, page_hints, diag
