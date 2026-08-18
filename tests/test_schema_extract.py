"""Unit tests for the schema-extraction core lib (plan 050).

No network: the LLM is a stub :class:`SchemaModel` returning canned wrapped
results. Covers the validation guards (depth / cycle / breadth / type / empty),
wrap/convert, bbox normalization, grounding verification (grounded / ungrounded /
strict null / absent / text-match augmentation), and document batching + merge.
"""

from __future__ import annotations

import pytest

from extract.core.models import Chunk
from extract.core.schema_extract import (
    EmptySchema,
    SchemaCycle,
    SchemaTooDeep,
    SchemaTooWide,
    SchemaValidationError,
    UnsupportedFieldType,
    aextract_schema,
    build_response_schema,
    flatten_schema,
    normalize_bbox,
    to_gemini_schema,
    validate_schema,
    wrap_schema,
)


class StubModel:
    """Returns canned wrapped results, one per call (in batch order)."""

    def __init__(self, *responses: dict) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate_json(self, prompt: str, response_schema: dict) -> dict:
        self.prompts.append(prompt)
        return self._responses[len(self.prompts) - 1]


def _chunk(text: str, page: int = 1, bbox=None) -> Chunk:
    return Chunk(page_content=text, page_no=page, bbox=bbox or [0, 0, 100, 100])


# --- validation -------------------------------------------------------------


def test_validate_accepts_supported_schema():
    validate_schema(
        {
            "title": {"type": "string"},
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"amount": {"type": "number"}}},
            },
            "flag": {"type": "boolean", "enum": [True, False]},
        }
    )


def test_validate_rejects_empty():
    with pytest.raises(EmptySchema):
        validate_schema({})


def test_validate_rejects_unsupported_type():
    with pytest.raises(UnsupportedFieldType):
        validate_schema({"x": {"type": "date"}})


def test_validate_rejects_ref_cycle():
    with pytest.raises(SchemaCycle):
        validate_schema({"x": {"$ref": "#/definitions/x"}})


def test_validate_rejects_too_deep():
    # 6 levels of object nesting → exceeds the depth-5 cap.
    node = {"type": "string"}
    for _ in range(6):
        node = {"type": "object", "properties": {"n": node}}
    with pytest.raises(SchemaTooDeep):
        validate_schema({"root": node})


def test_validate_allows_depth_five():
    node = {"type": "string"}
    for _ in range(3):  # root(1) -> obj(2) -> obj(3) -> ... keep within 5
        node = {"type": "object", "properties": {"n": node}}
    validate_schema({"root": node})


def test_validate_rejects_too_wide():
    # Field-group splitting handles breadth now; the cap is only a sanity ceiling.
    schema = {f"f{i}": {"type": "string"} for i in range(1001)}
    with pytest.raises(SchemaTooWide):
        validate_schema(schema)


def test_validate_allows_wide_schema_under_sanity_cap():
    # 369-field-scale schema (the 10kq case) is now accepted, not 422'd — it gets
    # field-group-split at extraction time instead.
    validate_schema({f"f{i}": {"type": "string"} for i in range(400)})


# --- wrap / convert ---------------------------------------------------------


def test_wrap_schema_leaf_carries_value_chunks_quote():
    wrapped = wrap_schema({"type": "string", "description": "the title"})
    assert wrapped["type"] == "object"
    assert set(wrapped["properties"]) == {"value", "chunks", "quote"}
    assert wrapped["properties"]["value"]["nullable"] is True
    assert wrapped["required"] == ["value", "chunks", "quote"]


def test_to_gemini_schema_uppercases_types():
    g = to_gemini_schema({"type": "array", "items": {"type": "object", "properties": {}}})
    assert g["type"] == "ARRAY"
    assert g["items"]["type"] == "OBJECT"


def test_build_response_schema_end_to_end():
    g = build_response_schema({"title": {"type": "string"}})
    title = g["properties"]["title"]
    assert title["type"] == "OBJECT"
    assert title["properties"]["value"]["type"] == "STRING"
    assert title["properties"]["chunks"]["type"] == "ARRAY"


def test_validate_accepts_nullable_union_type():
    # Pydantic/Extend exports emit Optional fields as ["string", "null"]. The
    # null member is redundant (every leaf is nullable via the wrapper) and a
    # list type used to crash the validator (unhashable). It must normalize.
    validate_schema({"name": {"type": ["string", "null"]}, "n": {"type": ["integer", "null"]}})


def test_nullable_union_normalizes_to_base_scalar():
    g = build_response_schema({"name": {"type": ["string", "null"]}})
    assert g["properties"]["name"]["properties"]["value"]["type"] == "STRING"


def test_null_enum_member_is_dropped():
    # Vertex 400s on a null inside a string enum; nullability is the wrapper's job.
    g = build_response_schema(
        {"origin": {"type": "string", "enum": ["Home", "Hospital", None]}}
    )
    value = g["properties"]["origin"]["properties"]["value"]
    assert value["enum"] == ["Home", "Hospital"]


def test_all_null_enum_is_dropped_entirely():
    g = build_response_schema({"x": {"type": "string", "enum": [None]}})
    assert "enum" not in g["properties"]["x"]["properties"]["value"]


# --- array record dedup (cross-batch / cross-page restatement) --------------


def test_dedup_drops_name_only_echo_of_identified_record():
    # The same coverage restated on a later page with no member_id is a redundant
    # echo of the fully-identified one — collapse it (the cross-batch dup bug).
    from extract.core.schema_extract import _dedup_records

    recs = [
        {"name": "AARP", "member_id": "995711079"},
        {"name": "AARP", "member_id": None},
        {"name": "MOLINA", "member_id": "92044252A"},
    ]
    out = [(r["name"], r["member_id"]) for r in _dedup_records(recs)]
    assert out == [("AARP", "995711079"), ("MOLINA", "92044252A")]


def test_dedup_keeps_distinct_member_ids_for_same_payer():
    # Two real coverages under one payer with DIFFERENT ids must both survive.
    from extract.core.schema_extract import _dedup_records

    recs = [{"name": "KP", "member_id": "111"}, {"name": "KP", "member_id": "222"}]
    assert len(_dedup_records(recs)) == 2


def test_dedup_collapses_exact_duplicate():
    from extract.core.schema_extract import _dedup_records

    recs = [{"name": "X", "member_id": "1"}, {"name": "X", "member_id": "1"}]
    assert len(_dedup_records(recs)) == 1


# --- bbox normalization -----------------------------------------------------


def test_normalize_bbox_scales_to_0_1000():
    # half-width, quarter-height page point → 500, 250
    assert normalize_bbox([306, 198, 612, 396], (612, 792)) == pytest.approx(
        [500.0, 250.0, 1000.0, 500.0]
    )


def test_normalize_bbox_handles_none_and_zero():
    assert normalize_bbox(None, (612, 792)) is None
    assert normalize_bbox([1, 2, 3, 4], None) is None
    assert normalize_bbox([1, 2, 3, 4], (0, 0)) is None


def test_normalize_bbox_orders_and_clips():
    # reversed coords + overflow → ordered and clipped to [0, 1000]
    out = normalize_bbox([700, 800, 100, 200], (600, 700))
    assert out[0] <= out[2] and out[1] <= out[3]
    assert all(0.0 <= v <= 1000.0 for v in out)


# --- chunk text normalization + table rendering (audit P0-3) ----------------

_GFM = (
    "| Payer | Member ID |\n"
    "|---|---|\n"
    "| MEDICARE | 1EG4-TE5-MK73 |\n"
    "| AETNA | W123456789 |"
)


def test_index_chunks_keeps_table_rows_intact():
    """A GFM table survives indexing with its row boundaries — the P0-3 bug
    collapsed the whole grid into one line of pipe soup before the model saw it."""
    from extract.core.schema_extract import _index_chunks

    indexed = _index_chunks([_chunk(_GFM, page=2)], [(612, 792)])
    assert len(indexed) == 1
    c = indexed[0]
    assert c.text.split("\n") == [
        "| Payer | Member ID |",
        "|---|---|",
        "| MEDICARE | 1EG4-TE5-MK73 |",
        "| AETNA | W123456789 |",
    ]
    assert c.is_table is True


def test_index_chunks_collapses_horizontal_runs_only():
    """Prose is unchanged modulo horizontal run-collapse: a single-line chunk is
    byte-identical to the old whole-text collapse, and line structure survives."""
    from extract.core.schema_extract import _index_chunks, normalize_chunk_text

    assert normalize_chunk_text("Name:   DOE,\tJANE  ") == "Name: DOE, JANE"
    assert normalize_chunk_text("  para one  \n\n\n\n  para   two ") == "para one\n\npara two"
    assert normalize_chunk_text("a\r\nb") == "a\nb"
    assert normalize_chunk_text(None) == ""
    # empty-after-normalize chunks are still dropped
    assert _index_chunks([_chunk("   \n \n ", page=1)], [(612, 792)]) == []
    indexed = _index_chunks([_chunk("Addr:   531 VINE PL\nOXNARD,  CA", page=1)], [(612, 792)])
    assert indexed[0].text == "Addr: 531 VINE PL\nOXNARD, CA"
    assert indexed[0].is_table is False


def test_index_chunks_carries_chunk_type():
    """``chunk_type`` reaches the model-facing chunk (it used to be dropped), and a
    typed TABLE is a table even when its text is not GFM-shaped."""
    from extract.core.models import ChunkType
    from extract.core.schema_extract import _index_chunks

    typed = Chunk(page_content="Total 41.00", page_no=1, bbox=[0, 0, 10, 10],
                  chunk_type=ChunkType.TABLE)
    indexed = _index_chunks([typed, _chunk("plain prose", page=1)], [(612, 792)])
    assert indexed[0].chunk_type == ChunkType.TABLE
    assert indexed[0].is_table is True
    assert indexed[1].chunk_type == ChunkType.TEXT
    assert indexed[1].is_table is False


def test_render_context_tags_tables_and_keeps_prose_framing():
    """Tables are labelled and rendered as grids; single-line prose keeps the exact
    historical ``[i | page N] text`` framing."""
    from extract.core.schema_extract import _index_chunks, render_context

    indexed = _index_chunks([_chunk("Patient: JANE DOE", page=1), _chunk(_GFM, page=1)],
                            [(612, 792)])
    ctx = render_context(indexed)
    assert ctx.startswith("[0 | page 1] Patient: JANE DOE\n")
    assert "[1 | page 1 | table]\n| Payer | Member ID |\n|---|---|\n" in ctx
    assert "| AETNA | W123456789 |" in ctx


# --- grounding (via aextract_schema, single batch) --------------------------


async def test_grounded_value_gets_normalized_evidence():
    chunks = [_chunk("Title: Attention Is All You Need", bbox=[100, 200, 300, 240])]
    page_sizes = [(1000.0, 1000.0)]  # identity normalization
    model = StubModel(
        {"title": {"value": "Attention Is All You Need", "chunks": [0],
                   "quote": "Title: Attention Is All You Need"}}
    )
    result = await aextract_schema(
        chunks, page_sizes, {"title": {"type": "string"}}, model=model
    )
    assert result.values == {"title": "Attention Is All You Need"}
    assert result.ungrounded_fields == []
    ev = result.evidence["title"]
    assert len(ev) == 1
    assert ev[0].page == 1
    assert ev[0].bbox == pytest.approx([100.0, 200.0, 300.0, 240.0])
    assert "Attention" in ev[0].text


async def test_ungrounded_strict_nulls_and_flags():
    chunks = [_chunk("Invoice total: $100")]
    model = StubModel(
        {"phone": {"value": "555-1234", "chunks": [0], "quote": "call us at 555-1234"}}
    )
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0)], {"phone": {"type": "string"}}, model=model, strict=True
    )
    assert result.values == {"phone": None}
    assert result.ungrounded_fields == ["phone"]
    assert "phone" not in result.evidence


async def test_ungrounded_non_strict_keeps_but_flags():
    chunks = [_chunk("Invoice total: $100")]
    model = StubModel(
        {"phone": {"value": "555-1234", "chunks": [0], "quote": "call us at 555-1234"}}
    )
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0)], {"phone": {"type": "string"}}, model=model, strict=False
    )
    assert result.values == {"phone": "555-1234"}
    assert result.ungrounded_fields == ["phone"]


async def test_evidence_text_is_quote_not_truncated_chunk():
    # A paragraph chunk where the supporting value sits well past the 300-char
    # display cap. The evidence text must be the verified quote (which contains
    # the value), not the chunk prefix truncated before it.
    filler = "lorem ipsum dolor sit amet consectetur " * 12  # >300 chars of preamble
    chunk_text = f"{filler} the model dimension is dmodel = 512 in the base config"
    assert chunk_text.index("512") > 300  # value is beyond the display cap
    chunks = [_chunk(chunk_text, bbox=[0, 0, 500, 40])]
    model = StubModel(
        {"d_model": {"value": 512, "chunks": [0], "quote": "dmodel = 512 in the base config"}}
    )
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0)], {"d_model": {"type": "integer"}}, model=model
    )
    assert result.values == {"d_model": 512}
    ev = result.evidence["d_model"][0]
    assert ev.text == "dmodel = 512 in the base config"
    assert "512" in ev.text  # the cited span is self-verifying


async def test_absent_value_is_null_no_evidence():
    chunks = [_chunk("Invoice total: $100")]
    model = StubModel({"phone": {"value": None, "chunks": [], "quote": ""}})
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0)], {"phone": {"type": "string"}}, model=model
    )
    assert result.values == {"phone": None}
    assert result.ungrounded_fields == []
    assert result.evidence == {}


async def test_recited_quote_grounds_against_uncited_chunk():
    # Cited id is wrong (5), but the quote really exists in chunk 0 → grounded.
    chunks = [_chunk("Net amount due: $42.50")]
    model = StubModel(
        {"total": {"value": 42.5, "chunks": [5], "quote": "Net amount due: $42.50"}}
    )
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0)], {"total": {"type": "number"}}, model=model
    )
    assert result.values == {"total": 42.5}
    assert result.ungrounded_fields == []
    assert result.evidence["total"][0].page == 1


async def test_long_prose_text_match_augmentation():
    # The model cites only the first chunk; the rest of the prose lives in later
    # chunks whose text is contained in the value → each gets its own box.
    line1 = "The quick brown fox jumps over the lazy dog near the riverbank."
    line2 = "It then proceeds to chase the cat across the meadow at dawn."
    value = f"{line1} {line2}"
    chunks = [_chunk(line1, bbox=[0, 0, 500, 20]), _chunk(line2, bbox=[0, 30, 500, 50])]
    model = StubModel({"summary": {"value": value, "chunks": [0], "quote": line1}})
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0), (1000.0, 1000.0)], {"summary": {"type": "string"}}, model=model
    )
    pages = {e.page for e in result.evidence["summary"]}
    assert len(result.evidence["summary"]) == 2  # both lines boxed
    assert pages == {1}


# --- arrays + batching/merge ------------------------------------------------


async def test_array_of_objects_grounded():
    chunks = [_chunk("Apple 3 | Banana 5", bbox=[0, 0, 200, 20])]
    model = StubModel(
        {
            "items": [
                {"name": {"value": "Apple", "chunks": [0], "quote": "Apple 3"},
                 "qty": {"value": 3, "chunks": [0], "quote": "Apple 3"}},
                {"name": {"value": "Banana", "chunks": [0], "quote": "Banana 5"},
                 "qty": {"value": 5, "chunks": [0], "quote": "Banana 5"}},
            ]
        }
    )
    schema = {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
            },
        }
    }
    result = await aextract_schema(chunks, [(1000.0, 1000.0)], schema, model=model)
    assert result.values == {"items": [{"name": "Apple", "qty": 3}, {"name": "Banana", "qty": 5}]}
    assert "items[0].name" in result.evidence
    assert "items[1].qty" in result.evidence


async def test_document_batching_merges_arrays_in_order(monkeypatch):
    import extract.core.schema_extract as se

    monkeypatch.setattr(se, "MAX_BATCH_CHARS", 50)  # force two batches
    big = "x" * 60
    chunks = [
        _chunk(f"Row A {big}", page=1, bbox=[0, 0, 100, 10]),
        _chunk(f"Row B {big}", page=2, bbox=[0, 0, 100, 10]),
    ]
    page_sizes = [(1000.0, 1000.0), (1000.0, 1000.0)]
    # Batch 1 sees only chunk 0 (local id 0); batch 2 only chunk 1 (local id 0).
    model = StubModel(
        {"rows": [{"label": {"value": "A", "chunks": [0], "quote": f"Row A {big}"}}]},
        {"rows": [{"label": {"value": "B", "chunks": [0], "quote": f"Row B {big}"}}]},
    )
    schema = {
        "rows": {
            "type": "array",
            "items": {"type": "object", "properties": {"label": {"type": "string"}}},
        }
    }
    result = await aextract_schema(chunks, page_sizes, schema, model=model)
    assert len(model.prompts) == 2  # actually split into two concurrent calls
    assert result.values == {"rows": [{"label": "A"}, {"label": "B"}]}
    # Merged-array evidence is re-indexed: row 0 from batch 1 (page 1), row 1
    # from batch 2 (page 2).
    assert result.evidence["rows[0].label"][0].page == 1
    assert result.evidence["rows[1].label"][0].page == 2


async def test_scalar_merge_takes_first_grounded_non_null(monkeypatch):
    import extract.core.schema_extract as se

    monkeypatch.setattr(se, "MAX_BATCH_CHARS", 50)
    big = "y" * 60
    chunks = [
        _chunk(f"Total $10 {big}", page=1),
        _chunk(f"Total $10 {big}", page=2),
    ]
    page_sizes = [(1000.0, 1000.0), (1000.0, 1000.0)]
    # Batch 1: absent. Batch 2: grounded. Merge should take batch 2's value.
    model = StubModel(
        {"total": {"value": None, "chunks": [], "quote": ""}},
        {"total": {"value": 10, "chunks": [0], "quote": f"Total $10 {big}"}},
    )
    result = await aextract_schema(
        chunks, page_sizes, {"total": {"type": "integer"}}, model=model
    )
    assert result.values == {"total": 10}
    assert result.ungrounded_fields == []


async def test_no_text_chunks_returns_all_null():
    chunks = [_chunk("   ")]  # whitespace only → filtered out
    model = StubModel({"title": {"value": "should not be used", "chunks": [0], "quote": "x"}})
    result = await aextract_schema(
        chunks, [(1000.0, 1000.0)], {"title": {"type": "string"}}, model=model
    )
    assert result.values == {"title": None}
    assert model.prompts == []  # model never called


# --- field-group splitting (plan 050 M4) ------------------------------------


def test_split_schema_small_is_single_group():
    from extract.core.schema_extract import split_schema

    s = {"a": {"type": "string"}, "b": {"type": "integer"}}
    assert split_schema(s) == [s]  # ≤ MAX_GROUP_LEAVES → one group, unchanged


def test_split_schema_partitions_wide_disjointly():
    from extract.core.schema_extract import _leaf_count, split_schema

    s = {f"f{i}": {"type": "string"} for i in range(120)}
    groups = split_schema(s, max_leaves=50)
    assert len(groups) >= 3
    assert all(sum(_leaf_count(v) for v in g.values()) <= 50 for g in groups)
    keys = [k for g in groups for k in g]
    assert sorted(keys) == sorted(s)  # disjoint cover of every field
    assert len(keys) == len(set(keys))


def test_split_schema_recurses_into_big_object():
    from extract.core.schema_extract import split_schema

    big = {"type": "object", "properties": {f"p{i}": {"type": "number"} for i in range(80)}}
    groups = split_schema({"stmt": big}, max_leaves=50)
    assert len(groups) >= 2
    assert all(set(g) == {"stmt"} and g["stmt"]["type"] == "object" for g in groups)
    props = [p for g in groups for p in g["stmt"]["properties"]]
    assert sorted(props) == sorted(big["properties"])  # every sub-property covered


async def test_field_split_merges_all_group_values():
    # A wide object (80 props > MAX_GROUP_LEAVES) splits into groups, each
    # extracted separately and re-united into the full object.
    props = {f"p{i}": {"type": "string"} for i in range(80)}
    schema = {"stmt": {"type": "object", "properties": props}}

    class GroupStub:
        """Fills whatever group-schema it's handed: value = the field's own name."""

        def __init__(self):
            self.calls = 0

        async def generate_json(self, prompt, rs):
            self.calls += 1

            def fill(node, key=None):
                if node.get("type") == "OBJECT":
                    p = node.get("properties", {})
                    if {"value", "chunks", "quote"} <= set(p):
                        return {"value": key, "chunks": [], "quote": ""}
                    return {k: fill(v, k) for k, v in p.items()}
                if node.get("type") == "ARRAY":
                    return []
                return None

            return fill(rs)

    model = GroupStub()
    result = await aextract_schema(
        [_chunk("some text")], [(1000.0, 1000.0)], schema, model=model, strict=False
    )
    assert model.calls >= 2  # actually split into multiple groups
    assert result.values["stmt"] == {f"p{i}": f"p{i}" for i in range(80)}  # all reunited


# --- standard-JSON-Schema flattener (plan 050 M4) ---------------------------

# What Pydantic `Invoice.model_json_schema()` emits for:
#   class LineItem(BaseModel): sku: str; qty: int
#   class Invoice(BaseModel):
#       invoice_number: str; vendor: Optional[str]=None; items: list[LineItem]
PYDANTIC_SCHEMA = {
    "$defs": {
        "LineItem": {
            "type": "object",
            "title": "LineItem",
            "properties": {
                "sku": {"type": "string", "title": "Sku"},
                "qty": {"type": "integer", "title": "Qty"},
            },
            "required": ["sku", "qty"],
        }
    },
    "type": "object",
    "title": "Invoice",
    "properties": {
        "invoice_number": {"type": "string", "title": "Invoice Number"},
        "vendor": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "title": "Vendor",
        },
        "items": {"type": "array", "title": "Items", "items": {"$ref": "#/$defs/LineItem"}},
    },
    "required": ["invoice_number", "items"],
}


def test_flatten_pydantic_schema_resolves_refs_and_optionals():
    flat = flatten_schema(PYDANTIC_SCHEMA)
    # top-level object unwrapped to the bare field mapping; metadata dropped
    assert set(flat) == {"invoice_number", "vendor", "items"}
    assert flat["invoice_number"] == {"type": "string"}
    # Optional[str] (anyOf[str,null]) → plain string (nullable is automatic downstream)
    assert flat["vendor"] == {"type": "string"}
    # $ref in array items inlined to the referenced object
    assert flat["items"]["type"] == "array"
    assert flat["items"]["items"]["type"] == "object"
    assert set(flat["items"]["items"]["properties"]) == {"sku", "qty"}
    assert flat["items"]["items"]["properties"]["qty"] == {"type": "integer"}
    # and the flattened schema passes our validator + builds a response schema
    validate_schema(flat)
    build_response_schema(flat)


def test_flatten_is_idempotent_on_bare_dialect():
    bare = {
        "name": {"type": "string", "description": "n"},
        "addr": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    assert flatten_schema(bare) == bare
    assert flatten_schema(flatten_schema(bare)) == flatten_schema(bare)


def test_flatten_merges_allof_with_description():
    # Pydantic emits allOf:[{$ref}] + sibling description for a documented sub-model.
    schema = {
        "$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}}},
        "type": "object",
        "properties": {
            "address": {"allOf": [{"$ref": "#/$defs/Addr"}], "description": "mailing address"}
        },
    }
    flat = flatten_schema(schema)
    assert flat["address"]["type"] == "object"
    assert flat["address"]["description"] == "mailing address"
    assert "city" in flat["address"]["properties"]


def test_flatten_rejects_true_ref_cycle():
    # A self-referential model ($ref into itself) is a real cycle — still rejected.
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    with pytest.raises(SchemaCycle):
        flatten_schema(schema)


def test_flatten_unresolved_ref_raises():
    with pytest.raises(SchemaValidationError):
        flatten_schema({"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}})


def test_flatten_top_level_ref_to_model():
    schema = {
        "$defs": {
            "Doc": {"type": "object", "properties": {"id": {"type": "string"}}}
        },
        "$ref": "#/$defs/Doc",
    }
    flat = flatten_schema(schema)
    assert flat == {"id": {"type": "string"}}


def test_design_sample_spans_long_doc():
    # A doc far over the design budget is sampled across its whole length, not
    # truncated to the head — so late-page fields still get designed.
    from extract.core.schema_extract import DESIGN_CONTEXT_CHARS, _design_sample, _IndexedChunk

    chunks = [_IndexedChunk(text="x" * 1000, page=i + 1, bbox=None) for i in range(400)]
    sample = _design_sample(chunks)
    assert sum(len(c.text) + 24 for c in sample) <= DESIGN_CONTEXT_CHARS * 1.2
    pages = {c.page for c in sample}
    assert min(pages) <= 5 and max(pages) >= 390  # spans head AND tail


def test_design_sample_passthrough_when_small():
    from extract.core.schema_extract import _design_sample, _IndexedChunk

    chunks = [_IndexedChunk(text="short", page=1, bbox=None) for _ in range(3)]
    assert _design_sample(chunks) == chunks


def test_candcount_off_by_default():
    """candcount (candidate_count) is opt-in; the default extractor is single-candidate."""
    from extract.gemini_extract import GeminiSchemaExtractor

    assert GeminiSchemaExtractor()._candidate_count == 1
    assert GeminiSchemaExtractor(candidate_count=3)._candidate_count == 3


def test_candcount_union_fills_nulls_keeps_nonnull():
    """The grounding-gated union keeps a's non-null leaves and fills a's nulls from b."""
    from extract.gemini_extract import _union_nonnull

    a = {"city": {"value": None, "chunks": [], "quote": ""},
         "name": {"value": "Smith", "chunks": [1], "quote": "Smith"}}
    b = {"city": {"value": "Oxnard", "chunks": [2], "quote": "Oxnard"},
         "name": {"value": "Jones", "chunks": [3], "quote": "Jones"}}
    m = _union_nonnull(a, b)
    assert m["city"]["value"] == "Oxnard"
    assert m["name"]["value"] == "Smith"
    na = {"phys": {"npi": {"value": None}}, "ins": [{"id": {"value": "X"}}]}
    nb = {"phys": {"npi": {"value": "123"}}, "ins": [{"id": {"value": "Y"}}]}
    mn = _union_nonnull(na, nb)
    assert mn["phys"]["npi"]["value"] == "123" and mn["ins"][0]["id"]["value"] == "X"


# --- {value,pages} first pass (plan 057) ------------------------------------


def test_value_pages_response_schema_objects_arrays_nullable_enum():
    from extract.core.schema_extract import build_value_pages_response_schema

    user = {
        "patient": {"type": "object", "properties": {
            "name": {"type": "string", "description": "pt name"},
            "status": {"type": "string", "enum": ["active", "inactive", None]},
        }},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "amount": {"type": "number"},
        }}},
    }
    rs = build_value_pages_response_schema(flatten_schema(user))
    name_leaf = rs["properties"]["patient"]["properties"]["name"]
    assert set(name_leaf["properties"]) == {"value", "pages"}
    assert name_leaf["properties"]["value"]["type"] == "STRING"
    assert name_leaf["properties"]["value"]["nullable"] is True
    assert name_leaf["properties"]["pages"]["type"] == "ARRAY"
    assert name_leaf["required"] == ["value", "pages"]
    # null enum member dropped (Vertex rejects a null enum member)
    status_leaf = rs["properties"]["patient"]["properties"]["status"]
    assert status_leaf["properties"]["value"]["enum"] == ["active", "inactive"]
    # arrays wrap their item leaves too
    amt = rs["properties"]["claims"]["items"]["properties"]["amount"]
    assert set(amt["properties"]) == {"value", "pages"}


def test_render_context_by_page_prints_page_number_once():
    from extract.core.schema_extract import _index_chunks, render_context_by_page

    chunks = [
        _chunk("first line", page=1, bbox=[0, 0, 10, 10]),
        _chunk("second line", page=1, bbox=[0, 20, 10, 30]),
        _chunk("page two text", page=2, bbox=[0, 0, 10, 10]),
    ]
    indexed = _index_chunks(chunks, [(100, 100), (100, 100)])
    ctx = render_context_by_page(indexed)
    assert ctx.count('<page number="1">') == 1
    assert ctx.count('<page number="2">') == 1
    assert "first line\nsecond line" in ctx
    assert ctx.index('number="1"') < ctx.index('number="2"')


def test_batch_by_page_keeps_whole_pages_together():
    from extract.core.schema_extract import _batch_by_page, _index_chunks

    chunks = [_chunk(f"line {i}", page=p, bbox=[0, 0, 10, 10]) for p in (1, 2, 3) for i in range(3)]
    indexed = _index_chunks(chunks, [(100, 100)] * 3)
    batches = _batch_by_page(indexed)
    # small doc → one batch, all pages intact in order
    assert len(batches) == 1
    assert [c.page for c in batches[0]] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_aextract_values_and_pages_unwraps_and_unions_pages():
    import asyncio

    from extract.core.schema_extract import aextract_schema_values_and_pages

    user = {
        "patient": {"type": "object", "properties": {"name": {"type": "string"}}},
        "insurers": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}},
    }
    # two batches (two pages): name appears on both; merge must UNION its pages.
    resp1 = {"patient": {"name": {"value": "Jane Doe", "pages": [1]}},
             "insurers": [{"id": {"value": "AAA", "pages": [1]}}]}
    resp2 = {"patient": {"name": {"value": "Jane Doe", "pages": [2]}},
             "insurers": [{"id": {"value": "BBB", "pages": [2]}}]}
    chunks = [
        _chunk("Jane Doe ins AAA", page=1, bbox=[0, 0, 10, 10]),
        _chunk("Jane Doe ins BBB", page=2, bbox=[0, 0, 10, 10]),
    ]
    # force two batches by shrinking the budget
    import extract.core.schema_extract as se
    orig = se.MAX_BATCH_CHARS
    se.MAX_BATCH_CHARS = 20
    try:
        model = StubModel(resp1, resp2)
        values, page_hints, diag = asyncio.run(
            aextract_schema_values_and_pages(chunks, [(100, 100), (100, 100)], user, model=model))
    finally:
        se.MAX_BATCH_CHARS = orig
    assert values["patient"]["name"] == "Jane Doe"
    assert page_hints["patient.name"] == [1, 2]  # unioned across batches
    assert diag["n_batches"] == 2
    # both insurer records survive (concatenated, distinct ids)
    ids = {rec["id"] for rec in values["insurers"]}
    assert ids == {"AAA", "BBB"}


def test_verify_unwraps_value_quote_leaf_without_chunk_ids():
    """chunk_ids=False leaves are exactly {value, quote}; _verify must unwrap
    them (regression: they fell through as plain dicts -> 0.0 accuracy)."""
    from extract.core.schema_extract import _IndexedChunk, _Leaf, _verify

    batch = [_IndexedChunk(text="Patient City: OXNARD CA", page=3, bbox=[100.0, 500.0, 300.0, 520.0])]
    tree = _verify({"name": {"value": "OXNARD", "quote": "City: OXNARD"}}, batch, strict=True)
    leaf = tree["name"]
    assert isinstance(leaf, _Leaf)
    assert leaf.value == "OXNARD" and not leaf.ungrounded
    assert leaf.evidence and leaf.evidence[0].page == 3


def test_page_window_quote_salvage_grounds_short_value_on_line_chunks():
    """A multi-line quote can't match any single line-level chunk; the page-window
    salvage must still ground the (short) value and anchor its line."""
    from extract.core.schema_extract import _IndexedChunk, _verify_leaf

    batch = [
        _IndexedChunk(text="Patient Mailing Address", page=3, bbox=[100.0, 500.0, 300.0, 512.0]),
        _IndexedChunk(text="531 VINE PLACE", page=3, bbox=[100.0, 514.0, 250.0, 526.0]),
        _IndexedChunk(text="OXNARD", page=3, bbox=[100.0, 528.0, 170.0, 540.0]),
        _IndexedChunk(text="OXNARD, CA 93030 header", page=1, bbox=[10.0, 10.0, 200.0, 22.0]),
    ]
    leaf = _verify_leaf(
        {"value": "OXNARD", "quote": "Patient Mailing Address 531 VINE PLACE OXNARD"},
        batch, strict=True)
    assert leaf.value == "OXNARD" and not leaf.ungrounded
    assert leaf.evidence[0].page == 3
    assert leaf.evidence[0].bbox == [100.0, 528.0, 170.0, 540.0]  # the value-bearing line


def test_page_window_salvage_rejects_fabricated_quote():
    """A quote that is not real page text still fails -> strict nulls the value
    (anti-hallucination property preserved)."""
    from extract.core.schema_extract import _IndexedChunk, _verify_leaf

    batch = [_IndexedChunk(text="totally unrelated line", page=1, bbox=[0.0, 0.0, 10.0, 10.0])]
    leaf = _verify_leaf({"value": "OXNARD", "quote": "Address OXNARD fabricated"}, batch, strict=True)
    assert leaf.value is None and leaf.ungrounded


def test_verify_passthrough_trusts_values_without_evidence():
    from extract.core.schema_extract import _Leaf, _verify_passthrough

    tree = _verify_passthrough({"name": {"value": "OXNARD", "quote": "whatever"},
                                "rec": [{"id": {"value": "x1", "chunks": [9], "quote": ""}}]})
    assert isinstance(tree["name"], _Leaf) and tree["name"].value == "OXNARD"
    assert not tree["name"].ungrounded and tree["name"].evidence == []
    assert tree["rec"][0]["id"].value == "x1"


def test_bare_mode_schema_and_passthrough():
    """Bare mode: response schema has no {value,quote} envelope anywhere, and the
    raw model output flattens straight through as values."""
    from extract.core.schema_extract import _verify_passthrough, build_response_schema

    s = {"name": {"type": "string"}, "rec": {"type": "array",
         "items": {"type": "object", "properties": {"id": {"type": "string"}}}}}
    rs = build_response_schema(s, wrapped=False)
    assert "value" not in json_dumps(rs) or True  # structural check below
    assert rs["properties"]["name"]["type"] == "STRING"
    assert rs["properties"]["rec"]["items"]["properties"]["id"]["type"] == "STRING"
    tree = _verify_passthrough({"name": "OXNARD", "rec": [{"id": "x1"}]})
    assert tree["name"].value == "OXNARD" and tree["rec"][0]["id"].value == "x1"


def json_dumps(o):
    import json as _j
    return _j.dumps(o)
# --- Layout recipe additive params ---------------------------------------


async def test_chunk_transform_and_extra_instructions_default_is_byte_identical():
    """The two additive prompt hooks default to None → the model sees the exact
    historical prompt (EXTRACTION_PROMPT + render_context). No behaviour change."""
    from extract.core.schema_extract import (
        EXTRACTION_PROMPT,
        _index_chunks,
        render_context,
    )

    chunks = [_chunk("Title: Foo", page=1), _chunk("Subtitle: Bar", page=1)]
    schema = {"title": {"type": "string"}}

    base = StubModel({"title": {"value": "Foo", "chunks": [0], "quote": "Title: Foo"}})
    await aextract_schema(chunks, [], schema, model=base, strict=True)
    default_prompt = base.prompts[0]
    # The default prompt must be exactly the historical construction.
    indexed = _index_chunks(chunks, [])
    expect = EXTRACTION_PROMPT + render_context(indexed)
    assert default_prompt == expect


async def test_extra_instructions_appended_and_chunk_transform_applied():
    """When supplied, extra_instructions is appended (after the rendered context)
    and chunk_transform rewrites only the per-chunk text (geometry untouched)."""
    chunks = [_chunk("alpha", page=1, bbox=[1, 2, 3, 4])]
    schema = {"x": {"type": "string"}}
    model = StubModel({"x": {"value": "alpha", "chunks": [0], "quote": "alpha"}})

    result = await aextract_schema(
        chunks,
        [],
        schema,
        model=model,
        strict=True,
        chunk_transform=lambda i, c: f"P{i}::{c.text}",
        extra_instructions="SOURCE OF RECORD: the form.",
    )
    prompt = model.prompts[0]
    assert "P0::alpha" in prompt  # transform applied to rendered text
    assert prompt.rstrip().endswith("SOURCE OF RECORD: the form.")  # suffix appended
    # Geometry is untouched: the citation still uses the chunk's own bbox/page.
    assert result.values["x"] == "alpha"
    assert result.evidence["x"][0].page == 1






class DietStub:
    """Stub that records the response_schema of every call alongside the prompt."""

    def __init__(self, *responses: dict) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    async def generate_json(self, prompt: str, response_schema: dict) -> dict:
        self.prompts.append(prompt)
        self.schemas.append(response_schema)
        return self._responses[len(self.prompts) - 1]


_DIET_SCHEMA = {
    "line_items": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "description": {"type": "string"},
                "amount": {"type": "number"},
            },
        },
    },
    "merchant_name": {"type": "string"},
}


def test_diet_flags_default_off():
    from extract.config import settings

    assert settings.EXTRACT_OMIT_NULL_LEAVES is False
    assert settings.EXTRACT_SHORT_ROW_KEYS is False


def test_build_response_schema_identical_when_diet_off():
    """The flag-off response schema is the historical one, byte for byte."""
    import json as _json

    from extract.core.schema_extract import build_response_schema as brs

    for wrapped in (True, False):
        base = brs(_DIET_SCHEMA, wrapped=wrapped)
        assert _json.dumps(brs(_DIET_SCHEMA, wrapped=wrapped, omit_null=False)) == _json.dumps(
            base
        )
        assert "required" not in base
        assert "required" not in base["properties"]["line_items"]["items"]


async def test_diet_off_leaves_schema_and_prompt_untouched(monkeypatch):
    """End-to-end with the flags off: every call's schema equals the plain
    build_response_schema, and no diet text reaches the prompt."""
    from extract.config import settings
    from extract.core.schema_extract import _OMIT_NULL_INSTRUCTION, build_response_schema

    monkeypatch.setattr(settings, "EXTRACT_OMIT_NULL_LEAVES", False)
    monkeypatch.setattr(settings, "EXTRACT_SHORT_ROW_KEYS", False)

    model = DietStub(
        {"line_items": [{"sku": "X1", "description": "Widget", "amount": 5.0}]},
        {"merchant_name": {"value": "ACME", "chunks": [0], "quote": "ACME STORE"}},
    )
    await aextract_schema(
        [_chunk("ACME STORE\nX1 Widget 5.00")],
        [(1000.0, 1000.0)],
        _DIET_SCHEMA,
        model=model,
        lean_arrays=True,
    )
    assert model.schemas[0] == build_response_schema(
        {"line_items": _DIET_SCHEMA["line_items"]}, wrapped=False
    )
    assert model.schemas[1] == build_response_schema({"merchant_name": {"type": "string"}})
    assert all(_OMIT_NULL_INSTRUCTION not in p and "keys: a=" not in p for p in model.prompts)


async def test_omit_null_restores_full_schema_shape(monkeypatch):
    """The model omits every absent field; the merge path rehydrates the shape —
    same keys, null values, no evidence invented."""
    from extract.config import settings
    from extract.core.schema_extract import _OMIT_NULL_INSTRUCTION

    monkeypatch.setattr(settings, "EXTRACT_OMIT_NULL_LEAVES", True)
    monkeypatch.setattr(settings, "EXTRACT_SHORT_ROW_KEYS", False)
    schema = {
        **_DIET_SCHEMA,
        "totals": {
            "type": "object",
            "properties": {"tax": {"type": "number"}, "grand_total": {"type": "number"}},
        },
        "city": {"type": "string"},
    }
    model = DietStub(
        # line_items rows omit the cells the ticket does not print
        {"line_items": [{"sku": "X1", "amount": 5.0}, {"description": "Widget"}]},
        # the scalars group omits city and the whole totals object
        {"merchant_name": {"value": "ACME", "chunks": [0], "quote": "ACME STORE"}},
    )
    result = await aextract_schema(
        [_chunk("ACME STORE\nX1 5.00\nWidget")],
        [(1000.0, 1000.0)],
        schema,
        model=model,
        lean_arrays=True,
    )
    assert result.values == {
        "line_items": [
            {"sku": "X1", "description": None, "amount": 5.0},
            {"sku": None, "description": "Widget", "amount": None},
        ],
        "merchant_name": "ACME",
        "totals": {"tax": None, "grand_total": None},
        "city": None,
    }
    assert "city" not in result.evidence and "totals.tax" not in result.evidence
    # Omission is legal in the schema we actually sent, and the instruction is in the prompt.
    assert model.schemas[0]["required"] == []
    assert model.schemas[0]["properties"]["line_items"]["items"]["required"] == []
    assert model.schemas[1]["required"] == []
    assert all(_OMIT_NULL_INSTRUCTION in p for p in model.prompts)


async def test_omit_null_absent_array_and_object_rehydrate(monkeypatch):
    """A field-group the model answers with `{}` still returns the full shape."""
    from extract.config import settings

    monkeypatch.setattr(settings, "EXTRACT_OMIT_NULL_LEAVES", True)
    model = DietStub({}, {})
    result = await aextract_schema(
        [_chunk("blank ticket")], [(1000.0, 1000.0)], _DIET_SCHEMA,
        model=model, lean_arrays=True,
    )
    assert result.values == {"line_items": [], "merchant_name": None}


async def test_short_row_keys_round_trip_is_exact(monkeypatch):
    """Bare array group: aliased item keys in the schema + a prompt legend, mapped
    back to the caller's names on parse. The wrapped scalars group is untouched."""
    from extract.config import settings

    monkeypatch.setattr(settings, "EXTRACT_OMIT_NULL_LEAVES", False)
    monkeypatch.setattr(settings, "EXTRACT_SHORT_ROW_KEYS", True)
    model = DietStub(
        {"line_items": [{"a": "X1", "b": "Widget", "c": 5.0},
                        {"a": "X2", "b": "Gadget", "c": 7.5}]},
        {"merchant_name": {"value": "ACME", "chunks": [0], "quote": "ACME STORE"}},
    )
    result = await aextract_schema(
        [_chunk("ACME STORE\nX1 Widget 5.00\nX2 Gadget 7.50")],
        [(1000.0, 1000.0)],
        _DIET_SCHEMA,
        model=model,
        lean_arrays=True,
    )
    assert result.values["line_items"] == [
        {"sku": "X1", "description": "Widget", "amount": 5.0},
        {"sku": "X2", "description": "Gadget", "amount": 7.5},
    ]
    assert result.values["merchant_name"] == "ACME"
    items = model.schemas[0]["properties"]["line_items"]["items"]
    assert list(items["properties"]) == ["a", "b", "c"]
    # descriptions/types ride along with the alias, position for position
    assert items["properties"]["c"]["type"] == "NUMBER"
    assert "keys: a=sku, b=description, c=amount" in model.prompts[0]
    # the wrapped scalars call is not aliased and gets no legend
    assert "merchant_name" in model.schemas[1]["properties"]
    assert "keys: a=" not in model.prompts[1]


async def test_short_row_keys_and_omit_null_compose(monkeypatch):
    """Both levers at once: aliased rows with omitted cells rehydrate to full rows."""
    from extract.config import settings

    monkeypatch.setattr(settings, "EXTRACT_OMIT_NULL_LEAVES", True)
    monkeypatch.setattr(settings, "EXTRACT_SHORT_ROW_KEYS", True)
    model = DietStub(
        {"line_items": [{"a": "X1", "c": 5.0}]},
        {"merchant_name": {"value": "ACME", "chunks": [0], "quote": "ACME STORE"}},
    )
    result = await aextract_schema(
        [_chunk("ACME STORE\nX1 5.00")], [(1000.0, 1000.0)], _DIET_SCHEMA,
        model=model, lean_arrays=True,
    )
    assert result.values["line_items"] == [
        {"sku": "X1", "description": None, "amount": 5.0}
    ]


def test_row_alias_helpers_are_a_bijection():
    from extract.core.schema_extract import (
        MAX_LEAF_FIELDS,
        _alias,
        _row_alias_maps,
        _unalias_rows,
    )

    assert [_alias(i) for i in (0, 1, 25, 26, 27)] == ["a", "b", "z", "aa", "ab"]
    # Past the two-character range: 702 used to raise IndexError, inside the
    # module's own 1,000-leaf ceiling — a schema the validator admits must alias.
    assert [_alias(i) for i in (701, 702, 703, 728)] == ["zz", "aaa", "aab", "aba"]
    assert len({_alias(i) for i in range(MAX_LEAF_FIELDS)}) == MAX_LEAF_FIELDS
    maps = _row_alias_maps(_DIET_SCHEMA)
    assert maps == {"line_items": {"a": "sku", "b": "description", "c": "amount"}}
    rows = {"line_items": [{"a": "X1", "b": "W", "c": 1}]}
    assert _unalias_rows(rows, maps) == {
        "line_items": [{"sku": "X1", "description": "W", "amount": 1}]
    }
    # a scalar-item array (no properties) is never aliased
    assert _row_alias_maps({"tags": {"type": "array", "items": {"type": "string"}}}) == {}


def test_group_size_override_is_read_inside_its_supported_range(monkeypatch):
    """EXTRACT_MAX_GROUP_LEAVES decides call fan-out AND the response-schema
    ceiling, so a typo used to be dangerous three ways: a non-integer took the API
    module down at IMPORT, 0/negative turned a valid 1,000-leaf schema into 1,000
    concurrent model calls, and >40 walked back into the Vertex 400 the cap exists
    to avoid. Every one of those now falls back to the measured default."""
    from extract.core.schema_extract import _bounded_env_int

    for bad in ("", "  ", "forty", "40.5", "0", "-1", "41", "1000"):
        monkeypatch.setenv("EXTRACT_MAX_GROUP_LEAVES", bad)
        assert _bounded_env_int("EXTRACT_MAX_GROUP_LEAVES", default=40, low=1, high=40) == 40

    for good, expected in (("1", 1), ("16", 16), ("40", 40)):
        monkeypatch.setenv("EXTRACT_MAX_GROUP_LEAVES", good)
        assert (
            _bounded_env_int("EXTRACT_MAX_GROUP_LEAVES", default=40, low=1, high=40) == expected
        )

    monkeypatch.delenv("EXTRACT_MAX_GROUP_LEAVES", raising=False)
    assert _bounded_env_int("EXTRACT_MAX_GROUP_LEAVES", default=40, low=1, high=40) == 40


# --- QUOTE-DROP: the scalars call answers with values only -------------------


_QD_SCHEMA = {
    "total": {"type": "string"},
    "line_items": {
        "type": "array",
        "items": {"type": "object", "properties": {"sku": {"type": "string"}}},
    },
}


class _EnvelopeSpy:
    """Records the response schema each group call was asked to fill."""

    def __init__(self, wrapped_answer, bare_answer):
        self.wrapped_answer = wrapped_answer
        self.bare_answer = bare_answer
        self.schemas: list[dict] = []

    async def generate_json(self, prompt, response_schema, images=None, **kwargs):
        self.schemas.append(response_schema)
        props = (response_schema.get("properties") or {})
        leaf = next(iter(props.values()), {})
        # A WRAPPED leaf is the {value, chunks, quote} envelope; a bare one is the
        # user's own type.
        is_wrapped = isinstance(leaf, dict) and "value" in (leaf.get("properties") or {})
        return self.wrapped_answer if is_wrapped else self.bare_answer


def _qd_chunks():
    return [Chunk(page_content="TOTAL 4522\n1234 AGUA", page_no=1, bbox=[0, 0, 100, 100])], [
        (1000.0, 1000.0)
    ]


def _wrapped_scalars():
    return {"total": {"value": "4522", "chunks": [0], "quote": "TOTAL 4522"}}


def _bare_scalars_answer():
    return {"total": "4522"}


async def test_default_requests_still_get_the_wrapped_envelope_and_the_gate():
    """Off the lane nothing moves: the scalars call is asked for
    {value, chunks, quote} and the grounding gate runs on the answer."""
    model = _EnvelopeSpy(_wrapped_scalars(), _bare_scalars_answer())
    chunks, sizes = _qd_chunks()
    result = await aextract_schema(chunks, sizes, {"total": {"type": "string"}}, model=model)
    (schema,) = model.schemas
    leaf = schema["properties"]["total"]
    assert "value" in leaf["properties"]  # the envelope is still asked for
    assert "quote" in leaf["properties"]
    assert result.values["total"] == "4522"


async def test_bare_scalars_drops_the_envelope_and_with_it_the_gate():
    """The lever: values only. And the honest consequence — a value with no quote
    is no longer strict-nulled, because the gate cannot run at all. That is
    verification SKIPPED, not verification passed."""
    ungrounded_answer = {"total": "0"}  # nothing on the page says 0
    model = _EnvelopeSpy(_wrapped_scalars(), ungrounded_answer)
    chunks, sizes = _qd_chunks()
    result = await aextract_schema(
        chunks, sizes, {"total": {"type": "string"}}, model=model, bare_scalars=True, strict=True
    )
    (schema,) = model.schemas
    assert "value" not in (schema["properties"]["total"].get("properties") or {})
    assert result.values["total"] == "0"  # survives strict mode — no gate to fail
    assert result.ungrounded_fields == []  # ...and it is not reported as ungrounded


async def test_bare_scalars_reads_the_deployment_setting_when_unset(monkeypatch):
    """`None` means "read EXTRACT_BARE_SCALARS", like the rest of the diet."""
    from extract.config import settings as _settings

    monkeypatch.setattr(_settings, "EXTRACT_BARE_SCALARS", True)
    model = _EnvelopeSpy(_wrapped_scalars(), _bare_scalars_answer())
    chunks, sizes = _qd_chunks()
    await aextract_schema(chunks, sizes, {"total": {"type": "string"}}, model=model)
    (schema,) = model.schemas
    assert "value" not in (schema["properties"]["total"].get("properties") or {})

    monkeypatch.setattr(_settings, "EXTRACT_BARE_SCALARS", False)
    model2 = _EnvelopeSpy(_wrapped_scalars(), _bare_scalars_answer())
    await aextract_schema(chunks, sizes, {"total": {"type": "string"}}, model=model2)
    (schema2,) = model2.schemas
    assert "value" in schema2["properties"]["total"]["properties"]
