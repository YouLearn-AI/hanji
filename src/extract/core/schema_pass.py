"""The production schema-extraction pass, defined exactly once.

Everything that decides HOW a schema gets filled lives here. A caller supplies
the inputs and gets a :class:`SchemaExtractResult`; it does not choose a
recipe, and it does not decide whether the cross-read applies.
"""

from __future__ import annotations

from typing import Any

from extract.core.schema_extract import SchemaExtractResult, SchemaModel


async def run_schema_pass(
    chunks: list[Any],
    page_sizes: list[tuple[float, float]],
    user_schema: dict[str, Any],
    *,
    model: SchemaModel,
    strict: bool = True,
    xref_images: dict[int, bytes] | None = None,
    xref_cache: Any = None,
    xref_second_reader: bool = True,
    xref_prefetched: list | None = None,
    page_images: dict[int, bytes] | None = None,
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
    """Fill ``user_schema`` the way production does.

    Picks the recipe (``EXTRACT_LAYOUT_RECIPE_ENABLED``, default ON) and threads
    the cross-engine read through whichever one runs. ``xref_images`` needs page
    pixels, so callers without the document bytes simply pass nothing and get the
    same pass without it.
    """
    from extract.core.layout_recipe import aextract_layout, should_use_layout_recipe
    from extract.core.schema_extract import aextract_schema

    if should_use_layout_recipe():
        return await aextract_layout(
            chunks,
            page_sizes,
            user_schema,
            model=model,
            strict=strict,
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
    return await aextract_schema(
        chunks,
        page_sizes,
        user_schema,
        model=model,
        strict=strict,
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
