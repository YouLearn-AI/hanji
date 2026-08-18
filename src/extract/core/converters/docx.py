"""DOCX / DOC → PDF via LibreOffice headless."""

from __future__ import annotations

from extract.core.converters._office import convert_office_to_pdf


async def convert_to_pdf(
    *,
    url: str | None = None,
    path: str | None = None,
    data: bytes | None = None,
    max_size: int | None = None,
    block_private: bool = True,
) -> bytes:
    return await convert_office_to_pdf(
        url=url,
        path=path,
        data=data,
        max_size=max_size,
        input_ext="docx",
        block_private=block_private,
    )
