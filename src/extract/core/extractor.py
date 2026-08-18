"""Top-level orchestrator — detects input kind, converts to PDF if needed,
delegates to the PDF pipeline.

The public ``ExtractRequest`` surface accepts only a ``url``. Library and
CLI callers can use ``aextract_from_path`` or ``aextract_from_bytes`` to
run the same pipeline against a local file or pre-loaded bytes without
going through the HTTP schema.
"""

from __future__ import annotations

import asyncio

import httpx

from extract.core.errors import UnsupportedInput
from extract.core.io import load_bytes
from extract.core.kind import detect_kind
from extract.core.models import ExtractRequest, ExtractResponse, InputKind
from extract.core.pdf import EXTRACTION_ENGINE_BASELINE, MAX_SIZE_BYTES, extract_pdf
from extract.observability.timing import StageTimer, maybe_span
from extract.storage.base import Storage

_OFFICE_KINDS = frozenset({InputKind.DOCX, InputKind.PPTX})


def _remember_office_rendition(
    response: ExtractResponse, kind: InputKind, pdf_bytes: bytes | None
) -> ExtractResponse:
    """Keep the converted PDF on office responses so the API can publish it.

    Native PDFs and images already *are* the page the bboxes refer to, so
    they do not stash bytes. Library/CLI callers leave ``pdf_rendition_url``
    unset; only the hosted publish step turns this into a URL.
    """
    if kind in _OFFICE_KINDS and pdf_bytes:
        response._rendition_pdf = pdf_bytes
    return response


def _engine_ocr_provider(extraction_engine: str, ocr_provider_name: str | None) -> str | None:
    """Resolve the OCR provider for a request.

    An explicitly passed provider always wins (the schema-parse routes pin
    their own). Otherwise a non-baseline extraction engine names its provider
    directly — per-customer routing: the DB engine value IS the provider
    registry name, validated upstream against ``VALID_EXTRACTION_ENGINES`` so
    unknown values never reach the registry. No per-customer route is live
    today (per-account provider splits were retired);
    the mechanism remains for the next one.
    """
    if ocr_provider_name:
        return ocr_provider_name
    if extraction_engine and extraction_engine != EXTRACTION_ENGINE_BASELINE:
        return extraction_engine
    return None


class Extractor:
    """Entry point for library, CLI, and API callers."""

    def __init__(
        self,
        *,
        storage: Storage | None = None,
        download_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._storage = storage
        self._download_client = download_client

    async def aextract(
        self,
        request: ExtractRequest,
        *,
        timer: StageTimer | None = None,
        extraction_engine: str = EXTRACTION_ENGINE_BASELINE,
        ocr_provider_name: str | None = None,
        phi_safe: bool = False,
        block_private: bool = True,
    ) -> ExtractResponse:
        kind = detect_kind(filename=request.url)
        # Fall back to magic-byte sniffing when the URL has no recognized
        # extension (e.g. arxiv.org/pdf/<id>). Downloads once here and
        # passes the bytes downstream so we don't fetch twice.
        downloaded: bytes | None = None
        if kind is None:
            with maybe_span(timer, "download_ms"):
                downloaded = await load_bytes(
                    url=request.url,
                    max_size=MAX_SIZE_BYTES,
                    client=self._download_client,
                    block_private=block_private,
                )
            kind = detect_kind(filename=request.url, data=downloaded)
            if kind is None:
                raise UnsupportedInput(
                    "Could not determine input kind from URL contents or extension."
                )

        if kind == InputKind.PDF:
            pdf_bytes: bytes | None = downloaded
        elif kind == InputKind.PPTX:
            from extract.core.converters.pptx import convert_to_pdf as pptx_convert

            with maybe_span(timer, "convert_ms"):
                pdf_bytes = await pptx_convert(
                    url=request.url if downloaded is None else None,
                    data=downloaded,
                    max_size=MAX_SIZE_BYTES,
                    block_private=block_private,
                )
        elif kind == InputKind.DOCX:
            from extract.core.converters.docx import convert_to_pdf as docx_convert

            with maybe_span(timer, "convert_ms"):
                pdf_bytes = await docx_convert(
                    url=request.url if downloaded is None else None,
                    data=downloaded,
                    max_size=MAX_SIZE_BYTES,
                    block_private=block_private,
                )
        elif kind == InputKind.IMAGE:
            from extract.core.converters.image import convert_to_pdf as image_convert

            with maybe_span(timer, "convert_ms"):
                pdf_bytes = await image_convert(
                    url=request.url if downloaded is None else None,
                    data=downloaded,
                    max_size=MAX_SIZE_BYTES,
                    timer=timer,
                    block_private=block_private,
                )
        else:  # pragma: no cover - exhaustive
            raise UnsupportedInput(f"Unsupported kind: {kind}")

        if timer is not None:
            timer.meta["doc_kind"] = kind.value

        response = await extract_pdf(
            request,
            data=pdf_bytes,
            storage=self._storage,
            download_client=self._download_client,
            timer=timer,
            extraction_engine=extraction_engine,
            **(
                {"ocr_provider_name": p}
                if (p := _engine_ocr_provider(extraction_engine, ocr_provider_name))
                else {}
            ),
            phi_safe=phi_safe,
            block_private=block_private,
        )
        return _remember_office_rendition(response, kind, pdf_bytes)

    async def aextract_from_path(
        self,
        path: str,
        *,
        request: ExtractRequest | None = None,
        extraction_engine: str = EXTRACTION_ENGINE_BASELINE,
        ocr_provider_name: str | None = None,
    ) -> ExtractResponse:
        """Library/CLI helper — extract from a local file.

        Not part of the HTTP surface. The request body's ``url`` (if any)
        is ignored; the path argument is authoritative.
        """
        request = request or ExtractRequest()
        kind = detect_kind(filename=path)
        if kind is None:
            raise UnsupportedInput("Could not determine input kind from file extension.")

        if kind == InputKind.PDF:
            pdf_bytes: bytes | None = None
        elif kind == InputKind.PPTX:
            from extract.core.converters.pptx import convert_to_pdf as pptx_convert

            pdf_bytes = await pptx_convert(path=path, max_size=MAX_SIZE_BYTES)
        elif kind == InputKind.DOCX:
            from extract.core.converters.docx import convert_to_pdf as docx_convert

            pdf_bytes = await docx_convert(path=path, max_size=MAX_SIZE_BYTES)
        elif kind == InputKind.IMAGE:
            from extract.core.converters.image import convert_to_pdf as image_convert

            pdf_bytes = await image_convert(path=path, max_size=MAX_SIZE_BYTES)
        else:  # pragma: no cover
            raise UnsupportedInput(f"Unsupported kind: {kind}")

        if pdf_bytes is None:
            from pathlib import Path as _Path

            pdf_bytes = _Path(path).read_bytes()

        response = await extract_pdf(
            request,
            data=pdf_bytes,
            storage=self._storage,
            download_client=self._download_client,
            extraction_engine=extraction_engine,
            **(
                {"ocr_provider_name": p}
                if (p := _engine_ocr_provider(extraction_engine, ocr_provider_name))
                else {}
            ),
        )
        return _remember_office_rendition(response, kind, pdf_bytes)

    async def aextract_from_bytes(
        self,
        data: bytes,
        *,
        filename: str | None = None,
        request: ExtractRequest | None = None,
        timer: StageTimer | None = None,
        extraction_engine: str = EXTRACTION_ENGINE_BASELINE,
        ocr_provider_name: str | None = None,
        phi_safe: bool = False,
    ) -> ExtractResponse:
        """Library / API entry point for pre-loaded document bytes.

        ``filename`` is advisory — used only when magic bytes are ambiguous
        (a malformed archive, for example). When both disagree, magic wins.
        """
        request = request or ExtractRequest()
        kind = detect_kind(filename=filename, data=data)
        if kind is None:
            raise UnsupportedInput("Could not determine input kind from file contents or filename.")

        if kind == InputKind.PDF:
            pdf_bytes = data
        elif kind == InputKind.PPTX:
            from extract.core.converters.pptx import convert_to_pdf as pptx_convert

            with maybe_span(timer, "convert_ms"):
                pdf_bytes = await pptx_convert(data=data, max_size=MAX_SIZE_BYTES)
        elif kind == InputKind.DOCX:
            from extract.core.converters.docx import convert_to_pdf as docx_convert

            with maybe_span(timer, "convert_ms"):
                pdf_bytes = await docx_convert(data=data, max_size=MAX_SIZE_BYTES)
        elif kind == InputKind.IMAGE:
            from extract.core.converters.image import convert_to_pdf as image_convert

            with maybe_span(timer, "convert_ms"):
                pdf_bytes = await image_convert(data=data, max_size=MAX_SIZE_BYTES, timer=timer)
        else:  # pragma: no cover - exhaustive
            raise UnsupportedInput(f"Unsupported kind: {kind}")

        if timer is not None:
            timer.meta["doc_kind"] = kind.value

        response = await extract_pdf(
            request,
            data=pdf_bytes,
            storage=self._storage,
            download_client=self._download_client,
            timer=timer,
            extraction_engine=extraction_engine,
            **(
                {"ocr_provider_name": p}
                if (p := _engine_ocr_provider(extraction_engine, ocr_provider_name))
                else {}
            ),
            phi_safe=phi_safe,
        )
        return _remember_office_rendition(response, kind, pdf_bytes)

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        """Synchronous wrapper that manages its own event loop."""
        return asyncio.run(self.aextract(request))
