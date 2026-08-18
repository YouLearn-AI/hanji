"""Public API: parsing and schema extraction.

Four routes that funnel into two shared runners:

- ``POST /v1/parse``          — JSON body with a ``url``. Download, convert, parse.
- ``POST /v1/parse/file``     — ``multipart/form-data`` upload of the document.
- ``POST /v1/extract/schema`` — fill a user JSON schema from a document URL,
  with verified citations.
- ``POST /v1/extract/schema/file`` — the multipart twin.

There is no auth or billing here. If you expose this server publicly, front it
with your own gateway.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from typing import Literal

import orjson
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import ValidationError

from extract.config import settings
from extract.core import (
    DocumentTooLarge,
    ExtractionFailed,
    Extractor,
    ExtractRequest,
    ExtractResponse,
    PageLimitExceeded,
    UnsupportedInput,
    UpstreamTimeout,
)
from extract.core.chunking import render_document_content
from extract.core.layout_recipe import should_use_layout_recipe
from extract.core.models import SchemaExtractRequest, SchemaExtractResponse
from extract.core.ocr.reread import reread_evidence
from extract.core.pdf import MAX_SIZE_BYTES
from extract.core.schema_extract import (
    SchemaModel,
    SchemaValidationError,
    agenerate_schema,
    flatten_schema,
    validate_schema,
)
from extract.core.schema_pass import run_schema_pass
from extract.core.textract_page import TextractPageCache
from extract.core.vision_tighten import tighten_evidence
from extract.gemini_extract import SchemaModelError
from extract.genai_client import genai_configured
from extract.observability.timing import StageTimer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])

# 1 MiB chunks — big enough to amortize per-chunk overhead, small enough
# that we reject a too-large upload within a MiB of crossing the cap.
MAX_UPLOAD_CHUNK = 1 * 1024 * 1024

#: Schema extraction parses with the same production parse model.
SCHEMA_PARSE_OCR_PROVIDER = "qwen_lora"

#: Non-standard but widely understood: the caller closed the connection before
#: the response was ready.
CLIENT_CLOSED_REQUEST = 499

#: The resolution MuPDF assumes for an image it opens as a document when the
#: file stores none.
_MUPDF_IMAGE_DPI = 96.0


def _extractor(request: Request) -> Extractor:
    return request.app.state.extractor


def _schema_model(request: Request) -> SchemaModel:
    if not genai_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Schema extraction is not configured. Set GEMINI_API_KEY or "
                "GOOGLE_VERTEX_PROJECT."
            ),
        )
    return request.app.state.schema_model


_extractor_dependency = Depends(_extractor)
_schema_model_dependency = Depends(_schema_model)


# --------------------------------------------------------------------------- #
# Request bounds: stop working when nobody is waiting.
# --------------------------------------------------------------------------- #


async def _client_hung_up(http_request: Request) -> None:
    """Resolve as soon as the caller closes the connection.

    BLOCKS on ``receive()`` rather than polling ``Request.is_disconnected()``
    (which only sees a disconnect already queued at that instant). Blocking is
    safe because this watch starts only after the route has read the request
    body, so ``http.disconnect`` is the sole message left to come.
    """
    try:
        while True:
            message = await http_request.receive()
            if message.get("type") == "http.disconnect":
                return
    except Exception:  # noqa: BLE001 — a broken watch must not fail the request
        logger.warning(
            "client-disconnect watch failed; the request deadline still applies",
            exc_info=True,
        )
    # Never resolve on failure: resolving would cancel a request whose client
    # is very much still waiting. EXTRACT_REQUEST_TIMEOUT_SECONDS still bounds it.
    await asyncio.Event().wait()


async def _extract_bounded(extract, *, http_request: Request | None):
    """Run the extraction, but stop when nobody is waiting for it any more.

    Two bounds: the caller hung up (nothing in the ASGI stack cancels a handler
    on client disconnect, so without this the server runs the whole job for a
    response no one receives), and ``EXTRACT_REQUEST_TIMEOUT_SECONDS`` — a
    backstop against an upstream that never answers, set well above the slowest
    legitimate document. Zero disables it.
    """
    timeout = float(settings.EXTRACT_REQUEST_TIMEOUT_SECONDS or 0)
    task = asyncio.ensure_future(extract())
    waiters: set[asyncio.Future] = {task}
    watcher: asyncio.Future | None = None
    if http_request is not None:
        watcher = asyncio.ensure_future(_client_hung_up(http_request))
        waiters.add(watcher)
    try:
        done, _ = await asyncio.wait(
            waiters,
            timeout=timeout or None,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        if watcher is not None and not watcher.done():
            watcher.cancel()
    if task in done:
        return task.result()
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    if watcher is not None and watcher in done:
        raise HTTPException(
            status_code=CLIENT_CLOSED_REQUEST,
            detail="Client closed the request before it completed.",
        )
    raise HTTPException(
        status_code=504,
        detail=f"Extraction exceeded the {timeout:.0f}s server deadline and was stopped.",
    )


async def _stop_when_nobody_is_waiting(awaitable, *, hangup, deadline):
    """Await one stage of a schema request, but stop if nobody is waiting.

    Applied per stage rather than around the whole runner so that raising from
    inside the runner's ``try`` runs its ``finally`` — prefetch abort set,
    background tasks cancelled. ``deadline`` is a monotonic instant shared by
    every stage, so the backstop bounds the REQUEST rather than resetting per
    await.
    """
    task = asyncio.ensure_future(awaitable)
    waiters: set[asyncio.Future] = {task}
    if hangup is not None:
        waiters.add(hangup)
    timeout = max(0.0, deadline - time.monotonic()) if deadline else None
    done, _ = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        return task.result()
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    if hangup is not None and hangup in done:
        raise HTTPException(
            status_code=CLIENT_CLOSED_REQUEST,
            detail="Client closed the request before it completed.",
        )
    raise HTTPException(
        status_code=504,
        detail="Extraction exceeded the server deadline and was stopped.",
    )


def _aborted(abort: threading.Event | None) -> bool:
    """Has the request that asked for this background work already finished?"""
    return abort is not None and abort.is_set()


# --------------------------------------------------------------------------- #
# Cross-read prefetch (schema path): render pages + warm the Textract cache
# concurrently with parse.
# --------------------------------------------------------------------------- #


def _prefetch_xref(
    doc_bytes: bytes,
    cache,
    *,
    want_chunks: bool = False,
    abort: threading.Event | None = None,
) -> tuple[dict[int, bytes], list | None]:
    """Render every page and warm the Textract cache, in a worker thread
    alongside parse.

    Because nothing here depends on parse output, it overlaps parse COMPLETELY,
    so the Textract round-trip costs no wall clock. The box tightener later
    reads this same cache, so a page that is both cross-read and cited is never
    sent to Textract twice. ``want_chunks`` extends the overlap to the whole
    cross-read (the layout recipe makes its own xref call instead, so its
    callers pass False). ``abort`` is cooperative cancellation: a worker thread
    cannot be cancelled, so the stages below check it before starting the next
    piece of paid work. Fail-open throughout: a failure degrades to the serial
    path, never an error.
    """
    try:
        import io as _io

        from PIL import Image

        if _aborted(abort):
            return {}, None
        pages = _page_images(doc_bytes, scale=settings.EXTRACT_XREF_RENDER_SCALE, abort=abort)
        if cache is not None:
            for img in pages.values():
                if _aborted(abort):
                    return pages, None
                with Image.open(_io.BytesIO(img)) as im:
                    cache.read(img, im.size)
    except Exception:  # noqa: BLE001 — a secondary source must never break extraction
        return {}, None
    if _aborted(abort):
        return pages, None
    xchunks = None
    if want_chunks and pages:
        try:
            from extract.core.xref_context import xref_chunks as _xref_fn

            xchunks = _xref_fn(
                pages,
                cache=cache,
                should_stop=(lambda: _aborted(abort)),
            )
        except Exception:  # noqa: BLE001 — fall back to the in-extract serial path
            xchunks = None
    return pages, xchunks


def _page_images(
    doc_bytes: bytes,
    *,
    scale: float = 2.0,
    abort: threading.Event | None = None,
) -> dict[int, bytes]:
    """Render every page to PNG for the cross-engine secondary read.

    Fail-open: an unreadable document yields ``{}`` and the xref pass simply
    does not run. MuPDF also decodes a bare JPEG/PNG/TIFF/BMP upload as a
    one-page document, so image uploads get their pixels here too.
    """
    return _render_pdf_pages(doc_bytes, scale, abort=abort)


def _render_pdf_pages(
    doc_bytes: bytes, scale: float, *, abort: threading.Event | None = None
) -> dict[int, bytes]:
    """Every page of a PDF (or of anything MuPDF opens) as a PNG, or ``{}``.

    ``abort`` is checked BETWEEN PAGES: rendering runs in a worker thread that
    cancelling the surrounding task does not stop. Returns the pages it has.
    """
    out: dict[int, bytes] = {}
    try:
        import io

        import pymupdf
        from PIL import Image

        pymupdf.TOOLS.mupdf_display_errors(False)
        doc = pymupdf.open(stream=doc_bytes, filetype="pdf")
        try:
            for i in range(doc.page_count):
                if _aborted(abort):
                    return out
                pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                buf = io.BytesIO()
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(buf, "PNG")
                out[i + 1] = buf.getvalue()
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 — a secondary source must never break extraction
        return {}
    return out


def _first_pages(images: dict[int, bytes] | None, limit: int) -> dict[int, bytes] | None:
    """The first ``limit`` rendered pages, in page order, or ``None``."""
    if not images or limit <= 0:
        return None
    return {p: images[p] for p in sorted(images)[:limit]}


# --------------------------------------------------------------------------- #
# Upload handling + error mapping.
# --------------------------------------------------------------------------- #


async def _check_upload_content_length(request: Request) -> None:
    """Route-level fast-reject based on the ``Content-Length`` header.

    Runs *before* FastAPI body parsing so a client that honestly declares a
    multi-GB body is rejected without reading it. The streaming check in
    ``_read_upload`` is authoritative for clients that lie or use chunked
    transfer encoding.
    """
    declared = request.headers.get("content-length")
    if declared is None:
        return
    try:
        declared_int = int(declared)
    except ValueError:
        return
    if declared_int > MAX_SIZE_BYTES + MAX_UPLOAD_CHUNK:
        raise HTTPException(
            status_code=413,
            detail=f"Upload declares {declared_int} bytes, exceeds max of {MAX_SIZE_BYTES}.",
        )


async def _read_upload(file: UploadFile, max_size: int) -> bytes:
    """Stream a multipart upload into memory with a hard size cap."""
    buf = bytearray()
    while True:
        chunk = await file.read(MAX_UPLOAD_CHUNK)
        if not chunk:
            break
        if len(buf) + len(chunk) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeded max size of {max_size} bytes.",
            )
        buf.extend(chunk)
    return bytes(buf)


def _map_extraction_error(exc: Exception) -> HTTPException:
    """Translate core exceptions into the same HTTP codes on every route."""
    if isinstance(exc, (DocumentTooLarge, PageLimitExceeded)):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, (UnsupportedInput, ExtractionFailed)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, UpstreamTimeout):
        # The model provider stalled past the request's budget. 504, not 400:
        # the caller's document and schema were fine, and the same request is
        # worth retrying.
        return HTTPException(status_code=504, detail=str(exc))
    raise exc  # pragma: no cover — unexpected; let FastAPI 500 it.


def _parse_user_schema(raw: str) -> dict:
    """Decode the user schema from a multipart form field (→ 422)."""
    try:
        schema = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid schema JSON: {e}") from e
    if not isinstance(schema, dict):
        raise HTTPException(status_code=422, detail="Schema must be a JSON object of fields.")
    return schema


def _validate_schema_or_422(user_schema: dict) -> None:
    """Flatten (Pydantic/zod $ref/anyOf/allOf) then run the schema guards
    (depth/cycle/breadth/type) → 422 on violation."""
    try:
        validate_schema(flatten_schema(user_schema))
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _validate_request_schema(user_schema: dict | None, auto_schema: bool) -> None:
    """Validate the schema/auto_schema combination at the route boundary.

    A supplied schema is always validated up front (→ 422), and wins —
    ``auto_schema`` designs one only when none is given. A request with neither
    is a 422.
    """
    if user_schema is not None:
        _validate_schema_or_422(user_schema)
    elif not auto_schema:
        raise HTTPException(
            status_code=422,
            detail="A 'schema' is required unless auto_schema=true.",
        )


def _build_extract_request(**fields) -> ExtractRequest:
    """Construct an ExtractRequest from multipart form fields, mapping model
    validation to the same 422 the JSON routes get from body validation."""
    try:
        return ExtractRequest(**fields)
    except ValidationError as e:
        errors = "; ".join(err.get("msg", "invalid value") for err in e.errors())
        raise HTTPException(status_code=422, detail=errors) from e


# --------------------------------------------------------------------------- #
# Multipart form fields (module-level singletons so FastAPI defaults avoid the
# B008 lint; descriptions mirror the JSON request models).
# --------------------------------------------------------------------------- #

_upload_file_field = File(
    ...,
    description=("PDF, PPTX, DOCX, or image (PNG, JPEG, WebP, TIFF, HEIC/HEIF, BMP) document."),
)
_table_output_format_form = Form(
    "markdown",
    description=(
        "How table chunks are serialized in page_content: 'markdown' (default) "
        "GitHub-flavored markdown, or 'html' an HTML table (row 0 emitted as "
        "the header row exactly as it is)."
    ),
)
_extract_text_form = Form(
    True,
    description=(
        "Include text chunks in the response. Set false to skip text spans; "
        "table chunks (and figures, when extract_images is true) are still "
        "returned."
    ),
)
_extract_images_form = Form(
    True,
    description=(
        "Include figure (image) chunks in the response. Set false to skip "
        "figure extraction; text and table chunks are still returned."
    ),
)
_chunking_form = Form(
    "none",
    description=(
        "'none' (default): response unchanged. 'semantic': additionally "
        "return `segments` — elements grouped toward chunk_size characters "
        "at semantic/structural boundaries (headings, gaps, page breaks), "
        "each segment carrying embed-ready markdown content plus member "
        "records with page numbers and bounding boxes."
    ),
)
_chunk_size_form = Form(
    1000,
    description=(
        "Target segment size in characters of segment content. Segments land "
        "in a +/-25% band around this target (default 750-1250). Only "
        "meaningful when chunking is enabled."
    ),
)
_include_content_form = Form(
    False,
    description=(
        "false (default): response unchanged. true: additionally return "
        "`content` — the entire parsed document as a single text string, "
        "concatenated in reading order."
    ),
)
_strict_form = Form(
    True,
    description=(
        "What happens to a value whose citation cannot be verified against "
        "the document. true (default): the value is nulled out and its path "
        "listed in ungrounded_fields, so a fabricated value never reaches "
        "you. false: the value is kept but still flagged in "
        "ungrounded_fields."
    ),
)
_auto_schema_form = Form(
    False,
    description=(
        "Set true (and omit schema) to have a schema designed from the "
        "document first, then filled with the same grounded extraction. The "
        "schema used is returned in generated_schema."
    ),
)
_schema_extract_images_form = Form(
    True,
    description=(
        "Include figures from the parse stage in the extraction context. Set "
        "false to extract from text and tables only."
    ),
)
_include_ocr_text_form = Form(
    False,
    description=(
        "false (default): response unchanged. true: additionally return "
        "`ocr_text` — the whole parsed document as a single text string in "
        "reading order, the same text POST /v1/parse returns as `content`."
    ),
)
_schema_form_field_optional = Form(
    None,
    description="The JSON schema to fill, as a JSON object string.",
)


# --------------------------------------------------------------------------- #
# Routes.
# --------------------------------------------------------------------------- #


@router.post(
    "/extract",
    response_model=ExtractResponse,
    include_in_schema=False,
)
@router.post(
    "/parse",
    response_model=ExtractResponse,
    operation_id="extract_v1",
    summary="Parse a document by URL",
    description=(
        "Download the document at `url`, parse it, and return text, table, and image "
        "chunks in reading order. Every chunk carries its page number and bounding box."
    ),
)
async def extract_endpoint(
    http_request: Request,
    body: ExtractRequest,
    extractor: Extractor = _extractor_dependency,
) -> Response:
    if not body.url:
        raise HTTPException(status_code=422, detail="A 'url' is required.")
    timer = StageTimer()
    return await _run_extraction(
        timer=timer,
        route=http_request.url.path,
        http_request=http_request,
        # Every caller gets the SSRF guard (resolve + block private/loopback/
        # link-local/metadata addresses, per-redirect revalidation).
        extract=lambda: extractor.aextract(body, timer=timer, block_private=True),
    )


@router.post(
    "/extract/file",
    response_model=ExtractResponse,
    include_in_schema=False,
    dependencies=[Depends(_check_upload_content_length)],
)
@router.post(
    "/parse/file",
    response_model=ExtractResponse,
    operation_id="extract_file_v1",
    dependencies=[Depends(_check_upload_content_length)],
    summary="Parse an uploaded document",
)
async def extract_file_endpoint(
    http_request: Request,
    file: UploadFile = _upload_file_field,
    extract_text: bool = _extract_text_form,
    extract_images: bool = _extract_images_form,
    table_output_format: Literal["markdown", "html"] = _table_output_format_form,
    chunking: Literal["none", "semantic"] = _chunking_form,
    chunk_size: int = _chunk_size_form,
    include_content: bool = _include_content_form,
    extractor: Extractor = _extractor_dependency,
) -> Response:
    """Accept a document as ``multipart/form-data`` and return chunks.

    Mirrors ``POST /v1/parse`` in every way except how the document arrives.
    The filename is advisory; magic bytes drive kind detection.
    """
    timer = StageTimer()
    with timer.span("upload_read_ms"):
        data = await _read_upload(file, MAX_SIZE_BYTES)
    request = _build_extract_request(
        extract_text=extract_text,
        extract_images=extract_images,
        table_output_format=table_output_format,
        chunking=chunking,
        chunk_size=chunk_size,
        include_content=include_content,
    )
    return await _run_extraction(
        timer=timer,
        route=http_request.url.path,
        http_request=http_request,
        extract=lambda: extractor.aextract_from_bytes(
            data,
            filename=file.filename,
            request=request,
            timer=timer,
        ),
    )


@router.post(
    "/extract/schema",
    response_model=SchemaExtractResponse,
    operation_id="extract_schema_v1",
    summary="Fill a schema from a document by URL",
)
async def extract_schema_endpoint(
    http_request: Request,
    body: SchemaExtractRequest,
    extractor: Extractor = _extractor_dependency,
    model: SchemaModel = _schema_model_dependency,
) -> Response:
    """Fill an arbitrary user JSON schema from a document URL, with citations."""
    _validate_request_schema(body.schema_, body.auto_schema)  # fail fast (422)
    if not body.url:
        raise HTTPException(status_code=422, detail="A 'url' is required.")
    timer = StageTimer()
    request = ExtractRequest(url=body.url, extract_images=body.extract_images)
    return await _run_schema_extraction(
        timer=timer,
        route="/v1/extract/schema",
        http_request=http_request,
        include_ocr_text=body.include_ocr_text,
        user_schema=body.schema_,
        strict=body.strict,
        model=model,
        auto_schema=body.auto_schema,
        extract=lambda: extractor.aextract(
            request,
            timer=timer,
            ocr_provider_name=SCHEMA_PARSE_OCR_PROVIDER,
            block_private=True,
        ),
    )


@router.post(
    "/extract/schema/file",
    response_model=SchemaExtractResponse,
    operation_id="extract_schema_file_v1",
    dependencies=[Depends(_check_upload_content_length)],
    summary="Fill a schema from an uploaded document",
)
async def extract_schema_file_endpoint(
    http_request: Request,
    file: UploadFile = _upload_file_field,
    schema_raw: str | None = _schema_form_field_optional,
    strict: bool = _strict_form,
    extract_images: bool = _schema_extract_images_form,
    auto_schema: bool = _auto_schema_form,
    include_ocr_text: bool = _include_ocr_text_form,
    extractor: Extractor = _extractor_dependency,
    model: SchemaModel = _schema_model_dependency,
) -> Response:
    """Schema extraction from a ``multipart/form-data`` upload."""
    user_schema = _parse_user_schema(schema_raw) if (schema_raw and not auto_schema) else None
    _validate_request_schema(user_schema, auto_schema)  # fail fast (422)
    timer = StageTimer()
    with timer.span("upload_read_ms"):
        data = await _read_upload(file, MAX_SIZE_BYTES)
    request = ExtractRequest(extract_images=extract_images)
    return await _run_schema_extraction(
        timer=timer,
        route="/v1/extract/schema/file",
        http_request=http_request,
        include_ocr_text=include_ocr_text,
        user_schema=user_schema,
        strict=strict,
        model=model,
        auto_schema=auto_schema,
        doc_bytes=data,
        extract=lambda: extractor.aextract_from_bytes(
            data,
            filename=file.filename,
            request=request,
            timer=timer,
            ocr_provider_name=SCHEMA_PARSE_OCR_PROVIDER,
        ),
    )


# --------------------------------------------------------------------------- #
# Shared runners.
# --------------------------------------------------------------------------- #


async def _run_extraction(
    *,
    timer: StageTimer,
    extract,
    route: str,
    http_request: Request | None = None,
) -> Response:
    """Shared parse path: extract → serialize → log.

    Both parse routes converge here so timing instrumentation and error
    mapping live in exactly one place. ``http_request`` lets the extraction be
    cancelled when the caller hangs up (``_extract_bounded``).
    """
    t0 = time.perf_counter()
    try:
        response = await _extract_bounded(extract, http_request=http_request)
        with timer.span("serialize_ms"):
            body_bytes = orjson.dumps(response.model_dump(mode="json"))
        return Response(content=body_bytes, media_type="application/json")
    except (
        DocumentTooLarge,
        PageLimitExceeded,
        UnsupportedInput,
        ExtractionFailed,
        UpstreamTimeout,
    ) as e:
        raise _map_extraction_error(e) from e
    finally:
        timer.record_ms("total_ms", (time.perf_counter() - t0) * 1000.0)
        logger.info("extract %s timings=%s", route, timer.timings_ms)


async def _run_schema_extraction(
    *,
    timer: StageTimer,
    extract,
    route: str,
    user_schema: dict | None,
    strict: bool,
    model: SchemaModel,
    auto_schema: bool = False,
    doc_bytes: bytes | None = None,
    include_ocr_text: bool = False,
    http_request: Request | None = None,
) -> Response:
    """Shared schema-extract path: parse → ground-extract → refine → serialize.

    The cross-read prefetch depends only on the uploaded bytes, never on parse
    output, so it runs CONCURRENTLY with parse: the Textract round-trip
    (~1-3s/page) is hidden behind a parse that takes longer anyway.
    """
    t0 = time.perf_counter()
    _xref_task: asyncio.Task | None = None
    _xref_abort = threading.Event()
    _parse_task: asyncio.Task | None = None
    _hangup: asyncio.Future | None = (
        asyncio.ensure_future(_client_hung_up(http_request)) if http_request is not None else None
    )
    _timeout = float(settings.EXTRACT_REQUEST_TIMEOUT_SECONDS or 0)
    _deadline = (time.monotonic() + _timeout) if _timeout else 0.0
    try:
        _parse_task = asyncio.create_task(extract())
        _use_layout = should_use_layout_recipe()
        # One Textract page cache for the whole request: the cross-read warms it
        # and the box tightener reads it later, so a page that is both
        # cross-read and cited costs one Textract call, not two.
        _tx_cache = TextractPageCache()
        # want_chunks only on the plain-extractor branch: the layout recipe
        # makes its own xref call inside aextract_layout, and prefetching here
        # too would run the recognizer read twice per request.
        _xref_task = (
            asyncio.create_task(
                asyncio.to_thread(
                    _prefetch_xref,
                    doc_bytes,
                    _tx_cache,
                    want_chunks=not _use_layout,
                    abort=_xref_abort,
                )
            )
            if doc_bytes
            else None
        )
        parsed = await _stop_when_nobody_is_waiting(
            _parse_task, hangup=_hangup, deadline=_deadline
        )
        timer.meta["page_count"] = parsed.page_count
        # Design a schema only when the caller supplied none; a provided schema wins.
        generated = auto_schema and user_schema is None
        if generated:
            with timer.span("schema_generate_ms"):
                user_schema = await _stop_when_nobody_is_waiting(
                    agenerate_schema(parsed.chunks, parsed.page_sizes, model),
                    hangup=_hangup,
                    deadline=_deadline,
                )
        with timer.span("schema_extract_ms"):
            _xref_imgs, _xref_ready = (
                await _stop_when_nobody_is_waiting(_xref_task, hangup=_hangup, deadline=_deadline)
                if _xref_task is not None
                else (None, None)
            )
            result = await _stop_when_nobody_is_waiting(
                run_schema_pass(
                    parsed.chunks,
                    parsed.page_sizes,
                    user_schema,
                    model=model,
                    strict=strict,
                    xref_images=_xref_imgs,
                    xref_cache=_tx_cache,
                    xref_prefetched=_xref_ready,
                ),
                hangup=_hangup,
                deadline=_deadline,
            )
        # Two independent, network-bound post-extract refinements over the SAME
        # evidence, run CONCURRENTLY (both only on the file path, which
        # supplies doc_bytes). Both are value-preserving and fail open:
        #   - Cloud Vision box-tightening: shrinks each coarse citation box to
        #     its value span.
        #   - Low-confidence re-read: skips instantly unless a high-stakes
        #     field is below the confidence gate, then re-reads only those
        #     crops and flags disagreements with a suggested value.
        tighten_on = settings.EXTRACT_VISION_TIGHTEN_ENABLED and bool(doc_bytes)
        reread_on = settings.SCHEMA_REREAD_ENABLED and bool(doc_bytes)
        if tighten_on or reread_on:
            with timer.span("post_extract_ms"):
                await _stop_when_nobody_is_waiting(
                    asyncio.gather(
                        asyncio.to_thread(
                            tighten_evidence, result.evidence, result.values, doc_bytes
                        )
                        if tighten_on
                        else asyncio.sleep(0, result=None),
                        asyncio.to_thread(
                            reread_evidence,
                            result.evidence,
                            result.values,
                            doc_bytes,
                            gate=settings.SCHEMA_REREAD_GATE,
                        )
                        if reread_on
                        else asyncio.sleep(0, result=None),
                    ),
                    hangup=_hangup,
                    deadline=_deadline,
                )
        # Whole-document text: the parse and the fields from one call. The text
        # is the parse's own render (identical to /v1/parse's `content`).
        _ocr_text = None
        if settings.EXTRACT_INCLUDE_OCR_TEXT and include_ocr_text:
            with timer.span("ocr_text_ms"):
                _ocr_text = render_document_content(parsed.chunks)
        response_model = SchemaExtractResponse(
            values=result.values,
            evidence=result.evidence,
            ungrounded_fields=result.ungrounded_fields,
            page_count=parsed.page_count,
            generated_schema=user_schema if generated else None,
            ocr_text=_ocr_text,
        )
        with timer.span("serialize_ms"):
            body_bytes = orjson.dumps(response_model.dump_public())
        return Response(content=body_bytes, media_type="application/json")
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except SchemaModelError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except (
        DocumentTooLarge,
        PageLimitExceeded,
        UnsupportedInput,
        ExtractionFailed,
        UpstreamTimeout,
    ) as e:
        raise _map_extraction_error(e) from e
    finally:
        # The prefetch outlives a failed request otherwise: it runs in a worker
        # thread and is only awaited by the success path, so a failure (or a
        # client disconnect) would leave it rendering pages and making paid
        # Textract calls for a response that is already an error.
        _xref_abort.set()
        for task in (_xref_task, _parse_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        if _hangup is not None and not _hangup.done():
            _hangup.cancel()
        timer.record_ms("total_ms", (time.perf_counter() - t0) * 1000.0)
        logger.info("schema %s timings=%s", route, timer.timings_ms)
