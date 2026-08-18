"""PDF extraction pipeline.

Emits one ``Chunk`` per PyMuPDF span (finest native granularity) for
native-text pages, plus one ``Chunk`` per Textract LAYOUT_TEXT block for
scanned pages. Image chunks are extracted in parallel when
``extract_images`` is true.

Page and size limits are enforced by raising ``PageLimitExceeded`` /
``DocumentTooLarge`` — never silently truncated.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
from dataclasses import dataclass, field

import httpx
import pymupdf

from extract.config import settings
from extract.core import loop_geometry
from extract.core.assemble import (
    assemble_chunks,
    decision_counters,
    render_table_html,
)
from extract.core.assemble import (
    render_table_markdown as _render_table_markdown,
)
from extract.core.chunking import (
    CHUNKING_VERSION,
    build_segments,
    page_dimensions,
    render_document_content,
    telemetry_counters,
)
from extract.core.errors import (
    ExtractionFailed,
    PageLimitExceeded,
    UnsupportedInput,
)
from extract.core.images import (
    ImageQualityFilter,
    compress_image_to_webp,
    should_compress,
)
from extract.core.io import load_bytes, repair_pdf_bytes
from extract.core.models import (
    Chunk,
    ChunkType,
    ExtractRequest,
    ExtractResponse,
    TableCell,
)
from extract.core.ocr import get_provider as get_ocr_provider
from extract.core.ocr.base import OCRPageResult, OCRTable
from extract.core.ocr.qwen_lora import (
    ESCALATED_RETRY_SAMPLING,
    RECOVERY_RETRY_SAMPLING,
    _extract_output_tokens,
    _extract_raw_text,
)
from extract.logger import get_logger
from extract.observability.timing import StageTimer, maybe_span
from extract.storage.base import Storage
from extract.storage.inline import InlineStorage

logger = get_logger()


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid integer for %s=%r; using default %d", name, raw, default)
            value = default
    if min_value is not None and value < min_value:
        logger.warning("%s=%d below minimum %d; clamping", name, value, min_value)
        value = min_value
    if max_value is not None and value > max_value:
        logger.warning("%s=%d above maximum %d; clamping", name, value, max_value)
        value = max_value
    return value


# Server-side guardrails — see the internal measurement records. Not user-configurable.
# Anything above either limit gets a 413 at the API layer.
PAGE_LIMIT = 2000
MAX_SIZE_BYTES = 150 * 1024 * 1024
OCR_IMAGE_RATIO_THRESHOLD = 0.75
OCR_PROVIDER = "textract"
# Default OCR stack for ALL traffic: our own Qwen3-VL model, with Gemini (eval-GT
# identical: gemini-3.1-pro-preview + thinking) as the per-page fallback when Qwen
# output is unusable/looping. OCR now runs on EVERY page ("always OCR" = the old
# `ocr="force"` behavior made permanent); the request `ocr` param is still accepted
# but no longer changes behavior. `OCR_PROVIDER` (textract) is kept only as the
# legacy constant referenced by non-default code paths and the eval harness.
OCR_PRIMARY_PROVIDER = "qwen_lora"
OCR_FALLBACK_PROVIDER = "gemini"
# Per-document page-OCR fan-out. Tune to the served model's throughput: it should
# match the model deployment's concurrency (max_running_requests / predict_concurrency)
# so the vLLM continuous batch stays full without queue build-up.
OCR_MAX_CONCURRENCY = _env_int("OCR_MAX_CONCURRENCY", 32, min_value=1)
# API-side raster policy for the default Qwen/Gemini OCR path. Keep the DPI as a
# quality ceiling, but cap pixels before upload so the API sends the same 2MP
# image budget that the model deployment serves.
OCR_DEFAULT_DPI = _env_int("OCR_DEFAULT_DPI", 150, min_value=72, max_value=300)
OCR_MAX_IMAGE_PIXELS = _env_int("OCR_MAX_IMAGE_PIXELS", 2_000_000, min_value=1)
DOTS_SUSPICIOUS_REPEAT_THRESHOLD = 6
DOTS_SEPARATOR_REPEAT_THRESHOLD = 40
DOTS_COMMON_SEPARATOR_TOKENS = frozenset({"-", "_", "."})
# Degenerate empty-cell loop (the Qwen markdown-table failure mode): the decode
# gets stuck emitting empty GFM cells ``|  |  |  | ...`` and never escapes. This
# is a DIFFERENT signal from ``_has_repeated_ngram``: the loop lives in a single
# unterminated ``text_content`` string, so ``parse_bbox_2d_json`` DROPS the whole
# truncated object and the parsed blocks/cells never carry it — the per-block
# ngram check sees nothing. So we scan the RAW model text instead.
#
# The LIVE predicate is ``core.loop_geometry`` (v2, experiment 130): structural, not
# width-based. Read that module for the derivation of every number it uses.
#
# ---- v1, FROZEN (2026-06-16 .. 2026-08-05) --------------------------------------
# Retained ONLY so a frozen off-line contract that pinned serving parity against it
# can still resolve the exact bytes it was fitted to (plan-114's reward floor imports
# this callable by name; its contract sha covers these two constants). NOT on any
# serving path. Do not add a caller — new work pins
# ``loop_geometry.LOOP_PREDICATE_VERSION``.
#
# It was calibrated on banked production scans: legit
# dense tables topped out at an 11-PIPE run always followed by more content; the one
# degenerate page (p0186) ends in a 22-PIPE run that IS the unterminated tail. Two
# defects that calibration hid, both measured in experiment 130:
#   * the comment says "empty-cell" but the code counts PIPES (N cells = N+1 pipes),
#     and that off-by-one leaked into EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN;
#   * n=1 positive and no wide-mostly-empty grid in the negatives, so a single
#     legitimate 24-blank-cell MAR row (25 pipes on one line) trips it. Over 97,297
#     correct pages it discards 235 — including 80 of the 83 plan-114 pages whose own
#     CORRECT GOLD fires it — while catching 60.16% of genuine loops.
#   * p0186 itself is ``max_tokens_hit: True``, so ``qwen_truncated`` had already
#     routed the sole calibrating exemplar before this predicate ever ran.
EMPTY_CELL_LOOP_TERMINAL_RUN = 12
EMPTY_CELL_LOOP_ABSOLUTE_RUN = 16
# A markdown empty cell = a "|" followed only by horizontal whitespace; a RUN of
# them is the empty-cell loop. Newlines break a run (a real multi-row table).
_EMPTY_CELL_RUN_RE = re.compile(r"(?:\|[^\S\r\n]*)+")
OCR_MIN_RELIABLE_TEXT_CHARS = 80
OCR_MIN_RELIABLE_BODY_TEXT_CHARS = 40
OCR_SHORT_TEXT_IMAGE_RATIO = 0.25
OCR_GIBBERISH_RATIO_THRESHOLD = 0.30
OCR_GLYPHLESS_FONT_RATIO_THRESHOLD = 0.60
OCR_BODY_MARGIN_RATIO = 0.08
OCR_BODY_MARGIN_MIN = 24.0
OCR_BODY_MARGIN_MAX = 72.0
OCR_VECTOR_ONLY_DRAWING_THRESHOLD = 80
EXTRACTION_ENGINE_BASELINE = "baseline"
VECTOR_MIN_CLUSTER_PATHS = 8
VECTOR_MIN_CLUSTER_AREA = 1_200
VECTOR_MAX_CLUSTER_PAGE_COVERAGE = 0.55
VECTOR_CLUSTER_GAP = 4.0
VECTOR_CLUSTER_PADDING = 3.0
VECTOR_PAGE_FURNITURE_MARGIN = 72.0
VECTOR_PAGE_FURNITURE_MIN_WIDTH_RATIO = 0.45
VECTOR_EDGE_PANEL_MARGIN = 128.0
VECTOR_EDGE_PANEL_MIN_WIDTH_RATIO = 0.80
VECTOR_UNBOUNDED_PAGE_COVERAGE = 0.80
VECTOR_UNBOUNDED_OVERFLOW_RATIO = 0.25
VECTOR_TABLE_ONLY_IOU = 0.65
VECTOR_TEXT_INSIDE_MAX_COUNT = 35
VECTOR_TEXT_INSIDE_MAX_AREA_RATIO = 0.08
VECTOR_LABEL_TEXT_MAX_AREA_RATIO = 0.18
VECTOR_LABEL_MIN_PATH_DENSITY_PER_1000_PT2 = 0.75
VECTOR_EXISTING_IMAGE_IOU = 0.80
VECTOR_TABLE_CONTAINMENT_MIN_AREA_RATIO = 0.50
PDF_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)
PDF_LIGATURE_CHARS = set(chr(codepoint) for codepoint in range(0xFB00, 0xFB07))
TEXT_FRAGMENT_MAX_GAP_RATIO = 0.08
TEXT_FRAGMENT_MAX_GAP_PT = 1.5
TEXT_FRAGMENT_MIN_Y_OVERLAP = 0.60


@dataclass(slots=True)
class PageOcrSignals:
    """Cheap page-level features used to decide whether OCR is worth paying for."""

    page_area: float
    text_chars: int = 0
    body_text_chars: int = 0
    chars_seen: int = 0
    replacement_chars: int = 0
    glyphless_font_chars: int = 0
    text_area: float = 0.0
    image_area: float = 0.0
    dominant_image_area: float = 0.0
    drawing_count: int = 0
    image_rects: list[pymupdf.Rect] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return self.text_chars > 0

    @property
    def has_any_text_signal(self) -> bool:
        return self.chars_seen > 0

    @property
    def image_area_ratio(self) -> float:
        if self.page_area <= 0:
            return 0.0
        return min(self.image_area / self.page_area, 1.0)

    @property
    def dominant_image_ratio(self) -> float:
        if self.page_area <= 0:
            return 0.0
        return min(self.dominant_image_area / self.page_area, 1.0)

    @property
    def replacement_ratio(self) -> float:
        if self.chars_seen <= 0:
            return 0.0
        return self.replacement_chars / self.chars_seen

    @property
    def glyphless_font_ratio(self) -> float:
        if self.text_chars <= 0:
            return 0.0
        return self.glyphless_font_chars / self.text_chars


async def extract_pdf(
    request: ExtractRequest,
    *,
    data: bytes | None = None,
    storage: Storage | None = None,
    download_client: httpx.AsyncClient | None = None,
    timer: StageTimer | None = None,
    enable_vector_images: bool = True,
    extraction_engine: str = EXTRACTION_ENGINE_BASELINE,
    phi_safe: bool = False,
    ocr_provider_name: str = OCR_PRIMARY_PROVIDER,
    block_private: bool = True,
) -> ExtractResponse:
    """Main entry point.

    Callers generally invoke this via ``Extractor``; this function is also
    safe to call directly from tests with ``data=<pdf bytes>``.
    """
    storage = storage or InlineStorage()

    if data is not None:
        pdf_bytes = data
    else:
        with maybe_span(timer, "download_ms"):
            pdf_bytes = await load_bytes(
                url=request.url,
                max_size=MAX_SIZE_BYTES,
                client=download_client,
                block_private=block_private,
            )
    if timer is not None:
        timer.meta["doc_size_bytes"] = len(pdf_bytes)

    doc = await asyncio.to_thread(_open_or_repair, pdf_bytes)
    try:
        num_pages = len(doc)
        if num_pages > PAGE_LIMIT:
            raise PageLimitExceeded(
                f"Document has {num_pages} pages, exceeds the {PAGE_LIMIT}-page limit."
            )
        # Captured while the doc is open (closed in the finally below). The
        # schema-extraction endpoint uses these to re-normalize chunk bboxes
        # (absolute points) to 0–1000 page-relative; the OCR response itself
        # does not serialize them.
        page_sizes = [(doc[i].rect.width, doc[i].rect.height) for i in range(num_pages)]

        # Pure-OCR pipeline: every page is rasterized and sent to our OCR model
        # (Qwen primary, Gemini fallback) bounded-concurrently. The model returns
        # the page's text, tables, and figure regions; there is no native PyMuPDF
        # text / table / image extraction. ``extract_text`` / ``extract_images``
        # gate which chunk kinds we return. Concurrency is tuned to the served
        # model's throughput (OCR_MAX_CONCURRENCY) — see _ocr_pages.
        skip_text_pages = set(range(num_pages)) if not request.extract_text else set()
        with maybe_span(timer, "ocr_ms"):
            chunks = await _ocr_pages(
                doc,
                list(range(num_pages)),
                provider_name=ocr_provider_name,
                fallback_provider_name=OCR_FALLBACK_PROVIDER,
                retry_unusable=True,
                skip_text_pages=skip_text_pages,
                include_figures=request.extract_images,
                collect_fallback_diagnostics=settings.EXTRACT_REVIEW_ARTIFACTS_ENABLED,
                table_output_format=request.table_output_format,
                phi_safe=phi_safe,
                source_pdf_bytes=pdf_bytes,
                timer=timer,
            )
        # Document-level assembly (plan 032): cross-page table merge, gated by
        # EXTRACT_ASSEMBLY_MODE (default "off" → this block is a no-op and the
        # output is byte-identical to pre-assembly behavior). Runs while the doc
        # is still open (page rects for the geometry gates; page rasters for
        # shadow diagnostics). "shadow" computes decisions + counters without
        # mutating output; "on" ships the merged chunks.
        if settings.EXTRACT_ASSEMBLY_MODE in ("shadow", "on") and num_pages > 1:
            page_heights = [doc[i].rect.height for i in range(num_pages)]
            assembled, decisions = assemble_chunks(chunks, page_heights)
            if timer is not None and decisions:
                timer.meta.update(decision_counters(decisions))
                timer.meta["assembly_mode"] = settings.EXTRACT_ASSEMBLY_MODE
                if settings.EXTRACT_REVIEW_ARTIFACTS_ENABLED:
                    # PHI-bearing per-boundary diagnostics (page images + the
                    # decision) ride the non-emitted sidecar, mirroring the
                    # gemini-fallback diagnostics pattern: never meta, never logs.
                    timer.sidecar[ASSEMBLY_DIAGNOSTICS_KEY] = [
                        {
                            "decision": d,
                            "page_a_image": await asyncio.to_thread(
                                _rasterize_page_for_ocr, doc[d.page_a - 1]
                            ),
                            "page_b_image": await asyncio.to_thread(
                                _rasterize_page_for_ocr, doc[d.page_b - 1]
                            ),
                        }
                        for d in decisions
                        if d.geometry_candidate
                    ]
            if settings.EXTRACT_ASSEMBLY_MODE == "on":
                chunks = assembled
        # HTML table serialization. A POST-PASS over the same cell model every
        # other mode uses, deliberately placed AFTER assembly so an assembled
        # chunk re-renders through the one canonical renderer rather than
        # carrying a stale markdown body. Every construction site keeps emitting
        # markdown, so this is the only place the two serializations can diverge.
        # A table with no cells (the degrade floor) keeps its markdown body and
        # echoes "markdown", exactly like the cell_grid fallback above.
        if request.table_output_format == "html":
            for chunk in chunks:
                if chunk.chunk_type != ChunkType.TABLE:
                    continue
                if chunk.cells and chunk.n_rows and chunk.n_cols:
                    chunk.page_content = render_table_html(
                        chunk.cells, chunk.n_rows, chunk.n_cols
                    )
                    chunk.table_output_format = "html"
                elif chunk.table_output_format is None:
                    chunk.table_output_format = "markdown"

    finally:
        doc.close()

    if timer is not None:
        timer.meta["page_count"] = num_pages
        timer.meta["pages_ocr_d"] = num_pages
        timer.meta["chunk_count_text"] = sum(1 for c in chunks if c.chunk_type == ChunkType.TEXT)
        timer.meta["chunk_count_image"] = sum(1 for c in chunks if c.chunk_type == ChunkType.IMAGE)
        timer.meta["chunk_count_table"] = sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)
        timer.meta["chunk_count_key_value"] = sum(
            1 for c in chunks if c.chunk_type == ChunkType.KEY_VALUE
        )
        _record_page_shape_metrics(timer, chunks, page_count=num_pages)
        timer.meta["text_chars_total"] = sum(
            len(c.page_content or "") for c in chunks if c.chunk_type == ChunkType.TEXT
        )

    # Plan 077: opt-in RAG chunking — a pure post-processing pass over the
    # final chunk list. Runs here (not in the routes) so sync, batch, and CLI
    # callers all inherit it. When off, the response fields stay None and the
    # serialized output is byte-identical to pre-chunking behavior.
    segments = None
    page_dims = None
    if request.chunking != "none":
        with maybe_span(timer, "chunking_ms"):
            segments = build_segments(chunks, chunk_size=request.chunk_size)
            page_dims = page_dimensions(page_sizes)
        if timer is not None:
            timer.meta["chunking_mode"] = request.chunking
            timer.meta["chunking_algo_version"] = CHUNKING_VERSION
            timer.meta.update(telemetry_counters(segments, chunk_size=request.chunk_size))

    # Opt-in whole-document content — like chunking, a pure post-processing pass
    # over the final chunk list, inherited by sync, batch, and CLI callers. Off
    # by default: the field stays None and is dropped on serialize.
    content = None
    if request.include_content:
        with maybe_span(timer, "content_render_ms"):
            content = render_document_content(chunks)

    response = ExtractResponse(
        chunks=chunks,
        segments=segments,
        page_dimensions=page_dims,
        content=content,
    )
    response._page_count = num_pages
    response._page_sizes = page_sizes
    return response


def _record_page_shape_metrics(timer: StageTimer, chunks: list[Chunk], *, page_count: int) -> None:
    per_page: dict[int, int] = {idx: 0 for idx in range(1, page_count + 1)}
    table_pages: set[int] = set()
    image_pages: set[int] = set()
    kv_pages: set[int] = set()
    for chunk in chunks:
        per_page[chunk.page_no] = per_page.get(chunk.page_no, 0) + 1
        if chunk.chunk_type == ChunkType.TABLE:
            table_pages.add(chunk.page_no)
        elif chunk.chunk_type == ChunkType.IMAGE:
            image_pages.add(chunk.page_no)
        elif chunk.chunk_type == ChunkType.KEY_VALUE:
            kv_pages.add(chunk.page_no)
    max_chunks = max(per_page.values(), default=0)
    timer.meta["empty_output_page_count"] = sum(1 for count in per_page.values() if count == 0)
    timer.meta["table_page_count"] = len(table_pages)
    timer.meta["image_page_count"] = len(image_pages)
    timer.meta["kv_page_count"] = len(kv_pages)
    timer.meta["max_chunks_per_page_bucket"] = _chunk_count_bucket(max_chunks)


def _chunk_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 10:
        return "1_10"
    if count <= 100:
        return "11_100"
    if count <= 1000:
        return "101_1000"
    return "gt_1000"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _usable_document(pdf_bytes: bytes) -> pymupdf.Document | None:
    """Open ``pdf_bytes`` and return it only if it has readable pages.

    PyMuPDF can reject bytes at either step, and by either mechanism: ``open``
    raises ``FileDataError``, or it succeeds and the page count then raises
    (``RuntimeError`` — a page tree whose ``/Count`` disagrees with reality) or
    comes back zero. All four verdicts are the same to a caller, so they are
    collapsed here, and anything unusable is closed before returning so no
    branch can leak a ``Document``.
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.FileDataError:
        return None
    try:
        if len(doc) > 0:
            return doc
    except RuntimeError:
        pass  # malformed page tree — unusable, same as reporting zero pages
    doc.close()
    return None


def _open_or_repair(pdf_bytes: bytes) -> pymupdf.Document:
    """Open a PDF, repairing once if PyMuPDF cannot use the bytes as-is.

    Previously only the zero-page case was handled, so an empty or corrupt
    upload escaped as a raw PyMuPDF exception and FastAPI turned it into a bare
    500 — telling the customer our server broke when their file did, and
    flagging ``error_customer_actionable=0`` on the one class of error only the
    customer can fix. ``converters/image.py`` already translates Pillow's decode
    failures this way; the PDF path never did.
    """
    if not pdf_bytes:
        # Nothing to repair, and "empty" is precisely the unreadable input that
        # Empty input maps to `unsupported_input` in the public errors table.
        raise UnsupportedInput("The PDF is empty.")
    doc = _usable_document(pdf_bytes)
    if doc is not None and doc.needs_pass:
        # Encryption is the one unusable input that passes every check above:
        # `open` does not raise, the page count is honest, and only the first
        # READ of a page fails — with `ValueError: document closed or
        # encrypted`, which has no typed branch, so it escaped as a bare 500
        # after several seconds of work. Observed on the public demo lane
        # (2026-08-01: 3 of 48 demo requests, a 6.3% failure rate against 0.3%
        # everywhere else) — strangers drop bank statements and payslips, which
        # are routinely password-protected. Repair cannot help: the bytes are
        # fine, we simply lack the password. Tell the customer that.
        doc.close()
        raise UnsupportedInput(
            "The PDF is password-protected. Remove the password and upload it again."
        )
    if doc is None:
        repaired = repair_pdf_bytes(pdf_bytes)
        if repaired:
            doc = _usable_document(repaired)
    if doc is None:
        raise ExtractionFailed("PDF is malformed and could not be repaired.")
    return doc


def _construct_chunk(**data: object) -> Chunk:
    construct = getattr(Chunk, "model_construct", None)
    if construct is not None:
        return construct(**data)
    return Chunk.construct(**data)


def _text_chunk(
    *,
    text: str,
    page_no: int,
    bbox: list[float] | None,
    confidence: float | None = None,
) -> Chunk:
    return _construct_chunk(
        page_content=text,
        page_no=page_no,
        bbox=bbox,
        chunk_type=ChunkType.TEXT,
        confidence=confidence,
    )


def _image_chunk(
    *,
    page_no: int,
    bbox: list[float] | None,
    image_url: str | None,
    image_b64: str | None,
    image_mime: str,
    image_width: int,
    image_height: int,
) -> Chunk:
    return _construct_chunk(
        page_content="",
        page_no=page_no,
        bbox=bbox,
        chunk_type=ChunkType.IMAGE,
        image_url=image_url,
        image_b64=image_b64,
        image_mime=image_mime,
        image_width=image_width,
        image_height=image_height,
    )


def _normalize_text_artifacts(text: str) -> str:
    if not text:
        return text
    return text.translate(PDF_LIGATURES)


def _table_chunk(*, table: OCRTable, page_no: int) -> Chunk:
    cells = [
        TableCell(
            text=c.text,
            row=c.row,
            col=c.col,
            row_span=c.row_span,
            col_span=c.col_span,
            bbox=c.bbox or None,
            confidence=c.confidence,  # B3 (plan 028): a real 0.0 must survive
        )
        for c in table.cells
    ]
    return _construct_chunk(
        page_content=_render_table_markdown(cells, table.n_rows, table.n_cols),
        page_no=page_no,
        bbox=table.bbox,
        chunk_type=ChunkType.TABLE,
        confidence=table.confidence,  # B3 (plan 028): a real 0.0 must survive
        cells=cells,
        n_rows=table.n_rows,
        n_cols=table.n_cols,
    )


def _kv_chunk(
    *,
    text: str,
    page_no: int,
    bbox: list[float] | None,
    confidence: float | None = None,
) -> Chunk:
    """A key-value form region. ``page_content`` is the pinned line grammar
    verbatim (``Key: Value`` per line, ``Key: <empty>``, ``[x]``/``[ ]`` lines, an
    optional leading title line); ``bbox`` is the region box. Unlike prose blocks
    it is NEVER overwritten by native-word extraction — the region text is a
    specific serialization, not free prose.
    See .claude/skills/data-curation/references/kv-region-contract.md."""
    return _construct_chunk(
        page_content=text,
        page_no=page_no,
        bbox=bbox,
        chunk_type=ChunkType.KEY_VALUE,
        confidence=confidence,
    )


def _table_chunk_for_page(
    *,
    table: OCRTable,
    page_no: int,
    page_rect: pymupdf.Rect,
) -> Chunk:
    cells = [
        TableCell(
            text=c.text,
            row=c.row,
            col=c.col,
            row_span=c.row_span,
            col_span=c.col_span,
            bbox=_clip_bbox_to_page(c.bbox, page_rect),
            confidence=c.confidence,  # B3 (plan 028): a real 0.0 must survive
        )
        for c in table.cells
    ]
    return _construct_chunk(
        page_content=_render_table_markdown(cells, table.n_rows, table.n_cols),
        page_no=page_no,
        bbox=_clip_bbox_to_page(table.bbox, page_rect),
        chunk_type=ChunkType.TABLE,
        confidence=table.confidence,  # B3 (plan 028): a real 0.0 must survive
        cells=cells,
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        # Cell-grid graceful degrade: a table the sidecar could not localize
        # echoes "markdown" so the response marks it as fallen back to the
        # markdown cell floor. Every other table is stamped "cell_grid" by the
        # post-pass in extract_pdf; a default (non-cell_grid) run leaves it None.
        table_output_format="markdown" if table.cell_grid_degraded else None,
    )


# _render_table_markdown moved to extract.core.assemble.render_table_markdown
# (single home; imported above) so the assembly layer re-renders merged tables
# through the exact same canonical GFM renderer as page-level tables.


def _extend_chunk_buckets(buckets: list[list[Chunk]], chunks: list[Chunk]) -> None:
    for chunk in chunks:
        buckets[chunk.page_no - 1].append(chunk)


def _get_page_images(
    page: pymupdf.Page,
    *,
    need_images: bool,
    need_ocr: bool,
) -> list:
    if not (need_images or need_ocr):
        return []
    return page.get_images(full=True)


def _probe_page_text(page: pymupdf.Page) -> bool:
    text = page.get_text(
        "text",
        flags=pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_PRESERVE_WHITESPACE,
    ).strip()
    return bool(text) and not _looks_like_gibberish(text)


def _detect_scan_fast_path(
    *,
    page: pymupdf.Page,
    image_list: list,
    mode: str,
) -> tuple | None:
    if mode != "auto":
        return None
    if not image_list:
        return None
    has_probe_text = _probe_page_text(page)
    return _scan_fast_path(
        has_text=has_probe_text,
        image_list=image_list,
        page_rect=page.rect,
    )


def _scan_page_text(
    page: pymupdf.Page,
    *,
    page_idx: int,
    collect_text: bool,
) -> tuple[list[Chunk], PageOcrSignals]:
    """Return native span chunks and cheap OCR-decision signals."""
    page_dict = page.get_text(
        "dict",
        flags=pymupdf.TEXT_PRESERVE_LIGATURES
        | pymupdf.TEXT_PRESERVE_WHITESPACE
        | pymupdf.TEXT_PRESERVE_IMAGES,
    )
    chunks: list[Chunk] = []
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    signals = PageOcrSignals(page_area=page_area)

    for block in page_dict.get("blocks", []):
        btype = block.get("type")
        bbox = block.get("bbox")
        if bbox is None:
            continue

        if btype == 0:  # text
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_text = span.get("text", "")
                    text = raw_text.strip()
                    if not text:
                        continue
                    signals.chars_seen += len(text)
                    signals.replacement_chars += text.count("\ufffd")
                    if _looks_like_gibberish(text):
                        continue

                    span_rect = _visible_rect_in_page_space(
                        pymupdf.Rect(*span.get("bbox", bbox)),
                        page,
                    )
                    if span_rect.is_empty:
                        continue
                    span_bbox = _bbox_from_rect(span_rect)
                    signals.text_chars += len(text)
                    signals.text_area += span_rect.get_area()
                    if _is_body_text_rect(span_rect, page_rect):
                        signals.body_text_chars += len(text)
                    font_name = str(span.get("font") or "").lower()
                    if "glyphless" in font_name:
                        signals.glyphless_font_chars += len(text)
                    if collect_text:
                        chunks.append(
                            _text_chunk(
                                text=text,
                                page_no=page_idx + 1,
                                bbox=span_bbox,
                            )
                        )
        elif btype == 1:  # image
            rect = _visible_rect_in_page_space(pymupdf.Rect(*bbox), page)
            if rect.is_empty:
                continue
            area = rect.get_area()
            signals.image_area += area
            signals.dominant_image_area = max(signals.dominant_image_area, area)
            signals.image_rects.append(rect)

    return _merge_fragmented_text_chunks(chunks), signals


def _merge_fragmented_text_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Join adjacent same-line text spans that are fragments of one word.

    PyMuPDF may split a single word across spans when a ligature or font change
    appears mid-word (for example ``de`` + ``fi`` + ``nes``). The spans touch
    geometrically, so merging them restores the visible word without another
    parsing pass.
    """
    if len(chunks) < 2:
        return [_normalized_text_chunk(chunk) for chunk in chunks]

    merged: list[Chunk] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        next_chunk = chunks[idx + 1] if idx + 1 < len(chunks) else None
        if next_chunk is not None and _should_merge_text_fragments(current, next_chunk):
            current = _merged_text_chunk(current, next_chunk)
            idx += 2
            tail = chunks[idx] if idx < len(chunks) else None
            if tail is not None and _should_merge_ligature_tail(current, tail):
                current = _merged_text_chunk(current, tail)
                idx += 1
            merged.append(_normalized_text_chunk(current))
            continue

        merged.append(_normalized_text_chunk(current))
        idx += 1
    return merged


def _should_merge_text_fragments(left: Chunk, right: Chunk) -> bool:
    if left.chunk_type != ChunkType.TEXT or right.chunk_type != ChunkType.TEXT:
        return False
    if left.page_no != right.page_no or not left.bbox or not right.bbox:
        return False
    left_text = left.page_content or ""
    right_text = right.page_content or ""
    if not left_text or not right_text:
        return False
    left_is_ligature_span = left_text in PDF_LIGATURE_CHARS
    right_is_ligature_span = right_text in PDF_LIGATURE_CHARS
    if not left_is_ligature_span and not right_is_ligature_span:
        return False
    if left_is_ligature_span and not _text_fragment_boundary(left_text[-1], right_text[0]):
        return False
    if right_is_ligature_span and not _text_fragment_boundary(left_text[-1], right_text[0]):
        return False

    return _same_line_tight_text_gap(left, right)


def _should_merge_ligature_tail(left: Chunk, right: Chunk) -> bool:
    if left.chunk_type != ChunkType.TEXT or right.chunk_type != ChunkType.TEXT:
        return False
    if left.page_no != right.page_no or not left.bbox or not right.bbox:
        return False
    left_text = left.page_content or ""
    right_text = right.page_content or ""
    if not left_text or not right_text:
        return False
    if left_text[-1] not in PDF_LIGATURE_CHARS:
        return False
    if not _text_fragment_boundary(left_text[-1], right_text[0]):
        return False

    return _same_line_tight_text_gap(left, right)


def _same_line_tight_text_gap(left: Chunk, right: Chunk) -> bool:
    left_rect = pymupdf.Rect(left.bbox)
    right_rect = pymupdf.Rect(right.bbox)
    if left_rect.is_empty or right_rect.is_empty:
        return False

    overlap = min(left_rect.y1, right_rect.y1) - max(left_rect.y0, right_rect.y0)
    min_height = max(min(left_rect.height, right_rect.height), 1.0)
    if overlap / min_height < TEXT_FRAGMENT_MIN_Y_OVERLAP:
        return False

    gap = right_rect.x0 - left_rect.x1
    max_gap = min(TEXT_FRAGMENT_MAX_GAP_PT, min_height * TEXT_FRAGMENT_MAX_GAP_RATIO)
    return -0.5 <= gap <= max_gap


def _text_fragment_boundary(left_char: str, right_char: str) -> bool:
    if left_char.isspace() or right_char.isspace():
        return False
    if left_char not in PDF_LIGATURE_CHARS and right_char not in PDF_LIGATURE_CHARS:
        return False
    if left_char.isalnum() and (right_char.isalnum() or right_char in "-_/"):
        return True
    return left_char in "-_/" and right_char.isalnum()


def _merged_text_chunk(left: Chunk, right: Chunk) -> Chunk:
    left_rect = pymupdf.Rect(left.bbox)
    right_rect = pymupdf.Rect(right.bbox)
    union = left_rect | right_rect
    confidence = None
    if left.confidence is not None and right.confidence is not None:
        confidence = min(left.confidence, right.confidence)
    elif left.confidence is not None:
        confidence = left.confidence
    elif right.confidence is not None:
        confidence = right.confidence
    return _text_chunk(
        text=f"{left.page_content or ''}{right.page_content or ''}",
        page_no=left.page_no,
        bbox=_bbox_from_rect(union),
        confidence=confidence,
    )


def _normalized_text_chunk(chunk: Chunk) -> Chunk:
    normalized = _normalize_text_artifacts(chunk.page_content or "")
    if normalized == (chunk.page_content or ""):
        return chunk
    return _text_chunk(
        text=normalized,
        page_no=chunk.page_no,
        bbox=chunk.bbox,
        confidence=chunk.confidence,
    )


def _is_body_text_rect(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    if rect.is_empty:
        return False
    margin = min(
        OCR_BODY_MARGIN_MAX,
        max(OCR_BODY_MARGIN_MIN, page_rect.height * OCR_BODY_MARGIN_RATIO),
    )
    return rect.y0 >= page_rect.y0 + margin and rect.y1 <= page_rect.y1 - margin


def _rect_in_page_space(rect: pymupdf.Rect, page: pymupdf.Page) -> pymupdf.Rect:
    """Return a rectangle in the same coordinate space as ``page.rect``.

    PyMuPDF may report embedded image bboxes in the unrotated media-box
    coordinate space on rotated pages. The public API uses rendered page
    coordinates, so choose the candidate that best fits the visible page.
    """
    if rect.is_empty or not page.rotation:
        return rect

    page_rect = page.rect
    rotated = rect * page.rotation_matrix

    def overflow_area(candidate: pymupdf.Rect) -> float:
        area = candidate.get_area()
        if area <= 0:
            return 0.0
        visible = candidate & page_rect
        return area - visible.get_area() if not visible.is_empty else area

    if overflow_area(rotated) < overflow_area(rect):
        return rotated
    return rect


def _visible_rect_in_page_space(rect: pymupdf.Rect, page: pymupdf.Page) -> pymupdf.Rect:
    return _rect_in_page_space(rect, page) & page.rect


def _bbox_from_rect(rect: pymupdf.Rect) -> list[float]:
    return [rect.x0, rect.y0, rect.x1, rect.y1]


def _clip_bbox_to_page(
    bbox: list[float] | None,
    page_rect: pymupdf.Rect,
) -> list[float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    rect = pymupdf.Rect(*bbox) & page_rect
    if rect.is_empty:
        return None
    return _bbox_from_rect(rect)


def _has_table_evidence(page: pymupdf.Page) -> bool:
    """Cheap pre-check before calling expensive ``page.find_tables()``.

    Triggers when the page has visible table-grid evidence in any of three
    forms:

    (a) **Ruled grid**: at least 2 distinct horizontal y-coordinates AND at
        least 2 distinct vertical x-coordinates of rule segments. Rectangles
        contribute their two horizontal edges (top + bottom) and two vertical
        edges (left + right).

    (b) **Booktabs-style**: at least 2 horizontal rules spanning >50% of page
        width — covers academic papers that draw only horizontal rules.

    (c) **Underline-heavy table**: at least 4 horizontal rules clustered
        within a 50-pt window — covers Apple 10-K-style tables that draw an
        underline beneath every row but have no vertical rules at all. Spread-
        out per-record header rules (e.g. invoice listings with one rule per
        record block) do NOT cluster tightly enough to trigger this.

    Important: ``get_cdrawings()`` returns rectangle items as a 4-tuple
    ``(x0, y0, x1, y1)``, NOT as a Rect object — the C-level format trades
    Pythonic accessors for speed. Treating the tuple as if it had ``.x0`` /
    ``.x1`` attributes silently drops every rectangle (caught by ``except``)
    and misses tables drawn with thin filled rectangles instead of stroked
    lines. We unpack as a tuple here.

    Whitespace-only tables (no rules at all) are an accepted miss: PyMuPDF's
    ``text``-strategy heuristics on those pages were producing false-positive
    detections that the downstream ``n_rows >= 2 ^ n_cols <= 20`` filter
    already discards.

    Cost: ``get_cdrawings()`` is ~0.25 ms/page on a 984-page invoice doc;
    skipping ``find_tables()`` saves ~13 ms/page locally / ~50 ms/page on
    Fargate where there's no grid evidence to find.
    """
    try:
        shapes = page.get_cdrawings()
    except Exception:  # pragma: no cover — defensive against PyMuPDF edge cases
        return True
    pw = page.rect.width
    h_ys: list[float] = []
    v_xs: set[int] = set()
    long_h_count = 0
    for s in shapes:
        for it in s.get("items", ()) or ():
            if not it:
                continue
            kind = it[0]
            if kind == "l":
                try:
                    (x0, y0), (x1, y1) = it[1], it[2]
                except (TypeError, ValueError):
                    continue
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                if dy < 0.5 and dx > 1:
                    h_ys.append(y0)
                    if pw > 0 and dx > 0.5 * pw:
                        long_h_count += 1
                elif dx < 0.5 and dy > 1:
                    v_xs.add(round(x0))
            elif kind == "re":
                if len(it) < 2:
                    continue
                r = it[1]
                if not (isinstance(r, (tuple, list)) and len(r) == 4):
                    continue
                try:
                    x0, y0, x1, y1 = r
                    rw = abs(x1 - x0)
                    rh = abs(y1 - y0)
                except Exception:
                    continue
                if rw < 0.5 or rh < 0.5:
                    continue
                if rh < 2 and rw > 1:
                    # Thin rectangle drawn as a horizontal rule (10-K style).
                    h_ys.append((y0 + y1) / 2)
                    if pw > 0 and rw > 0.5 * pw:
                        long_h_count += 1
                elif rw < 2 and rh > 1:
                    # Thin rectangle drawn as a vertical rule.
                    v_xs.add(round((x0 + x1) / 2))
                else:
                    # Real rectangle: contributes top + bottom + left + right.
                    h_ys.append(y0)
                    h_ys.append(y1)
                    v_xs.add(round(x0))
                    v_xs.add(round(x1))
                    if pw > 0 and rw > 0.5 * pw:
                        long_h_count += 2

    h_ys_unique = sorted({round(y) for y in h_ys})
    if len(h_ys_unique) >= 2 and len(v_xs) >= 2:
        return True
    if long_h_count >= 2:
        return True
    # Cluster check: 4+ horizontal rules within any 50-pt sliding window.
    if h_ys_unique:
        j = 0
        for i, yi in enumerate(h_ys_unique):
            while j < len(h_ys_unique) and h_ys_unique[j] - yi <= 50.0:
                j += 1
            if j - i >= 4:
                return True
    return False


def _scan_native_tables(page: pymupdf.Page, *, page_idx: int) -> list[Chunk]:
    """Detect tables on a born-digital page using PyMuPDF's native table finder.

    The OCR (Textract) path emits its own ``OCRTable`` chunks; this fills the
    equivalent gap for digital PDFs, which previously only got per-span text
    chunks (so a table's row-major reading destroyed structure on retrieval).
    Failures return an empty list — table detection is best-effort.
    """
    if not _has_table_evidence(page):
        return []
    try:
        finder = page.find_tables()
    except Exception:  # pragma: no cover — defensive against PyMuPDF edge cases
        return []
    chunks: list[Chunk] = []
    for tbl in getattr(finder, "tables", []) or []:
        try:
            rows = tbl.extract()
        except Exception:
            continue
        if not rows or not any(any((c or "").strip() for c in row) for row in rows):
            continue
        n_cols = max(len(r) for r in rows)
        # Pad ragged rows so the rendered grid is rectangular.
        rows = [list(r) + [""] * (n_cols - len(r)) for r in rows]
        n_rows = len(rows)
        # Guard against false positives. Real tables have at least 2 rows and
        # 2 columns; trivial detections (single-column lists, multi-column
        # prose, dot-leader TOCs) get reported as 1×N or N×1 grids and would
        # otherwise re-shape paragraph text into garbage markdown tables.
        if n_rows < 2 or n_cols < 2:
            continue
        # Wide grids of tiny cells are almost always layout artefacts —
        # token-level attention visualisations, calendar grids, code
        # alignment views. PyMuPDF reports them as 30-50 column "tables"
        # whose cells average 2-5 characters. Real wide tables (financial
        # pivots, multi-column comparison sheets) stay under 20 columns.
        if n_cols > 20:
            continue
        cells: list[TableCell] = []
        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                text = (val or "").strip().replace("\n", " ")
                if not text:
                    continue
                cells.append(
                    TableCell(
                        text=text,
                        row=r_idx,
                        col=c_idx,
                        row_span=1,
                        col_span=1,
                        bbox=None,
                        confidence=None,
                    )
                )
        if not cells:
            continue
        bbox = None
        if getattr(tbl, "bbox", None):
            table_rect = _visible_rect_in_page_space(pymupdf.Rect(tbl.bbox), page)
            if not table_rect.is_empty:
                bbox = _bbox_from_rect(table_rect)
        chunks.append(
            _construct_chunk(
                page_content=_render_table_markdown(cells, n_rows, n_cols),
                page_no=page_idx + 1,
                bbox=bbox,
                chunk_type=ChunkType.TABLE,
                confidence=None,
                cells=cells,
                n_rows=n_rows,
                n_cols=n_cols,
            )
        )
    return chunks


def _bbox_inside_any(child: list[float] | None, parents: list[list[float] | None]) -> bool:
    """True if the child bbox center sits inside any parent bbox."""
    if not child or len(child) != 4:
        return False
    cx = (child[0] + child[2]) / 2
    cy = (child[1] + child[3]) / 2
    for p in parents:
        if not p or len(p) != 4:
            continue
        if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]:
            return True
    return False


def _needs_ocr(
    *,
    signals: PageOcrSignals,
    page: pymupdf.Page | None = None,
    mode: str,
) -> bool:
    if mode != "auto":
        return mode == "force"
    if signals.page_area <= 0:
        return False
    if _has_existing_ocr_text_layer(signals):
        return False
    if _has_reliable_native_text(signals):
        return False
    if _has_corrupt_native_text(signals):
        return True

    image_ratio = signals.image_area_ratio
    dominant_ratio = signals.dominant_image_ratio
    has_image = bool(signals.image_rects)

    if not signals.has_text:
        if not has_image:
            return _has_dense_vector_content(signals, page)
        return image_ratio > 0.0 or dominant_ratio > 0.0

    # Header/footer-only or otherwise tiny text over a large page raster is a
    # common scan shape. OCR it, but don't do this for substantial native text.
    if dominant_ratio > OCR_IMAGE_RATIO_THRESHOLD:
        return True
    if image_ratio > OCR_IMAGE_RATIO_THRESHOLD:
        return True
    if image_ratio > OCR_SHORT_TEXT_IMAGE_RATIO:
        return True
    return _has_dense_vector_content(signals, page)


def _should_skip_ocr_text(
    *,
    signals: PageOcrSignals,
    mode: str,
    going_to_ocr: bool,
) -> bool:
    if not going_to_ocr:
        return True
    if mode == "force":
        return signals.has_text
    return _has_reliable_native_text(signals) or _has_existing_ocr_text_layer(signals)


def _has_reliable_native_text(signals: PageOcrSignals) -> bool:
    if signals.body_text_chars >= OCR_MIN_RELIABLE_BODY_TEXT_CHARS:
        return True
    return (
        signals.text_chars >= OCR_MIN_RELIABLE_TEXT_CHARS
        and signals.image_area_ratio < OCR_SHORT_TEXT_IMAGE_RATIO
    )


def _has_existing_ocr_text_layer(signals: PageOcrSignals) -> bool:
    return (
        signals.text_chars >= OCR_MIN_RELIABLE_BODY_TEXT_CHARS
        and signals.glyphless_font_ratio >= OCR_GLYPHLESS_FONT_RATIO_THRESHOLD
    )


def _has_corrupt_native_text(signals: PageOcrSignals) -> bool:
    return (
        signals.has_any_text_signal
        and not signals.has_text
        and signals.replacement_ratio >= OCR_GIBBERISH_RATIO_THRESHOLD
    )


def _has_dense_vector_content(signals: PageOcrSignals, page: pymupdf.Page | None) -> bool:
    if signals.drawing_count >= OCR_VECTOR_ONLY_DRAWING_THRESHOLD:
        return True
    if page is None:
        return False
    try:
        signals.drawing_count = len(page.get_cdrawings())
    except Exception:  # pragma: no cover - defensive against PyMuPDF edge cases
        return False
    return signals.drawing_count >= OCR_VECTOR_ONLY_DRAWING_THRESHOLD


def _looks_like_gibberish(text: str, threshold: float = 0.5) -> bool:
    """PyMuPDF emits U+FFFD when it can't map a glyph; if most of a span is
    replacement characters, treat it as unreadable and fall through to OCR.
    """
    if not text:
        return True
    bad = text.count("\ufffd")
    if bad == 0:
        return False
    return bad / len(text) > threshold


# ---------------------------------------------------------------------------
# Image scan / extract / upload
# ---------------------------------------------------------------------------


def _scan_page_images(
    *,
    page: pymupdf.Page,
    page_idx: int,
    doc: pymupdf.Document,
    has_text: bool,
    image_filter: ImageQualityFilter,
    image_list: list | None = None,
    fast_path_image: tuple | None = None,
) -> list[dict]:
    """Collect image candidates on a page. Runs pure-Python + PyMuPDF only."""
    out: list[dict] = []
    image_list = image_list if image_list is not None else page.get_images(full=True)
    if not image_list:
        return out

    fast = fast_path_image or _scan_fast_path(
        has_text=has_text,
        image_list=image_list,
        page_rect=page.rect,
    )
    candidates = [fast] if fast else image_list

    for img_info in candidates:
        xref = img_info[0]
        if not image_filter.check_duplicate(xref):
            continue
        smask_xref = img_info[1] if len(img_info) > 1 else 0
        has_mask = smask_xref != 0

        try:
            img_rects = page.get_image_rects(xref)
        except Exception:
            img_rects = []
        if not img_rects and not fast:
            continue
        img_bbox = _visible_rect_in_page_space(img_rects[0], page) if img_rects else page.rect
        if img_bbox.is_empty:
            continue

        try:
            base_img = doc.extract_image(xref)
            has_mask = (
                has_mask
                or base_img.get("smask") not in (None, 0)
                or base_img.get("smask_data") is not None
            )
            if has_mask:
                pix = pymupdf.Pixmap(doc, xref)
                try:
                    img_bytes = pix.tobytes(output="png")
                except Exception:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    img_bytes = pix.tobytes(output="png")
                img_ext = "png"
                width, height = pix.width, pix.height
            else:
                img_bytes = base_img["image"]
                img_ext = base_img["ext"]
                width = base_img.get("width", 0)
                height = base_img.get("height", 0)
        except Exception:
            img_bytes = None
            img_ext = "png"
            width = int(img_bbox.width)
            height = int(img_bbox.height)

        img_size = len(img_bytes) if img_bytes else 0

        meta_ok = image_filter.check_basic_metadata(
            width=width,
            height=height,
            file_size=img_size,
            page_width=page.rect.width,
            page_height=page.rect.height,
        )
        if not meta_ok:
            continue

        out.append(
            {
                "xref": xref,
                "page_idx": page_idx,
                "page": page,
                "img_rect": img_bbox,
                "img_bytes": img_bytes,
                "img_ext": img_ext,
                "width": width,
                "height": height,
                "size": img_size,
                "has_mask": has_mask,
                "bbox": [img_bbox.x0, img_bbox.y0, img_bbox.x1, img_bbox.y1],
            }
        )
    return out


def _scan_vector_figures(
    *,
    page: pymupdf.Page,
    page_idx: int,
    text_chunks: list[Chunk],
    exclude_bboxes: list[list[float] | None] | None = None,
    existing_image_bboxes: list[list[float] | None] | None = None,
) -> list[dict]:
    """Collect additive vector-drawn figure candidates.

    This is deliberately conservative. It may add an image chunk for a dense
    vector cluster, but it never suppresses native text or table chunks.
    """
    try:
        drawings = page.get_cdrawings()
    except Exception:  # pragma: no cover - defensive against PyMuPDF edge cases
        return []

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    if page_area <= 0:
        return []

    exclusions = [
        pymupdf.Rect(*bbox) for bbox in exclude_bboxes or [] if bbox is not None and len(bbox) == 4
    ]
    existing_images = [
        pymupdf.Rect(*bbox)
        for bbox in existing_image_bboxes or []
        if bbox is not None and len(bbox) == 4
    ]

    rects: list[pymupdf.Rect] = []
    for drawing in drawings:
        raw_rect = drawing.get("rect")
        if raw_rect is None:
            continue
        rect = pymupdf.Rect(raw_rect)
        if _is_unbounded_vector_drawing(rect, page_rect):
            continue
        if rect.is_empty:
            width = float(drawing.get("width") or 1.0)
            rect = rect + (-width, -width, width, width)
        rect = rect & page_rect
        if rect.is_empty:
            continue
        if _is_page_furniture_drawing(rect, page_rect):
            continue
        if rect.width < 0.5 and rect.height < 0.5:
            continue
        rects.append(rect)

    if not rects:
        return []

    candidates: list[dict] = []
    for cluster_rect, path_count in _cluster_rects(rects, gap=VECTOR_CLUSTER_GAP):
        padded = (
            cluster_rect
            + (
                -VECTOR_CLUSTER_PADDING,
                -VECTOR_CLUSTER_PADDING,
                VECTOR_CLUSTER_PADDING,
                VECTOR_CLUSTER_PADDING,
            )
        ) & page_rect
        area = padded.get_area()
        if path_count < VECTOR_MIN_CLUSTER_PATHS:
            continue
        if area < VECTOR_MIN_CLUSTER_AREA:
            continue
        if area / page_area > VECTOR_MAX_CLUSTER_PAGE_COVERAGE:
            continue
        if _is_page_furniture_cluster(padded, page_rect):
            continue
        if _is_table_like_vector_candidate(padded, exclusions):
            continue
        if _has_too_much_text_overlap(padded, text_chunks, path_count=path_count):
            continue
        if any(_rect_iou(padded, image) > VECTOR_EXISTING_IMAGE_IOU for image in existing_images):
            continue
        if any(
            _rect_iou(padded, existing["img_rect"]) > VECTOR_EXISTING_IMAGE_IOU
            for existing in candidates
        ):
            continue
        candidates.append(
            {
                "xref": None,
                "page_idx": page_idx,
                "page": page,
                "img_rect": padded,
                "img_bytes": None,
                "img_ext": "png",
                "width": int(max(1, padded.width * 2)),
                "height": int(max(1, padded.height * 2)),
                "size": 1_000,
                "has_mask": True,
                "bbox": [padded.x0, padded.y0, padded.x1, padded.y1],
                "source": "vector",
                "path_count": path_count,
            }
        )
    return candidates


def _cluster_rects(
    rects: list[pymupdf.Rect],
    *,
    gap: float,
) -> list[tuple[pymupdf.Rect, int]]:
    clusters: list[tuple[pymupdf.Rect, int]] = []
    for rect in sorted(rects, key=lambda r: (r.y0, r.x0)):
        expanded = rect + (-gap, -gap, gap, gap)
        matched = [idx for idx, (cluster, _) in enumerate(clusters) if expanded.intersects(cluster)]
        if not matched:
            clusters.append((pymupdf.Rect(rect), 1))
            continue
        first = matched[0]
        merged_rect, merged_count = clusters[first]
        merged_rect |= rect
        merged_count += 1
        for idx in reversed(matched[1:]):
            other_rect, other_count = clusters.pop(idx)
            merged_rect |= other_rect
            merged_count += other_count
        clusters[first] = (merged_rect, merged_count)
    return clusters


def _is_page_furniture_drawing(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    """Skip running-header/footer rules that can bridge into figure clusters."""
    min_width = page_rect.width * VECTOR_PAGE_FURNITURE_MIN_WIDTH_RATIO
    if rect.width >= min_width and rect.height <= 3.0:
        in_header = rect.y1 <= page_rect.y0 + VECTOR_PAGE_FURNITURE_MARGIN
        in_footer = rect.y0 >= page_rect.y1 - VECTOR_PAGE_FURNITURE_MARGIN
        return in_header or in_footer

    wide_edge_panel = rect.width >= page_rect.width * VECTOR_EDGE_PANEL_MIN_WIDTH_RATIO
    if not wide_edge_panel:
        return False
    in_header = rect.y0 <= page_rect.y0 + VECTOR_EDGE_PANEL_MARGIN
    in_footer = rect.y1 >= page_rect.y1 - VECTOR_EDGE_PANEL_MARGIN
    return in_header or in_footer


def _is_page_furniture_cluster(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    if (
        rect.width >= page_rect.width * VECTOR_PAGE_FURNITURE_MIN_WIDTH_RATIO
        and rect.height <= 12.0
    ):
        return True
    wide_edge_panel = rect.width >= page_rect.width * VECTOR_EDGE_PANEL_MIN_WIDTH_RATIO
    if not wide_edge_panel:
        return False
    in_header = rect.y0 <= page_rect.y0 + VECTOR_EDGE_PANEL_MARGIN
    in_footer = rect.y1 >= page_rect.y1 - VECTOR_EDGE_PANEL_MARGIN
    return in_header or in_footer


def _is_unbounded_vector_drawing(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    """Skip clipped paths whose reported geometry sprawls beyond the page."""
    if rect.is_empty:
        return False
    page_area = page_rect.width * page_rect.height
    if page_area <= 0:
        return False
    visible = rect & page_rect
    if visible.is_empty:
        return True
    if visible.get_area() / page_area < VECTOR_UNBOUNDED_PAGE_COVERAGE:
        return False
    overflow_x = max(0.0, page_rect.x0 - rect.x0) + max(0.0, rect.x1 - page_rect.x1)
    overflow_y = max(0.0, page_rect.y0 - rect.y0) + max(0.0, rect.y1 - page_rect.y1)
    overflow_ratio = max(
        overflow_x / page_rect.width if page_rect.width else 0.0,
        overflow_y / page_rect.height if page_rect.height else 0.0,
    )
    return overflow_ratio > VECTOR_UNBOUNDED_OVERFLOW_RATIO


def _is_table_like_vector_candidate(
    rect: pymupdf.Rect,
    table_rects: list[pymupdf.Rect],
) -> bool:
    rect_area = rect.get_area()
    if rect_area <= 0:
        return False
    for table_rect in table_rects:
        table_area = table_rect.get_area()
        if table_area <= 0:
            continue
        if _rect_iou(rect, table_rect) >= VECTOR_TABLE_ONLY_IOU:
            return True
        intersection = rect & table_rect
        if intersection.is_empty:
            continue
        table_to_candidate_area = table_area / rect_area
        if (
            table_to_candidate_area >= VECTOR_TABLE_CONTAINMENT_MIN_AREA_RATIO
            and intersection.get_area() / table_area >= 0.90
        ):
            return True
    return False


def _has_too_much_text_overlap(
    rect: pymupdf.Rect,
    text_chunks: list[Chunk],
    *,
    path_count: int = 0,
) -> bool:
    rect_area = rect.get_area()
    if rect_area <= 0:
        return True
    inside_count = 0
    text_area = 0.0
    for chunk in text_chunks:
        if chunk.chunk_type != ChunkType.TEXT or not chunk.bbox:
            continue
        text_rect = pymupdf.Rect(*chunk.bbox)
        if text_rect.is_empty:
            continue
        intersection = rect & text_rect
        if intersection.is_empty:
            continue
        center = ((text_rect.x0 + text_rect.x1) / 2, (text_rect.y0 + text_rect.y1) / 2)
        if rect.x0 <= center[0] <= rect.x1 and rect.y0 <= center[1] <= rect.y1:
            inside_count += 1
        text_area += intersection.get_area()
    text_area_ratio = text_area / rect_area
    if inside_count <= VECTOR_TEXT_INSIDE_MAX_COUNT:
        return text_area_ratio > VECTOR_TEXT_INSIDE_MAX_AREA_RATIO

    path_density = path_count / (rect_area / 1000.0) if rect_area > 0 else 0.0
    label_like_graphic = (
        path_density >= VECTOR_LABEL_MIN_PATH_DENSITY_PER_1000_PT2
        and text_area_ratio <= VECTOR_LABEL_TEXT_MAX_AREA_RATIO
    )
    return not label_like_graphic


def _rect_iou(left: pymupdf.Rect, right: pymupdf.Rect) -> float:
    intersection = left & right
    if intersection.is_empty:
        return 0.0
    inter_area = intersection.get_area()
    denom = left.get_area() + right.get_area() - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _scan_fast_path(
    *,
    has_text: bool,
    image_list: list,
    page_rect: pymupdf.Rect,
) -> tuple | None:
    """Scan-style page: no text + one dominant image covering the page."""
    if has_text or not image_list:
        return None
    if len(image_list) == 1:
        return image_list[0]

    def area(info):
        w = info[2] if len(info) > 2 else 0
        h = info[3] if len(info) > 3 else 0
        return int(w) * int(h)

    sized = sorted(((area(i), i) for i in image_list), reverse=True)
    dominant_area, dominant = sized[0]
    if dominant_area <= 0:
        return None
    total = sum(a for a, _ in sized)
    share = dominant_area / total if total else 0
    second_share = sized[1][0] / dominant_area if len(sized) > 1 and dominant_area else 0
    if share < 0.95 or second_share > 0.1:
        return None
    page_aspect = page_rect.width / page_rect.height if page_rect.height else 0
    image_width = dominant[2] if len(dominant) > 2 else 0
    image_height = dominant[3] if len(dominant) > 3 else 0
    image_aspect = image_width / image_height if image_height else 0
    if page_aspect > 0 and image_aspect > 0 and abs(page_aspect - image_aspect) > 0.2:
        return None
    return dominant


async def _filter_and_upload_images(
    candidates: list[dict],
    image_filter: ImageQualityFilter,
    storage: Storage,
) -> list[Chunk]:
    """Content-check + (render if needed) + compress + upload, in parallel."""

    async def _one(cand: dict) -> Chunk | None:
        img_bytes = cand.get("img_bytes")
        if img_bytes:
            ok = await asyncio.to_thread(image_filter.check_image_content, img_bytes)
            if not ok:
                return None

        try:
            upload_bytes, mime, width, height = await asyncio.to_thread(
                _finalize_image_bytes,
                cand,
                getattr(storage, "name", InlineStorage.name),
            )
        except Exception as e:
            logger.warning(
                "Image finalize failed for xref=%s page=%s: %s",
                cand.get("xref"),
                cand.get("page_idx", 0) + 1,
                e,
            )
            return None

        result = await storage.put(upload_bytes, mime=mime, prefix="images")
        return _image_chunk(
            page_no=cand["page_idx"] + 1,
            bbox=cand["bbox"],
            image_url=result.url,
            image_b64=result.b64,
            image_mime=mime,
            image_width=width,
            image_height=height,
        )

    results = await asyncio.gather(*(_one(c) for c in candidates))
    return [r for r in results if r is not None]


def _finalize_image_bytes(
    cand: dict,
    storage_name: str,
) -> tuple[bytes, str, int, int]:
    """Choose embedded-bytes fast path vs. page-clip render; compress to WebP
    if the result is big enough to benefit. Returns (bytes, mime, w, h).
    """
    img_bytes = cand.get("img_bytes")
    img_ext = cand.get("img_ext") or "png"
    has_mask = cand.get("has_mask", False)
    width = cand.get("width") or 0
    height = cand.get("height") or 0

    if (
        img_bytes
        and not has_mask
        and (len(img_bytes) <= 500 * 1024 or storage_name != InlineStorage.name)
    ):
        if _should_compress_for_storage(
            len(img_bytes),
            storage_name=storage_name,
        ):
            try:
                img_bytes, width, height, _ = compress_image_to_webp(img_bytes)
                return img_bytes, "image/webp", width, height
            except Exception:
                pass
        return img_bytes, f"image/{_normalize_ext(img_ext)}", width, height

    page: pymupdf.Page = cand["page"]
    clip = cand["img_rect"]
    scale = 2.0
    if clip.width > 0 and width:
        scale = max(1.0, min(width / clip.width, 2.0))
    png, rendered_width, rendered_height = _render_page_image_bytes(
        page,
        clip=clip,
        scale=scale,
    )
    if _should_compress_for_storage(
        len(png),
        storage_name=storage_name,
    ):
        try:
            out, w, h, _ = compress_image_to_webp(png)
            return out, "image/webp", w, h
        except Exception:
            pass
    return png, "image/png", rendered_width, rendered_height


def _render_page_image_bytes(
    page: pymupdf.Page,
    *,
    clip: pymupdf.Rect,
    scale: float,
) -> tuple[bytes, int, int]:
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        clip=clip,
        alpha=False,
        colorspace=pymupdf.csRGB,
    )
    return pix.tobytes(output="png"), pix.width, pix.height


def _should_compress_for_storage(
    size_bytes: int,
    *,
    storage_name: str,
) -> bool:
    if storage_name == InlineStorage.name:
        return should_compress(size_bytes)
    return False


def _normalize_ext(ext: str) -> str:
    e = ext.lower()
    if e == "jpg":
        return "jpeg"
    return e


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

# Identifies the Qwen-primary -> Gemini-fallback transition in metrics + bundles.
OCR_FALLBACK_KIND_QWEN_GEMINI = "qwen_lora_to_gemini"

# Key under ``StageTimer.sidecar`` holding the per-page fallback diagnostics the
# review-artifact writer consumes. The sidecar is never serialized into logs.
OCR_FALLBACK_DIAGNOSTICS_KEY = "gemini_fallback_pages"

# Reasons that demote a USABLE read into the fallback (quality detectors: the
# consistency gate and coverage route mode) — as opposed to the guard reasons,
# where the primary read is genuinely unusable. For demoted pages a CHUNK-LEVEL
# floor applies: if the fallback contributes zero servable chunks under the
# request's gates, the original read's chunks ship instead — a detector must
# never turn a servable page into an empty one. (Chunk-level on purpose: no
# provider-specific usability heuristic judges the fallback read — a
# Qwen-calibrated predicate must not veto a correct Gemini RTL read.)
_DEMOTED_USABLE_REASONS = frozenset({"qwen_inconsistent", "qwen_low_coverage"})

# Sidecar key for document-assembly boundary diagnostics (plan 032 §4 A4):
# one entry per geometry-candidate boundary with the BoundaryDecision and both
# page rasters. PHI-bearing — sidecar only, mirroring the fallback pattern.
ASSEMBLY_DIAGNOSTICS_KEY = "assembly_boundaries"


@dataclass
class PageFallbackDiagnostic:
    """One Qwen->Gemini fallback page, collected during ``_ocr_pages`` and read
    out-of-band (via ``StageTimer.sidecar``) by the review-artifact writer.

    Carries the rendered page bytes and both providers' raw model text, which is
    page content (PHI). It is deliberately kept off ``StageTimer.meta`` so it can
    never reach a log line or metric; only the encrypted, short-lived review
    bundle persists it.
    """

    page_no: int
    reason_code: str
    qwen_attempts: int
    image_bytes: bytes
    qwen_raw: object | None
    qwen_normalized: OCRPageResult | None
    qwen_chunk_counts: dict[str, int]
    gemini_raw: object | None
    gemini_normalized: OCRPageResult | None
    gemini_chunk_counts: dict[str, int]
    # Set when the Gemini fallback page is itself unusable (gemini_empty /
    # gemini_exception); ``None`` when the fallback produced a usable page.
    gemini_reason_code: str | None


def _max_empty_cell_run(raw: str) -> int:
    """Longest empty-GFM-cell run in ``raw``, measured in PIPE COUNT — the same
    unit :func:`_has_degenerate_empty_cell_loop` calls ``cells`` and calibrated
    its 12/16 thresholds in (N empty cells = N+1 pipes; the off-by-one is
    inherited deliberately so the at-risk threshold lives on the same scale).
    0 when none."""
    if not raw:
        return 0
    best = 0
    for m in _EMPTY_CELL_RUN_RE.finditer(raw):
        cells = m.group(0).count("|")
        if cells > best:
            best = cells
    return best


def _consistency_risk_reason(
    page_result, *, served_from_retry: bool, output_cap: int | None = None
) -> str | None:
    """Which measured volatility signal (if any) admits this usable read to the
    consistency gate's risk arm.

    The signals mirror the 2026-07-30 spec-decode probes: run-to-run instability
    is verdict-relevant only on the cap/loop-boundary class. A clean-LOOKING read
    sits on that boundary when (a) it was produced by a retry (the first attempt
    looped/truncated/emptied), (b) its raw output carries a sub-degenerate
    empty-cell run (the loop substrate, below the detector's trip thresholds), or
    (c) the decode finished near the serving token cap.
    """
    if served_from_retry:
        return "retry"
    raw = _extract_raw_text(page_result.raw) if page_result.raw is not None else ""
    if raw and _max_empty_cell_run(raw) >= settings.EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN:
        return "empty_run"
    n_out = _extract_output_tokens(page_result.raw)
    if n_out is not None:
        # The request's EFFECTIVE cap, not the global default: a sampling
        # override may lower/raise max_new_tokens and 85% must be measured
        # against what this decode was actually allowed.
        cap = output_cap or settings.PARSE_MODEL_MAX_NEW_TOKENS
        if n_out >= int(cap * settings.EXTRACT_OCR_CONSISTENCY_RISK_CAP_FRACTION):
            return "near_cap"
    return None


def _structural_disagreement(first_result, second) -> bool:
    """Coarse structural divergence between two reads of the same page — shadow
    telemetry only. Trips on a table-count mismatch or an empty-cell-run profile
    gap at least the at-risk run size (both cheap, both conservative; per-cell
    comparison is deliberately excluded — ragged GFM parses make it noisy)."""
    if len(first_result.tables) != len(second.tables):
        return True
    raw_a = _extract_raw_text(first_result.raw) if first_result.raw is not None else ""
    raw_b = _extract_raw_text(second.raw) if second.raw is not None else ""
    gap = abs(_max_empty_cell_run(raw_a) - _max_empty_cell_run(raw_b))
    return gap >= settings.EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN


async def _consistency_gate_pick(
    first_result,
    call_again,
    *,
    inconsistent_reason: str,
    require_min_values: bool = True,
    allow_revote: bool = True,
):
    """Numeric self-consistency vote over repeated reads of the SAME page (the
    production audit measured a volatile page subset: identical submissions return
    different numbers — invisible to any single-run guard BY DEFINITION).

    ``first_result`` must already be usable. Numeric-dense pages get a verification
    read via ``call_again`` (an async zero-arg provider call); on numeric
    disagreement a third read votes and the majority pair's earliest read is
    served; when no two reads agree the page is rejected with
    ``inconsistent_reason`` so the caller's normal per-page fallback runs instead
    of serving silently wrong numbers. A failed verification read fails OPEN (the
    first read is served — the gate must never make availability worse).

    Operational notes (review findings, documented deliberately):
    - Detection relies on serving-stack nondeterminism (continuous batching / GPU
      reduction order): decoding is greedy, yet a production silent-failure audit MEASURED the same
      page returning numeric f1 0.29/0.92/0.29/0.97 on identical prod calls. The
      enable-runbook still requires measuring checked/revoted rates first — if a
      future serving stack is fully deterministic this gate is dead weight and
      must stay dark.
    - On a 3-read vote the served read may be a LATER read (the majority pair's
      earliest member), i.e. the gate can swap the served read, not just
      keep-or-fallback.
    - Worst case per gated inconsistent page: verify + revote + the caller's
      fallback = up to 4 serial provider calls in one page slot (5 with a
      plan-028 retry before it — which is why stochastic-tier-accepted reads
      are excluded from gating at the call site).

    Returns ``(picked_result, reason_or_None, stats)`` where stats has
    ``checked``/``revoted``/``failed`` 0-or-1 counters.
    """
    from extract.core.numeric_agreement import (
        choose_majority_read,
        numeric_agreement,
        page_numeric_tokens,
    )

    stats = {"checked": 0, "revoted": 0, "failed": 0, "structural": 0}
    tokens_first = page_numeric_tokens(first_result)
    if require_min_values and len(tokens_first) < settings.EXTRACT_OCR_CONSISTENCY_MIN_VALUES:
        return first_result, None, stats

    async def _bounded_read():
        # Hard deadline per extra read: without it a sick verification read
        # inherits the provider's full retry loop (3 attempts x provider
        # timeout) while holding the page's concurrency slot. Expiry fails
        # open exactly like a crashed read (None -> unusable -> first served).
        try:
            return await asyncio.wait_for(
                call_again(), timeout=settings.EXTRACT_OCR_CONSISTENCY_VERIFY_TIMEOUT_S
            )
        except TimeoutError:
            return None

    stats["checked"] = 1
    second = await _bounded_read()
    if _ocr_unusable_reason(second) is not None:
        return first_result, None, stats  # fail open
    # Structural fingerprint — SHADOW ONLY in this iteration: the loop pathology
    # lives in raw text the parser drops, so numeric agreement can miss it. The
    # counter measures how often structure diverges while numerics agree; it
    # routes nothing until that rate is known (ragged-GFM parsing makes an
    # exact-cell-count demotion false-positive-prone).
    if _structural_disagreement(first_result, second):
        stats["structural"] = 1
    agreement = numeric_agreement(tokens_first, page_numeric_tokens(second))
    if agreement is None or agreement >= settings.EXTRACT_OCR_CONSISTENCY_MIN_AGREEMENT:
        return first_result, None, stats
    if not allow_revote:
        # Retry-produced read: disagreement goes straight to the caller's
        # fallback — the page has already spent its serial-call budget.
        stats["failed"] = 1
        return first_result, inconsistent_reason, stats
    stats["revoted"] = 1
    third = await _bounded_read()
    reads = [first_result, second]
    if _ocr_unusable_reason(third) is None:
        reads.append(third)
    pick = choose_majority_read(
        [page_numeric_tokens(r) for r in reads],
        settings.EXTRACT_OCR_CONSISTENCY_MIN_AGREEMENT,
    )
    if pick is None:
        stats["failed"] = 1
        return first_result, inconsistent_reason, stats
    return reads[pick], None, stats


def _shadow_page_coverage(image_bytes: bytes, page_result, page_w: float, page_h: float):
    """One-pass omission telemetry (OCR_COVERAGE_MODE): how much of the page's
    ink the served read's boxes fail to claim. Pure CPU on bytes already in hand;
    grid-normalized on both sides, so the adaptive raster DPI cancels out."""
    from extract.core.page_coverage import measure_coverage, page_result_boxes

    return measure_coverage(
        image_bytes,
        page_result_boxes(page_result, page_w=page_w, page_h=page_h),
        page_w=page_w,
        page_h=page_h,
    )


async def _ocr_pages(
    doc: pymupdf.Document,
    page_idxs: list[int],
    *,
    provider_name: str,
    fallback_provider_name: str | None = None,
    retry_unusable: bool = False,
    skip_text_pages: set[int] | None = None,
    include_figures: bool = True,
    collect_fallback_diagnostics: bool = False,
    table_output_format: str = "markdown",
    phi_safe: bool = False,
    source_pdf_bytes: bytes | None = None,
    timer: StageTimer | None = None,
) -> list[Chunk]:
    provider = get_ocr_provider(provider_name)
    if provider is None:
        logger.warning("OCR provider %r not registered; skipping OCR.", provider_name)
        return []
    fallback_provider = (
        get_ocr_provider(fallback_provider_name)
        if fallback_provider_name and fallback_provider_name != provider_name
        else None
    )

    skip_text_pages = skip_text_pages or set()
    semaphore = asyncio.Semaphore(max(1, min(len(page_idxs), OCR_MAX_CONCURRENCY)))
    primary_pages = 0
    fallback_pages = 0
    fallback_restored_pages = 0
    retry_pages = 0
    escalated_retry_pages = 0
    recovery_retry_pages = 0
    suspicious_pages = 0
    # SR-1 (2026-08-05): which limb condemned the page, and how long its blank-cell
    # run was. Both are low-cardinality integers over a fixed 7-name vocabulary; they
    # carry strictly less information than ``pages_qwen_suspicious``, which is already
    # PHI-allowlisted. They exist because 79-89% of the qwen_repeated_ngram class is
    # PHI-lane traffic with no retained output, so the limb split cannot be read any
    # other way — see the production-FP measurement §9.
    loop_limb_counts = {"word": 0, "cell": 0, "both": 0}
    loop_run_buckets = {"lt16": 0, "16_31": 0, "32_127": 0, "ge128": 0}
    # Post-flip confirmation: the retired v1 predicate's verdict alongside the live
    # one, so the offline verification stays checkable against real traffic.
    loop_shadow_counts = {"v1_suspicious": 0, "v1_only": 0, "v2_only": 0}
    reason_codes: list[str] = []
    diagnostics: list[PageFallbackDiagnostic] = []
    override_stats = {
        "pages_native_override": 0,
        "chunks_native_override": 0,
        "chunks_textlayer_recovered": 0,
    }
    gate_enabled = settings.EXTRACT_OCR_CONSISTENCY_GATE_ENABLED
    coverage_mode = settings.OCR_COVERAGE_MODE  # off | shadow | route
    consistency_checked = 0
    consistency_revoted = 0
    consistency_failed = 0
    consistency_checked_risk = 0
    consistency_checked_numeric = 0
    consistency_by_signal = {"retry": 0, "empty_run": 0, "near_cap": 0}
    consistency_disagree_by_signal = {"retry": 0, "empty_run": 0, "near_cap": 0}
    consistency_structural = 0
    consistency_admitted = 0
    consistency_skipped_budget = 0
    coverage_measured_pages = 0
    low_coverage_pages = 0
    coverage_max_uncovered = 0.0
    fastpath_pages = 0
    trusted_blank_pages = 0
    trusted_double_empty_pages = 0
    retry_override_floored = 0
    retry_override_judged = 0
    retry_override_rejected = 0
    fallback_ghost_dropped = 0
    tables_pipe_promoted = 0
    legal_postprocess_counts = {
        "transcript": 0,
        "multipanel": 0,
        "concordance": 0,
        "concordance_entry_count_mismatch": 0,
        "locator": 0,
        "locator_pdf_inspector": 0,
        "wrapped": 0,
        "exceptions": 0,
    }
    legal_pdf_bytes = source_pdf_bytes
    legal_pdf_bytes_failed = False

    def _apply_legal_postprocess(page_result, page, *, served_provider_name: str):
        """Run the CPU-only cascade on the event-loop thread.

        PyMuPDF does not support multithreaded access, so both ``page.get_text()``
        inside the adapter and the lazy whole-document serialization stay on the
        same thread as the rest of the PDF pipeline. Receipts stay count-only.
        """
        nonlocal legal_pdf_bytes, legal_pdf_bytes_failed
        from extract.core.legal_postprocess import apply_legal_postprocess

        processed, receipt = apply_legal_postprocess(
            page_result,
            page,
            provider_name=served_provider_name,
        )
        if receipt.get("needs_locator_evidence"):
            if legal_pdf_bytes is None and not legal_pdf_bytes_failed:
                try:
                    legal_pdf_bytes = doc.tobytes()
                except Exception:
                    # Source evidence is optional. A malformed/encrypted PDF that
                    # cannot be serialized must keep the already-selected model
                    # result instead of failing the whole parse request.
                    legal_pdf_bytes_failed = True
                    legal_postprocess_counts["exceptions"] += 1
            if legal_pdf_bytes_failed:
                return page_result
            processed, receipt = apply_legal_postprocess(
                page_result,
                page,
                provider_name=served_provider_name,
                pdf_bytes=legal_pdf_bytes,
            )
        route = receipt.get("route")
        if receipt.get("status") == "transformed" and route in legal_postprocess_counts:
            legal_postprocess_counts[route] += 1
            # Plan-146 entry-count audit observability: count-only, no content.
            if receipt.get("model_defect") == "model_native_entry_count_mismatch":
                legal_postprocess_counts["concordance_entry_count_mismatch"] += 1
        elif receipt.get("reason") == "postprocess_exception":
            legal_postprocess_counts["exceptions"] += 1
        return processed

    async def _one(idx: int) -> list[Chunk]:
        nonlocal primary_pages, fallback_pages, fallback_restored_pages
        nonlocal retry_pages, suspicious_pages
        nonlocal escalated_retry_pages, recovery_retry_pages
        # loop_limb_counts / loop_run_buckets are mutated in place, not rebound.
        nonlocal consistency_checked, consistency_revoted, consistency_failed
        nonlocal consistency_checked_risk, consistency_checked_numeric, consistency_structural
        nonlocal consistency_admitted, consistency_skipped_budget
        nonlocal consistency_by_signal, consistency_disagree_by_signal
        nonlocal coverage_measured_pages, low_coverage_pages, coverage_max_uncovered
        nonlocal fastpath_pages
        nonlocal trusted_blank_pages, fallback_ghost_dropped
        nonlocal trusted_double_empty_pages
        nonlocal retry_override_floored, retry_override_judged, retry_override_rejected
        nonlocal tables_pipe_promoted
        async with semaphore:
            page = doc[idx]
            # Plan 046 born-digital fast-path: if the page's text layer passes the
            # fail-closed trust gate, build chunks straight from it and skip the VLM
            # (rasterize + decode + retry + fallback). ~ms/page vs ~1.5s/page. Any
            # page that doesn't pass — scan, prior-OCR layer, corrupt, image-heavy —
            # falls through to the VLM path below exactly as today.
            if (
                settings.OCR_BORNDIGITAL_FASTPATH
                and idx not in skip_text_pages
                and _page_native_text_trustworthy(page, page_idx=idx)
            ):
                native_chunks, _ = _scan_page_text(page, page_idx=idx, collect_text=True)
                if native_chunks:
                    fastpath_pages += 1
                    return native_chunks
            used_fallback = False
            image_bytes = await asyncio.to_thread(_rasterize_page_for_ocr, page)
            page_result = await _call_ocr_provider(
                provider,
                image_bytes,
                page_width=page.rect.width,
                page_height=page.rect.height,
                page_no=idx + 1,
            )
            served_provider_name = provider_name
            qwen_attempts = 1
            accepted_overrides = None  # sampling tier of the read under verification
            reason = _ocr_unusable_reason(page_result)
            # v99 trusted-blank guard: the empty-page champion emits [] on
            # blank pages BY DESIGN. If the raster is visually blank, [] IS the
            # answer — skip retry AND the Gemini fallback (which hallucinates
            # ghost chunks on blank fax pages). Ink-bearing pages fall through
            # to the full legacy protection (retry -> fallback), keeping the
            # dropout insurance for real content (incl. sideways scans).
            if reason == "qwen_empty" and _page_visually_blank(image_bytes):
                trusted_blank_pages += 1
                reason = None
            primary_empty_result = page_result if reason == "qwen_empty" else None
            retry_result: OCRPageResult | None = None
            if retry_unusable and reason is not None:
                # Plan 028 retry tiers, in precedence order (all off in prod:
                # code defaults are off and the ECS task defs no longer override
                # them — OCR_RETRY_RECOVERY was true 96ef7f5f..2026-07-30, turned
                # off because sglang's NGRAM verify misindexes custom logit
                # processors (wrong-request bans under batching) and the same
                # path carries an uncatchable scheduler crash; see
                # the internal measurement records):
                #   OCR_RETRY_RECOVERY — A2 retry-only recovery: a greedy re-read
                #     whose ONLY override is the no_repeat_ngram=100 processor
                #     (RECOVERY_RETRY_SAMPLING; NO cap change — cap escalation
                #     was removed in cycle-4 after a null e2e, see qwen_lora.py).
                #     Do NOT re-enable while prod serves NGRAM spec-decode on
                #     sglang <fix — the guard reaches at most 1 of 16 draft rows
                #     and lands on the wrong request under batching;
                #   OCR_RETRY_ESCALATION — A1 temperature escalation (gate FAILED;
                #     kept dark, see a-retry/A1_VERDICT.json).
                # Both fire ONLY for the stochastic/diet failure modes — a
                # predominantly-RTL page is a capability gap, so the retry is
                # skipped entirely and the page routes straight to the fallback.
                retry_reasons = ("qwen_repeated_ngram", "qwen_truncated", "qwen_empty")
                recover = settings.OCR_RETRY_RECOVERY and reason in retry_reasons
                escalate = not recover and settings.OCR_RETRY_ESCALATION and reason in retry_reasons
                skip_retry = (
                    (settings.OCR_RETRY_RECOVERY or settings.OCR_RETRY_ESCALATION)
                    and reason == "qwen_low_quality_script"
                    and fallback_provider is not None
                )
                if not skip_retry:
                    retry_pages += 1
                    qwen_attempts = 2
                    if recover:
                        recovery_retry_pages += 1
                        overrides = RECOVERY_RETRY_SAMPLING
                    elif escalate:
                        escalated_retry_pages += 1
                        overrides = ESCALATED_RETRY_SAMPLING
                    else:
                        overrides = None
                    page_result = await _call_ocr_provider(
                        provider,
                        image_bytes,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                        page_no=idx + 1,
                        sampling_overrides=overrides,
                    )
                    retry_result = page_result
                    accepted_overrides = overrides
                    reason = _ocr_unusable_reason(page_result)
                    # A page still blank after the retry is a distinct, more telling
                    # signal than a first-attempt blank that the retry then fixed.
                    if reason == "qwen_empty":
                        reason = "qwen_empty_after_retry"
            # CHAMPION-EMPTY AUTHORITY (owner 2026-07-20: "trust the champion
            # more, especially on empty pages" — Gemini insurance stays for
            # every OTHER unusable reason):
            # (a) DOUBLE-EMPTY: the champion read the page as empty under BOTH
            #     decode tiers -> [] IS the answer; ship it, no Gemini call.
            # (b) A nonempty RETRY override of a champion-empty read must EARN
            #     the override: substance floor (kills "____"-class ghosts),
            #     then a one-word Gemini content JUDGE on the contested page.
            #     Judge "no" -> the champion's [] ships; judge error -> fail
            #     open to the override (a judge outage must never suppress
            #     real content).
            if primary_empty_result is not None:
                if reason == "qwen_empty_after_retry":
                    trusted_double_empty_pages += 1
                    page_result = primary_empty_result
                    reason = None
                elif reason is None and page_result is not primary_empty_result:
                    if _ocr_result_substance_chars(page_result) < _FALLBACK_MIN_TEXT_CHARS:
                        retry_override_floored += 1
                        page_result = primary_empty_result
                    else:
                        judge = getattr(fallback_provider, "judge_page_has_content", None)
                        verdict = await judge(image_bytes) if judge else None
                        retry_override_judged += 1
                        if verdict is False:
                            retry_override_rejected += 1
                            page_result = primary_empty_result
            v1_word, v1_cell, v2_word, v2_cell = _suspicious_limbs(page_result)
            if v2_word or v2_cell:
                suspicious_pages += 1
                # SR-1: attribute the LIVE fire by limb. Computed from the limbs already
                # evaluated on the line above: no extra pass, no provider call.
                loop_limb_counts["both" if (v2_word and v2_cell)
                                 else "word" if v2_word else "cell"] += 1
                _raw_for_bucket = (
                    _extract_raw_text(page_result.raw)
                    if page_result is not None and page_result.raw is not None else ""
                )
                _run = _max_empty_cell_run(_raw_for_bucket)
                loop_run_buckets[
                    "lt16" if _run < 16 else "16_31" if _run < 32
                    else "32_127" if _run < 128 else "ge128"
                ] += 1
            # POST-FLIP CONFIRMATION: the retired v1 predicate's verdict on the same
            # page, and where the two disagree. ``v1_only`` counts pages v2 stopped
            # discarding (the recovered false-positive class — expected to dominate);
            # ``v2_only`` counts pages v2 newly discards, which offline is entirely
            # cap-hit traffic ``qwen_truncated`` already routed. A ``v2_only`` rate that
            # departs from that is the signal to look.
            if v1_word or v1_cell:
                loop_shadow_counts["v1_suspicious"] += 1
            if (v1_word or v1_cell) and not (v2_word or v2_cell):
                loop_shadow_counts["v1_only"] += 1
            if (v2_word or v2_cell) and not (v1_word or v1_cell):
                loop_shadow_counts["v2_only"] += 1
            # The gate detects GREEDY-decode nondeterminism; a read accepted from
            # a deliberately stochastic retry tier (A1 escalation, temperature>0)
            # would disagree with its verification read BY CONSTRUCTION — every
            # such page would fire, revote, and burn up to 5 serial calls for a
            # disagreement the sampler caused. Those reads are not gate-eligible.
            # The A2 recovery tier stays eligible (greedy: no-repeat + raised cap).
            gate_eligible = not (
                accepted_overrides and float(accepted_overrides.get("temperature") or 0.0) > 0.0
            )
            # Provenance, not attempt count (2026-07-30 retune): the champion-empty
            # authority above can RESTORE the first-tier read while a retry still
            # happened — a restored read carries none of the retry's volatility
            # signal, so only a read that IS the retry's product counts.
            served_from_retry = retry_result is not None and page_result is retry_result
            if reason is None and gate_enabled and gate_eligible and fallback_provider is not None:
                # Self-consistency vote (metered canary): an unresolvable volatile
                # page becomes "qwen_inconsistent" and flows into the existing
                # per-page Gemini fallback below — a correct, already-paid-for
                # outcome instead of a silent bad serve. Eligibility is arm-based
                # (EXTRACT_OCR_CONSISTENCY_ELIGIBILITY): the risk arm targets the
                # measured volatile class (retry-produced / sub-degenerate
                # empty-cell run / near-cap generation); the numeric arm is the
                # original audit-derived numeric-density gate.
                mode = settings.EXTRACT_OCR_CONSISTENCY_ELIGIBILITY
                risk_signal = (
                    _consistency_risk_reason(
                        page_result,
                        served_from_retry=served_from_retry,
                        output_cap=(accepted_overrides or {}).get("max_new_tokens"),
                    )
                    if mode in ("risk", "both")
                    else None
                )
                numeric_arm = mode in ("numeric", "both")
                over_budget = False
                if risk_signal is not None or numeric_arm:
                    # Per-document admission budget (SLA guard): the extra reads
                    # run inside the page's concurrency slot, so a pathological
                    # document must not double its own read count. The check and
                    # increment have no await between them — race-free under
                    # asyncio's cooperative scheduling.
                    if consistency_admitted >= settings.EXTRACT_OCR_CONSISTENCY_MAX_PAGES_PER_DOC:
                        over_budget = True
                        consistency_skipped_budget += 1
                    else:
                        consistency_admitted += 1
                if (risk_signal is not None or numeric_arm) and not over_budget:
                    page_result, reason, cstats = await _consistency_gate_pick(
                        page_result,
                        lambda page=page, idx=idx, image_bytes=image_bytes, ovr=accepted_overrides: (
                            _call_ocr_provider(
                                provider,
                                image_bytes,
                                page_width=page.rect.width,
                                page_height=page.rect.height,
                                page_no=idx + 1,
                                sampling_overrides=ovr,  # same tier as the read under test
                            )
                        ),
                        inconsistent_reason="qwen_inconsistent",
                        # Risk-admitted pages skip the numeric-density floor —
                        # their volatility evidence is the signal itself.
                        require_min_values=risk_signal is None,
                        # A retry-produced page already cost 2 reads; on usable
                        # disagreement it routes straight to the fallback instead
                        # of a third read (serial-call cap 4, same as the retry
                        # path's own worst case).
                        allow_revote=not served_from_retry,
                    )
                    consistency_checked += cstats["checked"]
                    consistency_revoted += cstats["revoted"]
                    consistency_failed += cstats["failed"]
                    consistency_structural += cstats["structural"]
                    if cstats["checked"]:
                        if risk_signal is not None:
                            consistency_checked_risk += 1
                            # Per-signal canary split: which arm admitted the page,
                            # and did its verification read disagree (revote or
                            # reject)? A day of these two rates is the keep/kill
                            # evidence for each signal — near-cap especially.
                            consistency_by_signal[risk_signal] += 1
                            if cstats["revoted"] or cstats["failed"]:
                                consistency_disagree_by_signal[risk_signal] += 1
                        else:
                            consistency_checked_numeric += 1
                    qwen_attempts += cstats["checked"] + cstats["revoted"]
            if reason is None and coverage_mode != "off":
                # Omission coverage: measure + count (shadow), and in ROUTE mode
                # demote a low-coverage page into the existing per-page fallback.
                # ~30ms of numpy on bytes already in hand; zero provider calls.
                # FAIL-OPEN by contract: a measurement error must never route the
                # page or fail the customer's document (the page-level gather has
                # no return_exceptions — an escape here would kill the request).
                try:
                    cov = await asyncio.to_thread(
                        _shadow_page_coverage,
                        image_bytes,
                        page_result,
                        page.rect.width,
                        page.rect.height,
                    )
                    from extract.core.page_coverage import is_low_coverage

                    coverage_measured_pages += 1
                    coverage_max_uncovered = max(coverage_max_uncovered, cov.uncovered_fraction)
                    if is_low_coverage(
                        cov,
                        max_uncovered_fraction=settings.OCR_COVERAGE_MAX_UNCOVERED,
                        min_uncovered_band_frac=settings.OCR_COVERAGE_MIN_BAND_FRAC,
                    ):
                        low_coverage_pages += 1
                        if coverage_mode == "route":
                            # Demote the read; the fallback block below serves the
                            # page via Gemini, exactly like any guard-caught page.
                            reason = "qwen_low_coverage"
                except Exception:  # noqa: BLE001 — coverage measurement is fail-open
                    logger.debug("coverage measurement failed", exc_info=True)
            demoted_result = None
            if reason is not None and fallback_provider is not None:
                # Keep the primary read as the floor NO MATTER why it was
                # rejected. The old rule kept it only for demoted-usable reads,
                # on the reasoning that a guard-caught read is unusable and an
                # empty page is an acceptable contract — which holds only while
                # the fallback actually serves the page. When the fallback
                # itself throws (gemini_exception) nothing serves it, and we
                # shipped an empty page for a document the customer was billed
                # for in full. Observed in production 2026-08-03: a 7-page
                # financial statement lost its Balance Sheet page (221 Qwen
                # chunks discarded) and still returned 200.
                #
                # This cannot regress a working fallback: the restore below is
                # gated on `not chunks`, so it only ever replaces an EMPTY page,
                # never a good one. Degenerate reads are held back by the
                # substance floor at the restore site.
                demoted_result = page_result
                fallback_pages += 1
                used_fallback = True
                reason_codes.append(reason)
                qwen_result = page_result
                page_result = await _call_ocr_provider(
                    fallback_provider,
                    image_bytes,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                    page_no=idx + 1,
                )
                served_provider_name = fallback_provider_name or fallback_provider.name
                if collect_fallback_diagnostics:
                    # Captured BEFORE ``page_result`` is overwritten above so we keep
                    # both the rejected Qwen page and the Gemini page that shipped.
                    diagnostics.append(
                        PageFallbackDiagnostic(
                            page_no=idx + 1,
                            reason_code=reason,
                            qwen_attempts=qwen_attempts,
                            image_bytes=image_bytes,
                            qwen_raw=getattr(qwen_result, "raw", None),
                            qwen_normalized=qwen_result,
                            qwen_chunk_counts=_ocr_result_chunk_counts(qwen_result),
                            gemini_raw=getattr(page_result, "raw", None),
                            gemini_normalized=page_result,
                            gemini_chunk_counts=_ocr_result_chunk_counts(page_result),
                            gemini_reason_code=_gemini_fallback_reason(page_result),
                        )
                    )
                # Ghost floor: a fallback "rescue" of a qwen_empty page must
                # EARN the override. Gemini hallucinates tiny ghost chunks
                # ("The"/"1"/"I") on blank/near-blank fax pages; if the
                # fallback yields fewer than _FALLBACK_MIN_TEXT_CHARS
                # alphanumeric characters (and no table), the model's []
                # ships instead. A real rescue (a genuinely dropped text line)
                # clears the floor untouched. Routing-only — fallback output
                # is never edited, only accepted or rejected whole.
                if (
                    reason in ("qwen_empty", "qwen_empty_after_retry")
                    and _ocr_result_substance_chars(page_result) < _FALLBACK_MIN_TEXT_CHARS
                ):
                    page_result = qwen_result
                    served_provider_name = provider_name
                    fallback_ghost_dropped += 1
            if page_result is None and demoted_result is None:
                return []
            if not used_fallback and fallback_provider is not None:
                primary_pages += 1
            # Plan 132 legal-table cascade: after retry/fallback selection, before
            # table geometry sidecars and public chunk construction. Native PDF text
            # is routing/layout evidence only; an abstention returns the original
            # OCRPageResult object. Skip text-suppressed pages because this treatment
            # is a text serialization and must not override the caller's omission.
            if (
                settings.OCR_LEGAL_POSTPROCESS
                and provider_name == OCR_PRIMARY_PROVIDER
                and page_result is not None
                and idx not in skip_text_pages
            ):
                page_result = _apply_legal_postprocess(
                    page_result,
                    page,
                    served_provider_name=served_provider_name,
                )
            # Pipe-tabular promotion: the champion frequently reads a table
            # correctly but serializes it as delimiter-less pipe text, so it
            # was typed ``text`` and shipped without cells. Deterministic
            # post-typing on the served result: such blocks become ordinary
            # table chunks (cells share the block bbox; page_content
            # re-renders as canonical GFM). Receipts: plan 058 (extract-bench
            # paired no-reg + typed-table recovery 17→23/24 on a production slice).
            if page_result is not None and page_result.blocks:
                from extract.core.ocr.tabular_promotion import promote_tabular_blocks

                tables_pipe_promoted += promote_tabular_blocks(page_result)
        chunks = (
            _chunks_from_ocr_page_result(
                doc,
                idx,
                page_result,
                image_bytes=image_bytes,
                skip_text=idx in skip_text_pages,
                include_figures=include_figures,
                override_stats=override_stats,
            )
            if page_result is not None
            else []
        )
        if not chunks and _restore_is_admissible(reason, demoted_result):
            # QUALITY FLOOR for demoted-usable reads: judged at the CHUNK level
            # (what the customer actually receives under this request's gates),
            # never by provider-specific usability heuristics — a Qwen-calibrated
            # predicate must not veto a correct Gemini read (e.g. RTL pages), and
            # an images-only request must not lose table/figure chunks to a
            # text-only fallback. If the fallback contributed nothing servable,
            # the original read's chunks ship instead of an empty page.
            # Plan 058: the demoted read never saw promotion or the cell
            # ladder (both ran on the served-then-empty result) — apply them
            # here too, so a restored page matches the served contract.
            if (
                settings.OCR_LEGAL_POSTPROCESS
                and provider_name == OCR_PRIMARY_PROVIDER
                and idx not in skip_text_pages
            ):
                demoted_result = _apply_legal_postprocess(
                    demoted_result,
                    page,
                    served_provider_name=provider_name,
                )
            if demoted_result.blocks:
                from extract.core.ocr.tabular_promotion import promote_tabular_blocks

                tables_pipe_promoted += promote_tabular_blocks(demoted_result)
            restored = _chunks_from_ocr_page_result(
                doc,
                idx,
                demoted_result,
                image_bytes=image_bytes,
                skip_text=idx in skip_text_pages,
                include_figures=include_figures,
                # throwaway stats: the first build already counted this page —
                # a rare restore must not double-count native-override counters.
                override_stats={"pages_native_override": 0, "chunks_native_override": 0},
            )
            if restored:
                fallback_restored_pages += 1
                chunks = restored
        return chunks

    results = await asyncio.gather(*(_one(i) for i in page_idxs))
    out: list[Chunk] = []
    for batch in results:
        out.extend(batch)
    if (
        timer is not None
        and provider_name == OCR_PRIMARY_PROVIDER
        and fallback_provider is not None
    ):
        # Low-cardinality, PHI-safe request metrics for the LIVE Qwen->Gemini path.
        # Page
        # numbers and page content stay OUT of ``meta`` (they would leak into the
        # non-PHI CloudWatch line) — they live only in the encrypted review bundle,
        # surfaced via the non-emitted ``sidecar``.
        timer.meta["pages_qwen_ocr"] = primary_pages
        timer.meta["pages_qwen_retry"] = retry_pages
        timer.meta["pages_qwen_suspicious"] = suspicious_pages
        # SR-1 (2026-08-05): the limb split and the blank-cell run histogram for the
        # pages counted above. Seven fixed names, integers only. This is the ONLY
        # instrument that can read the PHI lane's qwen_repeated_ngram class, which is
        # 79-89% of it and has no retained output by compliance design. The buckets
        # are chosen against the two measured populations, not free integers: the
        # wide-grid envelope is exactly 16 pipes and the genuine-loop envelope is
        # 7,448-8,158, with the 17-7,447 band empty in the production sample — so
        # ``16_31`` vs ``ge128`` separates them with no ambiguity.
        for _limb, _n in loop_limb_counts.items():
            timer.meta[f"pages_qwen_loop_limb_{_limb}"] = _n
        for _bucket, _n in loop_run_buckets.items():
            timer.meta[f"pages_qwen_loop_run_{_bucket}"] = _n
        for _k, _n in loop_shadow_counts.items():
            timer.meta[f"pages_qwen_loop_{_k}"] = _n
        timer.meta["pages_gemini_fallback"] = fallback_pages
        timer.meta["fallback_count"] = fallback_pages
        # Pages where a failed/empty fallback met a restorable primary read and
        # that read was served instead (the chunk-level floor). Covers quality-
        # detector demotions AND the guard-caught reasons in
        # ``_RESTORABLE_GUARD_REASONS``. Nonzero while routing/gating is enabled
        # means the fallback lane is degraded — the alarm condition the floor
        # exists for.
        timer.meta["pages_fallback_restored"] = fallback_restored_pages
        # A1 (plan 028): how many retries went out with the escalated tier (0
        # whenever the dark flag is off — keeps dashboards diff-able pre/post).
        if settings.OCR_RETRY_ESCALATION:
            timer.meta["pages_qwen_retry_escalated"] = escalated_retry_pages
        # A2 retry-only recovery (plan 028 promotion).
        if settings.OCR_RETRY_RECOVERY:
            timer.meta["pages_qwen_retry_recovery"] = recovery_retry_pages
        # D3 (plan 028): native-text override fire counts (only emitted when the
        # dark flag is on, so the metric's presence marks flagged traffic).
        if settings.OCR_NATIVE_TEXT_OVERRIDE:
            timer.meta["pages_native_override"] = override_stats["pages_native_override"]
            timer.meta["chunks_native_override"] = override_stats["chunks_native_override"]
        # Text-layer dollar-completeness recovery fire count (only emitted when the
        # dark flag is on, so the metric's presence marks flagged traffic).
        if settings.OCR_TEXTLAYER_COMPLETENESS:
            timer.meta["chunks_textlayer_recovered"] = override_stats["chunks_textlayer_recovered"]
        # Self-consistency gate counters (only emitted when the dark flag is on,
        # so the metric's presence marks gated traffic).
        if gate_enabled:
            timer.meta["pages_ocr_consistency_checked"] = consistency_checked
            timer.meta["pages_ocr_consistency_revoted"] = consistency_revoted
            timer.meta["pages_ocr_consistency_failed"] = consistency_failed
            timer.meta["pages_ocr_consistency_checked_risk"] = consistency_checked_risk
            timer.meta["pages_ocr_consistency_checked_numeric"] = consistency_checked_numeric
            timer.meta["pages_ocr_consistency_structural"] = consistency_structural
            timer.meta["pages_ocr_consistency_skipped_budget"] = consistency_skipped_budget
            for _sig, _n in consistency_by_signal.items():
                timer.meta[f"pages_ocr_consistency_checked_{_sig}"] = _n
                timer.meta[f"pages_ocr_consistency_disagree_{_sig}"] = (
                    consistency_disagree_by_signal[_sig]
                )
        # Omission-coverage counters (only emitted when measuring). The measured
        # count is the explicit denominator: caught/fallback pages are never
        # measured, so rates must be computed against THIS, not page_count.
        if coverage_mode != "off" and coverage_measured_pages:
            timer.meta["pages_coverage_measured"] = coverage_measured_pages
            timer.meta["pages_low_coverage"] = low_coverage_pages
            timer.meta["coverage_max_uncovered"] = round(coverage_max_uncovered, 4)
        if fallback_pages:
            timer.meta["fallback_kind"] = OCR_FALLBACK_KIND_QWEN_GEMINI
            timer.meta["fallback_reason_codes"] = ",".join(sorted(set(reason_codes)))
        if diagnostics:
            timer.sidecar[OCR_FALLBACK_DIAGNOSTICS_KEY] = diagnostics
    # Plan 046 born-digital fast-path fire count (only when the dark flag is on, so
    # the metric's presence marks flagged traffic). Emitted independently of the
    # Qwen->Gemini meta block above so it also surfaces on eval/no-fallback runs.
    if timer is not None and settings.OCR_BORNDIGITAL_FASTPATH:
        timer.meta["pages_native_fastpath"] = fastpath_pages
    # Trusted-blank guard + ghost floor counters (always-on behavior, so always
    # emitted): how many pages shipped the model's [] because the raster was
    # visually blank / because the fallback rescue was an insubstantial ghost.
    if timer is not None:
        timer.meta["pages_trusted_blank"] = trusted_blank_pages
        timer.meta["pages_trusted_double_empty"] = trusted_double_empty_pages
        timer.meta["pages_retry_override_floored"] = retry_override_floored
        timer.meta["pages_retry_override_judged"] = retry_override_judged
        timer.meta["pages_retry_override_rejected"] = retry_override_rejected
        timer.meta["pages_fallback_ghost_dropped"] = fallback_ghost_dropped
    # Pipe-tabular promotion count (all modes; emitted only when it fired so
    # the metric's presence marks affected pages). Counts only — no content.
    if timer is not None and tables_pipe_promoted:
        timer.meta["tables_pipe_promoted"] = tables_pipe_promoted
    if timer is not None and settings.OCR_LEGAL_POSTPROCESS:
        for route in (
            "transcript",
            "multipanel",
            "concordance",
            "concordance_entry_count_mismatch",
            "locator",
            "locator_pdf_inspector",
            "wrapped",
        ):
            timer.meta[f"pages_legal_postprocess_{route}"] = legal_postprocess_counts[route]
        timer.meta["pages_legal_postprocess_exceptions"] = legal_postprocess_counts["exceptions"]
    return out


def _chunks_from_ocr_page_result(
    doc: pymupdf.Document,
    idx: int,
    page_result,
    *,
    image_bytes: bytes,
    skip_text: bool,
    include_figures: bool,
    override_stats: dict | None = None,
) -> list[Chunk]:
    page = doc[idx]
    # 2026-07-30 reading-order fix: chunks are assembled in the provider's own
    # emission order when elements carry ``seq`` (the fine-tuned model emits in
    # reading order — column-aware, tables in place). Elements without ``seq``
    # (fallback providers) keep the legacy type-bucketed order EXACTLY: they
    # sort as +inf on the seq key, and the tiebreak preserves the historical
    # blocks → kv → tables → figures grouping.
    entries: list[tuple[float, int, Chunk]] = []

    def _push(seq: int | None, chunk: Chunk) -> None:
        key = float(seq) if seq is not None else float("inf")
        entries.append((key, len(entries), chunk))

    if not skip_text:
        # D2/D3 (plan 028, dark flag): on a trustworthy born-digital page, replace
        # the VLM transcription of each text block with the PDF's own text layer
        # clipped to the block's bbox. Bboxes never change (grounding untouched);
        # any per-chunk ambiguity keeps the VLM text. Tables/figures excluded in
        # v1 — clip extraction would destroy GFM structure.
        native_words = None
        if (
            settings.OCR_NATIVE_TEXT_OVERRIDE
            and page_result.blocks
            and _page_native_text_trustworthy(page, page_idx=idx)
        ):
            native_words = page.get_text("words")
        page_overrides = 0
        for b in page_result.blocks:
            if not b.text:
                continue
            bbox = _clip_bbox_to_page(b.bbox, page.rect)
            text = b.text
            if native_words and bbox:
                native_text = _native_text_in_rect(native_words, pymupdf.Rect(bbox))
                if native_text and _native_text_matches_vlm(native_text, b.text):
                    text = native_text
                    page_overrides += 1
            _push(
                getattr(b, "seq", None),
                _text_chunk(
                    text=text,
                    page_no=idx + 1,
                    bbox=bbox,
                    # B3 (plan 028): pass the provider confidence through verbatim.
                    # The old ``b.confidence or None`` coerced a REAL 0.0 to None,
                    # so the lowest-confidence chunks were the ones that lost
                    # their score; providers now emit None when they have no
                    # signal, so no coercion belongs here.
                    confidence=b.confidence,
                ),
            )
        if page_overrides and override_stats is not None:
            override_stats["pages_native_override"] += 1
            override_stats["chunks_native_override"] += page_overrides
    # KV form regions: the region text is a pinned serialization (Key: Value
    # lines), never overwritten by native-word extraction the way prose blocks are.
    for kv in page_result.key_values:
        if not (kv.text or "").strip():
            continue
        _push(
            getattr(kv, "seq", None),
            _kv_chunk(
                text=kv.text,
                page_no=idx + 1,
                bbox=_clip_bbox_to_page(kv.bbox, page.rect),
                confidence=kv.confidence,
            ),
        )
    for t in page_result.tables:
        _push(
            getattr(t, "seq", None),
            _table_chunk_for_page(table=t, page_no=idx + 1, page_rect=page.rect),
        )
    if include_figures and page_result.figures:
        for fig, fig_chunk in _figure_chunks_from_raster(
            image_bytes=image_bytes,
            figures=page_result.figures,
            page_no=idx + 1,
            page_rect=page.rect,
        ):
            _push(getattr(fig, "seq", None), fig_chunk)
    entries.sort(key=lambda e: (e[0], e[1]))
    chunks: list[Chunk] = [chunk for _, _, chunk in entries]
    # Text-layer dollar-completeness recovery (dark flag). Runs AFTER all
    # chunks (blocks + tables + figures) are assembled, only when the flag is on
    # AND the page passes the same fail-closed born-digital trust gate as the
    # native override. ADDS separate grounded chunks for currency amounts the VLM
    # dropped; never modifies the chunks above. Default-off → no-op on prod
    # traffic.
    if (
        not skip_text
        and settings.OCR_TEXTLAYER_COMPLETENESS
        and _page_native_text_trustworthy(page, page_idx=idx)
    ):
        recovered = _recover_textlayer_amounts(chunks, page, idx=idx)
        if recovered:
            chunks.extend(recovered)
            if override_stats is not None and "chunks_textlayer_recovered" in override_stats:
                override_stats["chunks_textlayer_recovered"] += len(recovered)
    return chunks


# ---------------------------------------------------------------------------
# D (plan 028): native-PDF-text override for trustworthy born-digital pages
# ---------------------------------------------------------------------------

# D2 similarity gate: NFKC+casefold token Dice similarity between the clipped
# native text and the VLM transcription must reach this before the native text
# replaces the VLM text. A misaligned bbox, a hallucinated block, or clip bleed
# from a neighboring column all fail the gate and keep the VLM text.
NATIVE_OVERRIDE_MIN_SIMILARITY = 0.80
# Clip tolerance: VLM bboxes run systematically a few points tight (measured on
# the borndigital surface: a trailing word's center sat 7pt outside the box and
# the override dropped it), so the word-center test expands the rect
# HORIZONTALLY by this fraction of its height (capped). Horizontal only: lines
# stack at ~1.2x height pitch, so any vertical slack pulls the neighboring
# line's words in (measured: dates/MRNs bleeding into adjacent header chunks).
NATIVE_OVERRIDE_CLIP_EPSILON_RATIO = 0.75
NATIVE_OVERRIDE_CLIP_EPSILON_MAX_PT = 8.0


def _page_native_text_trustworthy(page: pymupdf.Page, *, page_idx: int) -> bool:
    """D1 trust gate (fail-closed): the page's embedded text layer is real,
    native, uncorrupted text — NOT a prior OCR engine's layer, not gibberish,
    not a scan with incidental text. Any ambiguity → False (no override).

    This is the S0.6 spike gate verbatim (``scripts/spike_borndigital_prevalence.py``),
    promoted to the one production call site; the orphaned signal machinery
    (``_scan_page_text`` and friends) stays the single source of the signals.
    """
    try:
        _, signals = _scan_page_text(page, page_idx=page_idx, collect_text=False)
    except Exception:  # pragma: no cover — pymupdf edge cases fail closed
        return False
    return (
        _has_reliable_native_text(signals)
        and not _has_existing_ocr_text_layer(signals)
        and not _has_corrupt_native_text(signals)
        and _probe_page_text(page)
        and signals.image_area_ratio <= OCR_IMAGE_RATIO_THRESHOLD
    )


def _native_text_in_rect(words: list, rect: pymupdf.Rect) -> str:
    """Native words whose CENTER lies inside ``rect``, reassembled in reading
    order (block / line / word), lines joined with newlines.

    Word-center filtering (not intersection) avoids span bleed: a word from a
    neighboring line or column that merely grazes the clip rect stays out.
    ``words`` is ``page.get_text("words")`` — ``(x0, y0, x1, y1, word, block_no,
    line_no, word_no)`` tuples.
    """
    if rect.is_empty:
        return ""
    eps = min(
        rect.height * NATIVE_OVERRIDE_CLIP_EPSILON_RATIO,
        NATIVE_OVERRIDE_CLIP_EPSILON_MAX_PT,
    )
    rx0, ry0, rx1, ry1 = rect.x0 - eps, rect.y0, rect.x1 + eps, rect.y1
    hits = []
    for x0, y0, x1, y1, word, block_no, line_no, word_no in words:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            hits.append((block_no, line_no, word_no, word))
    if not hits:
        return ""
    hits.sort()
    lines: list[str] = []
    cur_key: tuple | None = None
    cur_words: list[str] = []
    for block_no, line_no, _word_no, word in hits:
        key = (block_no, line_no)
        if key != cur_key and cur_words:
            lines.append(" ".join(cur_words))
            cur_words = []
        cur_key = key
        cur_words.append(word)
    if cur_words:
        lines.append(" ".join(cur_words))
    return "\n".join(lines)


def _override_norm_tokens(text: str) -> list[str]:
    import unicodedata

    return unicodedata.normalize("NFKC", text).casefold().split()


def _native_text_matches_vlm(native_text: str, vlm_text: str) -> bool:
    """D2 similarity gate: token-multiset Dice similarity ≥ 0.80 (NFKC+casefold).
    Symmetric on purpose — both 'native text is a superset (clip bleed)' and
    'native text is a subset (partial layer)' fail the gate.

    Numeric-preservation guard: the override must never DROP a number the VLM
    read — a clip miss that loses a trailing value is strictly worse than any
    digit substitution it could fix. Dropping shrinks the numeric token COUNT
    (correcting a substituted digit keeps it), so the gate requires the native
    text to carry at least as many numeric tokens as the VLM text.
    """
    from collections import Counter

    a = Counter(_override_norm_tokens(native_text))
    b = Counter(_override_norm_tokens(vlm_text))
    total = sum(a.values()) + sum(b.values())
    if total == 0:
        return False
    overlap = sum((a & b).values())
    if (2.0 * overlap / total) < NATIVE_OVERRIDE_MIN_SIMILARITY:
        return False
    n_native_numeric = sum(1 for t in a.elements() if any(ch.isdigit() for ch in t))
    n_vlm_numeric = sum(1 for t in b.elements() if any(ch.isdigit() for ch in t))
    return n_native_numeric >= n_vlm_numeric


# ---------------------------------------------------------------------------
# Text-layer dollar-completeness recovery (born-digital AP/PO documents)
# ---------------------------------------------------------------------------
#
# On dense GL-continuation tables the VLM occasionally drops a line-item dollar
# amount (recall ~0.97). Every dropped amount is present in the PDF text layer.
# After chunks are assembled we cross-check the assembled output's currency
# amounts against the page's text-layer currency amounts and ADD the missing
# ones as separate grounded chunks. Default-off (OCR_TEXTLAYER_COMPLETENESS),
# gated on the same fail-closed born-digital trust gate as the native override.
#
# Precision over recall, phantom-free: only literal cents-shaped currency WORDS
# from the text layer are ever added; percentages / dates / GL-code tails / bare
# integers are excluded; a multiset dedup never duplicates an amount already in
# the output. The matcher is intentionally prod-side and self-contained — no
# import from evals2 (the serving containers don't ship that package).


def _is_currency_token(tok: str) -> int | None:
    """Return signed integer cents iff ``tok`` is a self-contained currency
    amount, else ``None``. This is the canonical "is this a dollar amount?" test;
    it never depends on neighbors (a GL-code tail like ``...70.49.00`` is rejected
    on its own shape, and the percentage/date guards live in the caller).

    A token is a currency amount iff, after stripping ONE optional leading ``$``
    and ONE optional surrounding pair of parens (accounting negatives), the
    remainder is:
      - a comma-grouped or plain integer part of one or more digits, with each
        comma group exactly 3 digits (``1,234`` / ``262`` ok; ``12,34`` / ``1,2``
        no), and
      - EXACTLY two decimal places (``.50``), no more, no fewer.
    Examples that MATCH: ``$1,234.50``, ``262.50``, ``(303.60)``, ``$0.00``.
    Examples that DON'T: ``100`` (no decimals), ``2026-01-04`` (date),
    ``70.49.00`` (two dots), ``001.008.563.2.41`` (GL tail), ``1,2345.00`` (bad
    grouping), ``565165`` (bare integer), ``50%`` (percent — also blocked here
    because ``%`` is not a digit/comma/dot/paren/``$``).
    """
    s = tok
    negative = False
    # One optional surrounding paren pair = accounting negative.
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        negative = True
    # One optional leading currency sign.
    if s.startswith("$"):
        s = s[1:]
    if not s:
        return None
    if "." not in s:
        return None
    int_part, _, frac_part = s.rpartition(".")
    # EXACTLY two decimals, both digits — this is the cents shape. A second dot
    # (``70.49.00``) leaves a dot inside ``int_part``, which the integer check
    # below rejects, so multi-dot GL tails never slip through.
    if len(frac_part) != 2 or not frac_part.isdigit():
        return None
    if not int_part:
        return None
    # Integer part: either plain digits, or comma-grouped with exactly-3-digit
    # groups after the first 1-3 digit group.
    if "," in int_part:
        groups = int_part.split(",")
        if len(groups) < 2:
            return None
        if not (1 <= len(groups[0]) <= 3) or not groups[0].isdigit():
            return None
        for g in groups[1:]:
            if len(g) != 3 or not g.isdigit():
                return None
        digits = "".join(groups)
    else:
        if not int_part.isdigit():
            return None
        digits = int_part
    # Reject a leading-zero integer part (``01.25``/``02.26`` month.year period
    # codes, ``04.20`` period markers): real currency is never written with a
    # leading zero except sub-dollar ``0.NN``. This is the one cents-shaped class
    # on county AP reports that is NOT money.
    if len(digits) > 1 and digits[0] == "0":
        return None
    cents = int(digits) * 100 + int(frac_part)
    return -cents if negative else cents


def _textlayer_currency_amounts(words: list) -> list[tuple]:
    """Currency-amount words on the page, each as ``(cents, x0, y0, x1, y1,
    text)``. ``words`` is ``page.get_text("words")`` →
    ``(x0, y0, x1, y1, word, block_no, line_no, word_no)``.

    Percentage exclusion (geometry): the AP tables render allocation as
    ``100.00 %`` but PDF extraction reorders the ``%`` away from its number, so
    we drop the numeric word immediately to the LEFT of every standalone ``%``
    word on the same text line (y-overlap; the largest x1 that is < the ``%``'s
    x0). Any single ``N%`` token is excluded by ``_is_currency_token`` already
    (``%`` is not a currency char). We compare the recovered word VERBATIM, so an
    amount is never synthesized — only literal text-layer words are returned.
    """
    # Indices (into ``words``) of numeric amounts immediately left of a "%" word.
    excluded_by_pct: set[int] = set()
    for x0, y0, _x1, y1, w, _b, _l, _n in words:
        if w != "%":
            continue
        best_idx: int | None = None
        best_x1 = float("-inf")
        for i, (_wx0, wy0, wx1, wy1, ww, _wb, _wl, _wn) in enumerate(words):
            # same text line = vertical overlap with the "%" word's band
            if not (wy0 < y1 and wy1 > y0):
                continue
            if wx1 >= x0:  # must be to the LEFT of the "%"
                continue
            if _is_currency_token(ww) is None:
                continue
            if wx1 > best_x1:
                best_x1 = wx1
                best_idx = i
        if best_idx is not None:
            excluded_by_pct.add(best_idx)

    out: list[tuple] = []
    for i, (x0, y0, x1, y1, w, _b, _l, _n) in enumerate(words):
        if i in excluded_by_pct:
            continue
        cents = _is_currency_token(w)
        if cents is None:
            continue
        out.append((cents, x0, y0, x1, y1, w))
    return out


def _assembled_currency_multiset(chunks: list[Chunk]):
    """Multiset (Counter keyed on signed cents) of currency amounts already
    present in the assembled chunks. Used to dedup so recovery only ADDS amounts
    the output is genuinely missing.

    A table chunk's ``page_content`` is just a markdown RENDERING of its
    ``cells``, so for a table we count the cells ONLY (the structured truth) —
    counting both would double-count every amount and could suppress a genuine
    recovery. For a text block we count ``page_content``.

    Tokenization mirrors the matcher's notion of a "word": split on whitespace,
    then test each whitespace-delimited token with ``_is_currency_token``. The
    text-layer side splits the same way (``get_text("words")`` is whitespace
    tokenization), so a present amount and its text-layer twin canonicalize to the
    same cents and cancel.
    """
    from collections import Counter

    present: Counter = Counter()

    def _consume(text: str) -> None:
        for tok in text.split():
            cents = _is_currency_token(tok)
            if cents is not None:
                present[cents] += 1

    for ch in chunks:
        if ch.cells:
            # table: cells are the structured truth; page_content is a redundant
            # markdown render of the same cells.
            for cell in ch.cells:
                if cell.text:
                    _consume(cell.text)
        elif ch.page_content:
            _consume(ch.page_content)
    return present


def _recover_textlayer_amounts(
    chunks: list[Chunk],
    page: pymupdf.Page,
    *,
    idx: int,
) -> list[Chunk]:
    """Return NEW grounded amount chunks for text-layer currency amounts that the
    assembled output is missing (multiset-deduped). Pure: never modifies the
    input ``chunks``, table cells, or block text — recovery only ADDS.

    Each recovered chunk = the text-layer word's text + its (x0,y0,x1,y1) bbox via
    ``_text_chunk``, ``page_no=idx+1``, ``confidence=None``.
    """
    words = page.get_text("words")
    layer_amounts = _textlayer_currency_amounts(words)
    if not layer_amounts:
        return []
    present = _assembled_currency_multiset(chunks)
    recovered: list[Chunk] = []
    for cents, x0, y0, x1, y1, text in layer_amounts:
        if present.get(cents, 0) > 0:
            present[cents] -= 1  # consume one; this amount is already in the output
            continue
        bbox = _clip_bbox_to_page([x0, y0, x1, y1], page.rect)
        if bbox is None:  # pragma: no cover — defensive; word bboxes are on-page
            continue
        recovered.append(_text_chunk(text=text, page_no=idx + 1, bbox=bbox, confidence=None))
    return recovered


async def _call_ocr_provider(
    provider,
    image_bytes: bytes,
    *,
    page_width: float,
    page_height: float,
    page_no: int,
    sampling_overrides: dict | None = None,
):
    """Call one provider for one page; ``None`` on failure (the caller falls back).

    ``sampling_overrides`` is passed on the plan-028 retry tiers (A1/A2) and on
    the consistency gate's verification/revote reads (which reuse the accepted
    read's tier — greedy tiers only; stochastic-tier reads are not gate-eligible).
    All of those fire exclusively on the primary Qwen provider (the only provider
    whose ``ocr_page`` takes the kwarg); passing it to a provider without the
    kwarg is a programming error and surfaces as the logged exception below.
    """
    try:
        if sampling_overrides:
            return await provider.ocr_page(
                image_bytes,
                page_width=page_width,
                page_height=page_height,
                sampling_overrides=sampling_overrides,
            )
        return await provider.ocr_page(
            image_bytes,
            page_width=page_width,
            page_height=page_height,
        )
    except Exception as e:
        logger.warning("OCR provider %r failed for page %s: %s", provider.name, page_no, e)
        return None


def _ocr_page_result_has_no_text(page_result) -> bool:
    if page_result is None:
        return True
    if any((block.text or "").strip() for block in page_result.blocks):
        return False
    if any((kv.text or "").strip() for kv in page_result.key_values):
        return False
    for table in page_result.tables:
        if any((cell.text or "").strip() for cell in table.cells):
            return False
    return True


# Scripts the fine-tuned model reads poorly (verified on the launch benchmark:
# Arabic/Hebrew score ~0.1-0.4 vs ~0.9 for the stronger fallback). A page whose
# OWN model output is predominantly one of these is routed to the fallback, which
# handles them near-perfectly. Ranges: Hebrew, Arabic (+ supplement/extended),
# Arabic presentation forms.
_RTL_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0870, 0x089F),  # Arabic Extended-B
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB1D, 0xFDFF),  # Hebrew/Arabic presentation forms A
    (0xFE70, 0xFEFF),  # Arabic presentation forms B
    (0x1EE00, 0x1EEFF),  # Arabic Mathematical Alphabetic Symbols
)
_RTL_FALLBACK_FRACTION = 0.30
_RTL_MIN_LETTERS = 20


def _is_rtl_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _RTL_RANGES)


def _ocr_page_result_is_low_quality_script(page_result) -> bool:
    """True when the model's output is predominantly an RTL script it handles
    poorly — detected from the model's own emitted letters, so it routes the page
    to the fallback even though the (wrong) output looks coherent."""
    if page_result is None:
        return False
    texts: list[str] = [b.text for b in page_result.blocks if b.text]
    texts.extend(kv.text for kv in page_result.key_values if kv.text)
    for table in page_result.tables:
        texts.extend(c.text for c in table.cells if c.text)
    letters = [ch for ch in "".join(texts) if ch.isalpha()]
    if len(letters) < _RTL_MIN_LETTERS:
        return False
    rtl = sum(1 for ch in letters if _is_rtl_char(ch))
    return rtl / len(letters) >= _RTL_FALLBACK_FRACTION


# Trusted-blank guard + ghost floor (v99 empty-page follow-through, 2026-07-20).
# The champion emits [] on blank pages BY DESIGN; the legacy pipeline treated
# that as qwen_empty -> retry -> Gemini fallback, and Gemini hallucinates tiny
# ghost chunks ("The"/"1"/"I") on blank fax pages — re-defeating the fix at the
# API layer. Both mechanisms are ROUTING-ONLY (no output is ever edited) and
# deliberately constant-tuned, not flag-gated (owner 2026-07-20: no feature
# flags); rollback = revert.
#
# Max dark-pixel fraction for "visually blank" (0.4%; calibrated on real
# blank/sparse fax pages: true blanks <= 0.28%, faintest real content >= 0.41%).
_BLANK_INK_MAX_FRACTION = 0.004
# Grayscale level below which a pixel counts as ink.
_BLANK_INK_DARK_LEVEL = 170
# Ghost floor: a qwen_empty page's fallback rescue must yield at least this
# many alphanumeric characters (or a table) to override the model's [].
_FALLBACK_MIN_TEXT_CHARS = 4


def _page_visually_blank(image_bytes: bytes) -> bool:
    """Deterministic blank-page check on the already-rasterized page PNG.

    Downsamples to grayscale and measures the dark-pixel fraction; below
    ``_BLANK_INK_MAX_FRACTION`` the page is visually blank (speckle/fax edge
    marks stay under it; any real text/table/handwriting is far above). Used to
    accept the champion's ``[]`` on blank pages instead of routing them into
    the Gemini fallback (which hallucinates ghost chunks on blank fax pages).
    Fail-closed: any error returns ``False`` so the legacy retry/fallback path
    runs.
    """
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            g = im.convert("L")
            # ~200px wide is plenty to detect ink presence; keeps this ~free.
            if g.width > 200:
                g = g.resize((200, max(1, int(g.height * 200 / g.width))))
            hist = g.histogram()  # 256 bins
        dark = sum(hist[:_BLANK_INK_DARK_LEVEL])
        total = sum(hist)
        if not total:
            return False
        return (dark / total) < _BLANK_INK_MAX_FRACTION
    except Exception:  # noqa: BLE001 — fail-closed to legacy behavior
        return False


def _ocr_unusable_reason(
    page_result, *, ignore_truncation: bool = False
) -> str | None:
    """Why a page is unusable, in priority order, or ``None`` if it is usable.

    Returns a non-``None`` reason exactly when ``_ocr_page_result_is_unusable``
    returns ``True`` — the two share this one function so they can never drift —
    but names *which* failure mode triggered the fallback so a Qwen->Gemini
    fallback can be attributed for metrics and review-bundle curation:

        None result        -> qwen_exception   (provider raised / timed out)
        truncated          -> qwen_truncated   (decode hit the token cap)
        no text            -> qwen_empty
        suspicious         -> qwen_repeated_ngram
        low quality script -> qwen_low_quality_script

    A decode cut off at the token cap is incomplete → fall back rather than ship a
    partial page; a predominantly-RTL page is routed to the stronger fallback.

    """
    if page_result is None:
        return "qwen_exception"
    if page_result.truncated and not ignore_truncation:
        return "qwen_truncated"
    if _ocr_page_result_has_no_text(page_result):
        return "qwen_empty"
    if _ocr_page_result_is_suspicious(page_result):
        return "qwen_repeated_ngram"
    if _ocr_page_result_is_low_quality_script(page_result):
        return "qwen_low_quality_script"
    return None


def _ocr_page_result_is_unusable(page_result) -> bool:
    return _ocr_unusable_reason(page_result) is not None


# Guard-caught reasons whose read is partial-but-REAL and therefore worth
# shipping when the fallback produces nothing. Truncation means the model ran
# out of output budget mid-page: everything it emitted before the cut is
# genuine.
#
# genuine. Deliberately excluded: `qwen_repeated_ngram` and
# `qwen_low_quality_script` (a read caught looping or emitting junk is not
# partial content — and a loop trivially clears any quantity floor precisely
# BECAUSE it repeats, so length is no evidence of quality), and `qwen_empty`
# (nothing to restore).
#
# THE 75-BLOCKS EXHIBIT AND WHY THIS SET IS STILL RIGHT (2026-08-05). The production
# read (the internal measurement records
# §6) found a page where Qwen emitted 75 text blocks, Gemini returned zero chunks, and
# the customer was served an EMPTY page because this exclusion refused the restore.
# The tempting fix — admit `qwen_repeated_ngram` here — was designed, implemented and
# then REJECTED on review as unreachable: `_restore_is_admissible` re-runs the same
# classifier on the same immutable bytes, so a read the classifier just condemned can
# never pass it. Admitting the reason would have been a permission that no live page
# can exercise, and the test that "proved" it worked had to hand-construct a
# reason/result pair the routing path cannot produce.
#
# The exhibit is fixed at its cause instead: the page was flagged by the separator
# rule, which `loop_geometry.has_repeated_content` deletes. A correctly-read dot
# leader is never demoted, so there is nothing to restore. The invariant "a guard that
# fires must never make the page worse than what it guarded" is satisfied by not
# firing, which is the only place it can be satisfied honestly.
_RESTORABLE_GUARD_REASONS = frozenset({"qwen_truncated"})


def _restore_is_admissible(reason: str | None, demoted_result) -> bool:
    """May the primary read ship in place of an empty fallback page?

    Demoted-usable reads (a quality detector demoted an otherwise fine page)
    restore unconditionally — that is the long-standing contract and this
    predicate must not narrow it.

    Guard-caught reads are the class newly eligible, but only the ones in
    ``_RESTORABLE_GUARD_REASONS``. Those that qualify must still clear the shared
    ghost floor, so a page whose only content is a hallucinated figure sentinel is
    never promoted.

    WHY ``qwen_repeated_ngram`` IS NOW RESTORABLE (2026-08-05). The re-check below is
    the canonical classifier, so admitting the reason to the set only matters for a
    read the classifier NO LONGER CONDEMNS. That is exactly the population the old
    separator rule invented: a correctly-read dot leader used to be flagged, and the
    exclusion then destroyed it (one measured page: 75 Qwen text blocks, zero Gemini
    chunks, an EMPTY page served). With that rule deleted, such a page passes the
    re-check and is restored. A genuine repetition loop still fails the re-check and is
    still refused — ``test_a_looping_read_is_not_restored`` pins that, and it is the
    reason the appeal is a re-classification rather than a blanket exemption.
    """
    if demoted_result is None:
        return False
    if reason in _DEMOTED_USABLE_REASONS:
        return True
    if reason not in _RESTORABLE_GUARD_REASONS:
        return False
    # The label alone is not enough. `_ocr_unusable_reason` returns the FIRST
    # guard that fires, and `truncated` is checked first — so a read that is
    # also looping, empty, or junk still comes back labelled `qwen_truncated`.
    # A loop, in particular, usually runs until it exhausts the output budget,
    # which is exactly what truncation is.
    #
    # So re-run the classifier against the read itself with truncation
    # forgiven: truncation is the one rejection that leaves real content
    # behind, and every other guard still disqualifies. Asking the canonical
    # classifier rather than re-listing its predicates here means a guard added
    # later automatically protects this path too.
    # NOT tightened to the strict loop arm (2026-08-05, review). Doing so was drafted
    # and measured: it denies 33 of 154 high-token_f1 truncated reads instead of 29 —
    # four MORE customer pages left empty — which is an independent behaviour change
    # that has no business riding inside a false-positive removal. If the restore floor
    # should be stricter, that needs its own evidence and its own PR.
    if _ocr_unusable_reason(demoted_result, ignore_truncation=True) is not None:
        return False
    return _ocr_result_substance_chars(demoted_result) >= _FALLBACK_MIN_TEXT_CHARS


def _gemini_fallback_reason(page_result) -> str | None:
    """Why the Gemini fallback page is itself unusable, or ``None`` if it is fine.

    Used only to tag the review bundle when the fallback path failed too; never
    affects routing. For guard-caught pages the shipped page is the Gemini result
    regardless WHEN IT PRODUCES CHUNKS; when the fallback yields nothing
    servable the original read ships instead, via the chunk-level floor in
    ``_one`` (counted in pages_fallback_restored). That floor covers
    DEMOTED-usable pages (qwen_inconsistent / qwen_low_coverage) plus the
    guard-caught reasons in ``_RESTORABLE_GUARD_REASONS``.
    """
    if page_result is None:
        return "gemini_exception"
    if _ocr_page_result_has_no_text(page_result):
        return "gemini_empty"
    return None


def _ocr_result_substance_chars(page_result) -> int:
    """Alphanumeric-character count of an OCR page result (ghost-floor input).

    Blocks + key_values contribute their alphanumeric text length; any table
    counts as unconditionally substantive (a table rescue is never a ghost).
    Figure sentinels (``<image>``) contribute nothing — a lone hallucinated
    figure chunk on a blank page is exactly the ghost class being floored.
    """
    if page_result is None:
        return 0
    n = 0
    for b in page_result.blocks:
        n += sum(ch.isalnum() for ch in (b.text or ""))
    for kv in page_result.key_values:
        n += sum(ch.isalnum() for ch in (kv.text or ""))
    if page_result.tables:
        n += 10_000
    return n


def _ocr_result_chunk_counts(page_result) -> dict[str, int]:
    """Low-cardinality {text,table,image,key_value} counts for one OCR page result.
    Counts only — no page content — so it is safe in the review-bundle manifest."""
    if page_result is None:
        return {"text": 0, "table": 0, "image": 0, "key_value": 0}
    return {
        "text": sum(1 for b in page_result.blocks if (b.text or "").strip()),
        "table": len(page_result.tables),
        "image": len(page_result.figures),
        "key_value": sum(1 for kv in page_result.key_values if (kv.text or "").strip()),
    }


def _ocr_page_result_is_suspicious(page_result) -> bool:
    """Is this read junk? LIVE PREDICATE: ``core.loop_geometry`` (v2, experiment 130).

    Verified offline against every surface we own before the flip (owner decision
    2026-08-05, no shadow phase):

      * 37,726 high-token_f1 EOS-clean reads across 54 eval surfaces — v1 discards
        716 of them, v2 discards **0**.
      * The 166 retained PRODUCTION fallback bundles — of the 44 pages production
        routed as ``qwen_repeated_ngram``, v2 fires on **0**; all 5 genuine
        degenerations (7,448-8,158 pipe runs, plus one 410-blank-row page) are still
        caught; both 16-pipe wide-grid false positives are dropped.
      * 43,535 correct pages swept for every legitimate alphanumeric-repetition class
        (Yes/No columns, answer grids, CJK option labels, schedule rows) — 0 fires.

    ``_suspicious_limbs`` still computes v1 alongside and the counters record the
    disagreement, so the flip is confirmable from production telemetry rather than
    asserted.
    """
    _v1_word, _v1_cell, v2_word, v2_cell = _suspicious_limbs(page_result)
    return v2_word or v2_cell


def _suspicious_limbs(page_result) -> tuple[bool, bool, bool, bool]:
    """``(v1_word, v1_cell, v2_word, v2_cell)`` — both predicates, one pass.

    SR-1 needs the v1 split to answer the question it was commissioned for ("does the
    same-line detector route anything on the PHI lane?"), and the v2 split to size the
    change before making it. Recording only v2 would measure the candidate and leave
    the commissioned question unanswered — the defect this signature exists to avoid.

    v1 is byte-identical to the shipped predicate: ``_ocr_page_result_is_suspicious``
    is exactly ``v1_word or v1_cell``.
    """
    if page_result is None:
        return (False, False, False, False)
    texts: list[str] = []
    texts.extend(block.text for block in page_result.blocks if block.text)
    texts.extend(kv.text for kv in page_result.key_values if kv.text)
    for table in page_result.tables:
        texts.extend(cell.text for cell in table.cells if cell.text)
    v1_word = any(_has_repeated_ngram(text) for text in texts)
    v2_word = any(loop_geometry.has_repeated_content(text) for text in texts)
    # The degenerate empty-cell loop never survives into ``texts``: it lives in a
    # single unterminated ``text_content`` that ``parse_bbox_2d_json`` drops, so
    # the parsed blocks/cells above can't carry it. Catch it on the RAW model
    # output, which the provider attaches to ``page_result.raw`` best-effort.
    raw = _extract_raw_text(page_result.raw) if page_result.raw is not None else ""
    v1_cell = _has_degenerate_empty_cell_loop(raw)
    v2_cell = loop_geometry.is_degenerate_loop(raw)
    return (v1_word, v1_cell, v2_word, v2_cell)


def _has_degenerate_empty_cell_loop(raw: str) -> bool:
    """FROZEN v1 predicate — see the constant block at the top of this module.

    Kept resolvable ONLY for the off-line contract that pinned serving parity against
    it (plan-114's reward floor imports this exact callable, and its contract sha
    covers these bytes). It is on NO serving path: the live predicate is
    ``loop_geometry.is_degenerate_loop``. Do not add a caller.

    It keyed on the count of consecutive PIPES in a single-line run — a quantity that
    measures how WIDE one table row is, not whether the decode is healthy.
    """
    if not raw:
        return False
    n = len(raw)
    # The final ~1% of the output (>= 3 chars) counted as "terminal". Asserted, never
    # fitted — one of the numbers experiment 130 removed rather than re-derived.
    eof_window = max(3, n // 100)
    for m in _EMPTY_CELL_RUN_RE.finditer(raw):
        cells = m.group(0).count("|")
        if cells >= EMPTY_CELL_LOOP_ABSOLUTE_RUN:
            return True
        if cells >= EMPTY_CELL_LOOP_TERMINAL_RUN and (n - m.end()) <= eof_window:
            return True
    return False


def _has_repeated_ngram(text: str) -> bool:
    """FROZEN v1 word limb — superseded by ``loop_geometry.has_repeated_content``.

    Kept resolvable for off-line contracts and for the A/B receipt; on NO serving
    path. Its thresholds (separator 40, single-punctuation 24, word 12, window 6, plus
    the ``{-, _, .}`` whitelist) all landed uncommented on 2026-05-07 in two commits
    that documented other changes, and production's own read attributes 40.7-45.5% of
    the Gemini fallback bill to the separator rule firing on correctly-read dot
    leaders.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    if not tokens:
        return False

    if _has_consecutive_repeated_windows(tokens, n=3, threshold=DOTS_SUSPICIOUS_REPEAT_THRESHOLD):
        return True
    if _has_consecutive_repeated_windows(tokens, n=2, threshold=DOTS_SUSPICIOUS_REPEAT_THRESHOLD):
        return True

    run_token = tokens[0]
    run_len = 1
    for token in tokens[1:]:
        if token == run_token:
            run_len += 1
            if token in DOTS_COMMON_SEPARATOR_TOKENS:
                threshold = DOTS_SEPARATOR_REPEAT_THRESHOLD
            elif len(token) == 1 and not token.isalnum():
                threshold = 24
            else:
                threshold = 12
            if run_len >= threshold:
                return True
        else:
            run_token = token
            run_len = 1
    return False


def _has_consecutive_repeated_windows(
    tokens: list[str],
    *,
    n: int,
    threshold: int,
) -> bool:
    if len(tokens) < n * threshold:
        return False
    repeats = 1
    previous = tuple(tokens[0:n])
    for start in range(n, len(tokens) - n + 1, n):
        current = tuple(tokens[start : start + n])
        if current == previous:
            repeats += 1
            repeat_threshold = (
                DOTS_SEPARATOR_REPEAT_THRESHOLD
                if _is_common_separator_window(current)
                else threshold
            )
            if repeats >= repeat_threshold:
                return True
        else:
            previous = current
            repeats = 1
    return False


def _is_common_separator_window(tokens: tuple[str, ...]) -> bool:
    return bool(tokens) and all(token in DOTS_COMMON_SEPARATOR_TOKENS for token in tokens)


def _ocr_raster_zoom(
    page: pymupdf.Page,
    *,
    dpi: int = OCR_DEFAULT_DPI,
    max_pixels: int = OCR_MAX_IMAGE_PIXELS,
) -> float:
    zoom = dpi / 72
    if max_pixels <= 0:
        return zoom

    width_pt = max(float(page.rect.width), 1.0)
    height_pt = max(float(page.rect.height), 1.0)
    for _ in range(4):
        width_px = max(1, math.ceil(width_pt * zoom))
        height_px = max(1, math.ceil(height_pt * zoom))
        pixels = width_px * height_px
        if pixels <= max_pixels:
            return zoom
        zoom *= math.sqrt(max_pixels / pixels) * 0.999
    return zoom


def _rasterize_page_for_ocr(
    page: pymupdf.Page,
    *,
    dpi: int = OCR_DEFAULT_DPI,
    max_pixels: int = OCR_MAX_IMAGE_PIXELS,
) -> bytes:
    zoom = _ocr_raster_zoom(page, dpi=dpi, max_pixels=max_pixels)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        alpha=False,
        colorspace=pymupdf.csRGB,
    )
    return pix.tobytes(output="png")


def _figure_chunks_from_raster(
    *,
    image_bytes: bytes,
    figures: list,
    page_no: int,
    page_rect: pymupdf.Rect,
) -> list[tuple[object, Chunk]]:
    """Crop figure regions out of a rasterized page PNG and emit image chunks.

    Returns ``(source_figure, chunk)`` pairs so the caller can place each chunk
    at its figure's emission-order position (2026-07-30 reading-order fix).

    Page bboxes are in PDF user-space points (matching ``_rasterize_page_for_ocr``'s
    1× pixmap, which maps points → pixels 1:1 for born-digital pages). For very
    small or degenerate regions we skip emission rather than embed empty bytes.
    """
    import base64
    import io as _io

    from PIL import Image as _Image

    out: list[tuple[object, Chunk]] = []
    if not figures:
        return out
    try:
        page_img = _Image.open(_io.BytesIO(image_bytes))
    except Exception:
        return out
    pw_pts, ph_pts = page_rect.width, page_rect.height
    if pw_pts <= 0 or ph_pts <= 0:
        return out
    img_w, img_h = page_img.size
    sx = img_w / pw_pts
    sy = img_h / ph_pts
    for fig in figures:
        x0, y0, x1, y1 = fig.bbox
        # Clamp + skip degenerate regions.
        px0, py0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
        px1, py1 = min(img_w, int(x1 * sx)), min(img_h, int(y1 * sy))
        if px1 - px0 < 8 or py1 - py0 < 8:
            continue
        crop = page_img.crop((px0, py0, px1, py1))
        buf = _io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append((
            fig,
            _image_chunk(
                page_no=page_no,
                bbox=[x0, y0, x1, y1],
                image_url=None,
                image_b64=b64,
                image_mime="image/png",
                image_width=crop.width,
                image_height=crop.height,
            ),
        ))
    return out
