"""Public data model for the extraction pipeline.

Text chunks are PyMuPDF spans (contiguous runs of glyphs with the same
font and size); image chunks are figures extracted from the document.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer, model_validator


class InputKind(str, Enum):
    PDF = "pdf"
    PPTX = "pptx"
    DOCX = "docx"
    # Internal routing state only — callers never send a kind. Raster images
    # (PNG/JPEG/WebP/TIFF/HEIC/BMP) are converted to PDF before extraction.
    IMAGE = "image"


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    # A key-value form region (an attribute panel / boxed field group). ``page_content``
    # carries the pinned line grammar (``Key: Value`` per line, ``Key: <empty>`` for a
    # blank field, ``[x]``/``[ ]`` for checkbox options, an optional leading title line);
    # ``bbox`` is the region box. Per-pair boxes are a v2 upgrade — v1 is region-level.
    # See .claude/skills/data-curation/references/kv-region-contract.md.
    KEY_VALUE = "kv"


class TableCell(BaseModel):
    """One cell of a table chunk. Indices are 0-based."""

    text: str = ""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    bbox: list[float] | None = None
    confidence: float | None = None
    page_no: int | None = Field(
        default=None,
        description=(
            "Source page of this cell when the table was assembled across pages "
            "(see Chunk.merged_from_pages). None for single-page tables."
        ),
    )


class Chunk(BaseModel):
    """A single extracted element: either a text span or an image figure."""

    page_content: str = ""
    page_no: int
    bbox: list[float] | None = Field(
        default=None, description="[x0, y0, x1, y1] in PDF user-space points."
    )
    chunk_type: ChunkType = ChunkType.TEXT
    # Set by OCR providers; None for native text.
    confidence: float | None = Field(
        default=None,
        description=(
            "OCR confidence for this chunk, 0-1, scored on the weakest "
            "character of the recognized text. null for chunks read from the "
            "document's native text layer, where recognition is not a factor."
        ),
    )

    # Image chunks only. Exactly one of image_url / image_b64 is populated,
    # depending on the configured storage backend.
    image_url: str | None = None
    image_b64: str | None = None
    image_mime: str | None = None
    image_width: int | None = None
    image_height: int | None = None

    # Table chunks only. ``page_content`` carries a markdown rendering so
    # plain-text consumers still see readable output; ``cells`` is the
    # structured representation.
    cells: list[TableCell] | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    table_output_format: Literal["markdown", "html"] | None = Field(
        default=None,
        description=(
            "Table chunks only: the format this chunk was delivered in. "
            "page_content stays a markdown compatibility render in every mode."
        ),
    )
    merged_from_pages: list[int] | None = Field(
        default=None,
        description=(
            "Pages this chunk was assembled from when document-level assembly "
            "merged a table spanning a page boundary (1-based, ascending). "
            "page_no/bbox refer to the first fragment; each cell's page_no "
            "carries its own source page. None for unmerged chunks."
        ),
    )



# --------------------------------------------------------------------------- #
# RAG chunking (plan 077 — opt-in ``chunking="semantic"``)
# --------------------------------------------------------------------------- #
class TablePartRef(BaseModel):
    """Provenance of a synthesized split-table part inside a segment.

    ``row_start``/``row_end`` are 0-based, inclusive BODY-row coordinates of
    the source table (header rows excluded); ``index`` is the 0-based part
    number out of ``count`` parts. Concatenating all parts' body rows in
    ``index`` order reproduces the source table's body rows exactly once.
    """

    index: int
    count: int
    row_start: int
    row_end: int


class TextPartRef(BaseModel):
    """Provenance of a synthesized split-text part inside a segment.

    A text element bigger than the segment size band splits at line breaks;
    ``char_start``/``char_end`` are half-open character offsets into the source
    chunk's ``page_content`` (``page_content[char_start:char_end]`` is this
    part's text — no text is duplicated on the member). Parts share the source
    chunk's bbox (the honest fidelity floor — OCR gives one box per element).
    Consecutive parts satisfy ``next.char_start == prev.char_end + 1``: the
    slices cover the source text exactly, in ``index`` order, minus the single
    newline separator between adjacent parts.
    """

    index: int
    count: int
    char_start: int
    char_end: int


class SegmentMember(BaseModel):
    """One element of a segment: a grounding record pointing back at the flat
    ``chunks`` list.

    ``source_index`` is the element's index in ``ExtractResponse.chunks`` —
    full payloads (text, structured cells, image data) live there and are
    deliberately not duplicated here. The exception is a synthesized
    split-table part (``table_part`` set): it exists nowhere else, so it
    carries its own re-rendered ``page_content``/``cells``/``n_rows``/
    ``n_cols``; its ``source_index`` points at the original, unsplit table.
    """

    source_index: int
    chunk_type: ChunkType
    page_no: int
    bbox: list[float] | None = Field(
        default=None, description="[x0, y0, x1, y1] in PDF user-space points."
    )
    table_part: TablePartRef | None = None
    text_part: TextPartRef | None = None
    page_content: str | None = None
    cells: list[TableCell] | None = None
    n_rows: int | None = None
    n_cols: int | None = None


class Segment(BaseModel):
    """A size-banded, embed-ready group of elements (``chunking="semantic"``).

    ``content`` is the segment's markdown text (member contributions joined
    with blank lines; tables as GFM, figures as a placeholder/URL reference —
    never inline image data). ``char_count == len(content)``. ``chunks`` are
    the member grounding records in reading order; bounding boxes live on
    members (a segment can span pages and columns, so it has no single box).
    """

    content: str
    char_count: int
    pages: list[int] = Field(description="1-based pages this segment spans, ascending.")
    chunks: list[SegmentMember]


class PageDimensions(BaseModel):
    """Width/height of one source page in PDF user-space points — the same
    coordinate space as every ``bbox`` in the response, so clients can
    normalize boxes without reopening the source document."""

    page_no: int
    width: float
    height: float


class ExtractRequest(BaseModel):
    """Input to a parse request.

    Server-side limits (page count, file size) are not user-configurable.
    Unknown fields are accepted and ignored for backward compatibility.
    The ``ocr`` knob is deprecated and currently has no effect.
    """

    # NOTE: this docstring is exported as the public OpenAPI schema description.
    # Keep it contract-level; internal guardrails (image/OCR thresholds, the
    # legacy ``granularity`` field) are deliberately not named here.

    url: str | None = Field(
        default=None,
        description=(
            "HTTP(S) URL of the document to parse (URL route only). Input "
            "type is detected from the file itself; the extension is a hint."
        ),
    )
    extract_text: bool = Field(
        default=True,
        description=(
            "Include text chunks in the response. Set false to skip text "
            "spans; table chunks (and figures, when extract_images is true) "
            "are still returned."
        ),
    )
    extract_images: bool = Field(
        default=True,
        description=(
            "Include figure (image) chunks in the response. Set false to skip "
            "figure extraction; text and table chunks are still returned."
        ),
    )
    # Deprecated / no-op, accepted for backward compatibility.
    ocr: Literal["auto", "never", "force"] = Field(
        default="auto",
        description="Deprecated. Accepted for backward compatibility but currently has no effect.",
    )
    # Plan 058: opt-in cell-level table structure. "markdown" (default) is
    table_output_format: Literal["markdown", "html"] = Field(
        default="markdown",
        description=(
            "How table chunks are serialized in page_content: GitHub-flavored "
            "markdown (default) or an HTML table. Echoed on each table chunk."
        ),
    )
    # Plan 077: opt-in RAG chunking. "none" leaves the response byte-identical
    # to prior behavior; "semantic" additionally returns `segments`.
    chunking: Literal["none", "semantic"] = Field(
        default="none",
        description=(
            "'none' (default): response unchanged. 'semantic': additionally "
            "return `segments` — elements grouped toward chunk_size characters "
            "at semantic/structural boundaries (headings, gaps, page breaks), "
            "each segment carrying embed-ready markdown content plus member "
            "records with page numbers and bounding boxes."
        ),
    )
    chunk_size: int = Field(
        default=1000,
        description=(
            "Target segment size in characters of segment content. Segments "
            "land in a +/-25% band around this target (default 750-1250). "
            "Only meaningful when chunking is enabled; validated (200-8000) "
            "only in that case and ignored otherwise."
        ),
    )
    # plan-088 SHIPPING contract. Opt-in, DEFAULT OFF (owner decision): billed
    # +0.5 credits/page on top of the parse, so it is never on unless asked for.
    # Opt-in whole-document content. "false" leaves the response unchanged;
    # "true" additionally returns `content` — the entire parsed document as one
    # text string, in reading order. Off by default because it roughly doubles
    # the response size (it restates every chunk's text).
    include_content: bool = Field(
        default=False,
        description=(
            "false (default): response unchanged. true: additionally return "
            "`content` — the entire parsed document as a single text string, "
            "concatenated in reading order. Not reformatted as structured "
            "markdown; it is the document's own text content end to end."
        ),
    )

    @model_validator(mode="after")
    def _validate_chunking(self) -> "ExtractRequest":
        # chunk_size bounds apply only when chunking is enabled: this field was
        # previously an ignored unknown field, so a disabled request must never
        # start failing validation on it (plan 077 back-compat rule).
        if self.chunking != "none":
            if not (200 <= self.chunk_size <= 8000):
                raise ValueError(
                    "chunk_size must be between 200 and 8000 characters when chunking is enabled"
                )
        elif not (200 <= self.chunk_size <= 8000):
            self.chunk_size = 1000
        return self


class BillingUsage(BaseModel):
    """What this request charged, in credits (plan 078 D6).

    Present only on responses whose request was actually charged — absent
    (never null) on unbilled lanes (demo, legacy-PHI ledger) so pre-credits
    response shapes stay byte-identical. ``credits = pages × credits_per_page``
    at the v1 card: parse 1.0, schema extract 4.0 all-in.
    """

    # Source document pages (pre-multiplier — what the customer thinks of as
    # the document's size).
    pages: int
    # Total credits deducted from the balance for this request.
    credits: float
    # The applied rate. float so future fractional surcharges need no shape
    # change.
    credits_per_page: float


class ExtractResponse(BaseModel):
    chunks: list[Chunk]
    # Plan 077: populated only when the request set chunking != "none". Both
    # are omitted from serialized output when unset (see _serialize below) so
    # default responses stay byte-identical to pre-chunking output.
    segments: list[Segment] | None = None
    page_dimensions: list[PageDimensions] | None = None
    # Populated only when the request set include_content=true. Absent (not
    # null) otherwise, same rule as the chunking fields.
    content: str | None = None
    # Plan 078: set by the route AFTER a successful charge; absent when the
    # request wasn't billed (demo lane, legacy-PHI ledger, billing outage).
    usage: BillingUsage | None = None
    # Set by the API/worker after uploading the LibreOffice-converted PDF
    # for DOCX/PPTX. Absent (not null) on native PDFs, images, library/CLI
    # callers, and when the upload fails (fail-open).
    pdf_rendition_url: str | None = Field(
        default=None,
        description=(
            "Expiring URL of the PDF we paginated this document onto. Present "
            "only for DOCX and PPTX: chunk bboxes are in this PDF's point "
            "space, so overlay them here rather than on a Word/PowerPoint "
            "render of the original. Download the file onto your own storage "
            "before the link expires. Omitted for PDFs and images."
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> Any:
        """Single public serialization home (plan 077): every surface that
        dumps this model (sync routes, batch worker, CLI) inherits the same
        rule — chunking fields are absent, not null, when chunking is off.
        Same rule for ``usage`` (plan 078) on unbilled responses and for
        ``pdf_rendition_url`` when the input was not an office document."""
        data = handler(self)
        if isinstance(data, dict):
            if self.segments is None:
                data.pop("segments", None)
                data.pop("page_dimensions", None)
            if self.content is None:
                data.pop("content", None)
            if self.usage is None:
                data.pop("usage", None)
            if self.pdf_rendition_url is None:
                data.pop("pdf_rendition_url", None)
        return data

    # Internal: the extractor records the source document's page count here
    # so callers can meter usage.
    # Not part of the public schema — excluded from JSON and OpenAPI.
    _page_count: int = PrivateAttr(default=0)
    # Internal: per-page ``(width, height)`` in PDF points, 1-based by list
    # position (page N → ``page_sizes[N-1]``). Captured during parse while the
    # document is still open. Used by the schema-extraction endpoint to
    # re-normalize chunk bboxes (absolute points) to 0–1000 page-relative.
    # Not part of the public schema — excluded from JSON and OpenAPI.
    _page_sizes: list[tuple[float, float]] = PrivateAttr(default_factory=list)
    # Internal: LibreOffice-converted PDF bytes for DOCX/PPTX, stashed so
    # the API/worker can upload a rendition the chunk bboxes refer to.
    # Dropped after publish (and never serialized).
    _rendition_pdf: bytes | None = PrivateAttr(default=None)

    @property
    def page_count(self) -> int:
        return self._page_count

    @property
    def page_sizes(self) -> list[tuple[float, float]]:
        return self._page_sizes


# ---------------------------------------------------------------------------
# Schema extraction (plan 050 — ``POST /v1/extract/schema``)
# ---------------------------------------------------------------------------
class SchemaExtractRequest(BaseModel):
    """Input to a schema-extraction request.

    Given a document and an arbitrary user JSON schema, fill the schema's
    fields from the document and cite where each value came from.
    """

    # NOTE: this docstring is exported as the public OpenAPI schema description;
    # keep it contract-level. Implementation (kept out of the public spec): the
    # model only ever emits chunk indices + a verbatim quote; the backend remaps
    # index -> {page, bbox, text} and verifies the quote, so a value can be
    # grounded to a real box without the model emitting (or hallucinating)
    # coordinates.

    model_config = ConfigDict(populate_by_name=True)

    # PHI keys: 400 (file-upload only), mirrors the OCR ``/v1/extract`` URL gate.
    url: str | None = Field(
        default=None,
        description=(
            "HTTP(S) URL of the document to extract from. PHI-enabled keys "
            "must use the file-upload route instead."
        ),
    )
    # User JSON schema. Aliased so the wire field is ``schema`` while the Python
    # attribute avoids shadowing pydantic's ``.schema``. Optional only when
    # ``auto_schema`` is set (the schema is then designed from the document);
    # otherwise required (a missing schema → 422).
    schema_: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description=(
            "JSON Schema describing the fields to extract; field descriptions "
            "are instructions the extractor follows. Required unless "
            "auto_schema is true."
        ),
    )
    # Hallucination policy. True (default) nulls out any non-null value whose
    # grounding quote can't be verified AND reports its path in
    # ``ungrounded_fields``; False keeps the value but still flags it.
    # Either way, a leaf the SCHEMA makes unquotable — an enum option, a
    # normalized number/date, a zero default — is exempt from the gate and
    # reported separately (schema_extract._unquotable_exemption): nulling those
    # destroyed 577 correct cells per 30 real fabrications caught (EXP-A).
    strict: bool = Field(
        default=True,
        description=(
            "What happens to a value whose citation cannot be verified "
            "against the document. true (default): the value is nulled out "
            "and its path listed in ungrounded_fields, so a fabricated value "
            "never reaches you. false: the value is kept but still flagged "
            "in ungrounded_fields."
        ),
    )
    # ✨ "Generate the schema from the document first" (plan 050 M4): design a flat
    # schema from the document, then run the normal grounded extract against it.
    # The schema used is echoed back in ``generated_schema``.
    auto_schema: bool = Field(
        default=False,
        description=(
            "Set true (and omit schema) to have a schema designed from the "
            "document first, then filled with the same grounded extraction. "
            "The schema used is returned in generated_schema."
        ),
    )
    extract_images: bool = Field(
        default=True,
        description=(
            "Include figures from the parse stage in the extraction context. "
            "Set false to extract from text and tables only."
        ),
    )
    # Opt-in whole-document text, exactly like ExtractRequest.include_content on
    # the parse route: "false" leaves the response byte-identical, "true" adds
    # `ocr_text`. Off by default because it roughly doubles the response (it
    # restates every chunk's text next to the fields). Accounts on the receipt
    # profile get it without asking — see EXTRACT_OCR_TEXT_CUSTOMERS.
    include_ocr_text: bool = Field(
        default=False,
        description=(
            "false (default): response unchanged. true: additionally return "
            "`ocr_text` — the whole parsed document as a single text string, "
            "concatenated in reading order (the same text POST /v1/parse "
            "returns as `content`)."
        ),
    )


class SchemaExtractChunksRequest(BaseModel):
    """Internal-only input for the playground/demo BFF (plan 050 M4): extract from
    chunks the caller ALREADY parsed, skipping the parse stage entirely.

    The BFF holds the parse output (it rendered the bbox overlay from it), so it
    sends those chunks straight to extraction — no re-parse, no double billing.
    Evidence ``text`` (the verified quote) is what the BFF re-maps back to its
    own chunk for the overlay, so bbox normalization is irrelevant here.
    """

    model_config = ConfigDict(populate_by_name=True)

    # [{page_content, page_no, bbox}] — the public parse-chunk shape.
    chunks: list[dict[str, Any]]
    # Optional per-page [width, height] in the same units as the chunk bboxes
    # (index = page_no - 1). Send it so the layout recipe gets (x%,y%) chunk
    # coordinates and evidence bboxes come back normalized — EXACT parity with
    # the full-pipeline routes, which always have page geometry. Omitted → the
    # layout recipe still runs but without coordinate prefixes.
    page_sizes: list[tuple[float, float]] | None = None
    # Optional: omit (or send null) to have the schema designed from the chunks
    # first (the playground's no-schema "auto-design" path → echoed back in
    # ``generated_schema``).
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool = False  # playground shows everything extracted; caller decides
    # Bill this extraction (plan 028 F3). The dashboard BFF sets this so a
    # signed-in playground schema run over held chunks is charged the extract
    # PREMIUM — the parse it came from was already billed 1×, so the premium
    # (EXTRACT_PAGE_BILLING_MULTIPLIER − 1) tops it up to the 4× all-in rate
    # instead of being free. The anonymous demo lane omits it → stays unbilled.
    bill: bool = False


class FieldEvidence(BaseModel):
    """One grounding citation for an extracted value."""

    page: int
    # [x0, y0, x1, y1], 0–1000 normalized, page-relative, top-left origin.
    # None when the source chunk carried no bbox (e.g. converted office docs).
    bbox: list[float] | None = None
    # Verbatim source span (the model's grounding quote, verified to be a real
    # substring of the cited chunk).
    text: str
    # Parse-side confidence of the cited chunk (``exp(min value-token logprob)``,
    # see ``ocr.chunk_confidence``). None for native/no-logprob chunks. The
    # low-confidence re-read gate (below) reads the MIN of these across a field's
    # evidence.
    confidence: float | None = None
    # Set by the tier-2 low-confidence re-read (``ocr.reread``) on high-stakes
    # fields (member IDs) whose confidence fell below the gate and whose isolated
    # crop re-read disagreed with the parse value: the field is surfaced for human
    # review rather than silently trusted, with the re-read shown as a suggestion.
    needs_review: bool = False
    suggested_value: str | None = None


class SchemaExtractResponse(BaseModel):
    # Schema-shaped, clean values; null where the document lacks the field.
    values: dict[str, Any]
    # Per-field citations keyed by field path, e.g. ``"line_items[3].amount"``.
    evidence: dict[str, list[FieldEvidence]]
    # Non-null values whose quote failed grounding (suspected hallucination).
    # Suppressed for phi_safe requests.
    ungrounded_fields: list[str] = Field(default_factory=list)
    page_count: int
    # The schema designed from the document, present only when ``auto_schema`` ran
    # (so the caller can see/reuse the fields it filled). None for a supplied schema.
    generated_schema: dict[str, Any] | None = None
    # Plan 078: set by the route AFTER a successful charge; absent when the
    # request wasn't billed (demo overlay, billing declined before serialize).
    # Unlike ExtractResponse this model has NO wrap serializer (one would
    # collapse its OpenAPI schema to {}), so routes strip a null ``usage``
    # via :meth:`dump_public` to keep unbilled shapes byte-stable.
    usage: BillingUsage | None = None
    # The whole document as text, for the callers that asked for the parse and
    # the fields in one call — ``include_ocr_text`` on the request, or an account
    # provisioned for it (EXTRACT_OCR_TEXT_CUSTOMERS / the receipt profile). Same
    # string /v1/parse returns as ``content``. None — hence ABSENT, see
    # dump_public — otherwise, so no existing response shape gains a key.
    ocr_text: str | None = None

    def dump_public(self) -> dict[str, Any]:
        """The public JSON shape: ``usage`` and ``ocr_text`` absent, not null,
        when they do not apply."""
        data = self.model_dump(mode="json")
        if data.get("usage") is None:
            data.pop("usage", None)
        if data.get("ocr_text") is None:
            data.pop("ocr_text", None)
        return data
