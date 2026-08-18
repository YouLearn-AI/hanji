"""One Textract read per page, shared by every consumer in a request.

``DetectDocumentText`` returns LINE blocks and WORD blocks from the SAME response, and
the two Textract consumers in the schema-extract path want one view each:

* :mod:`extract.core.xref_context` wants the LINE text, to append as secondary context
  before extraction.
* :mod:`extract.core.textract_tighten` wants the per-WORD boxes, to shrink citation boxes
  after extraction.

Before this module they each made their own call, so any page that was both cross-read and
cited was sent to Textract TWICE — double cost, double latency, identical bytes. A
request-scoped cache keyed on the image content collapses that to one call, and the two
consumers just take different fields off it.

Keyed on a hash of the image bytes rather than a page number so it is correct even when
callers render pages independently or at different scales.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from extract.config import settings

log = logging.getLogger(__name__)


#: A WORD below this Textract confidence (0-100) is treated as a shaky read. Textract
#: sits in the high 90s on clean print, so this only catches genuinely degraded glyphs.
LOW_CONFIDENCE = 90.0

#: Textract's synchronous `detect_document_text` byte ceiling is 10MB; stay under
#: it with room for the multipart framing around the payload.
TEXTRACT_SYNC_LIMIT_BYTES = 9_000_000


@dataclass
class PageRead:
    """One Textract page response, in both the views its consumers need."""

    #: LINE blocks joined in reading order — what the extractor sees as secondary context.
    text: str = ""
    #: (text, left, top, width, height) in PAGE PIXELS — what the box tightener consumes.
    words: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    error: str | None = None
    #: WORD blocks Textract typed as handwriting rather than print.
    handwriting_words: int = 0
    #: WORD blocks Textract read below :data:`LOW_CONFIDENCE`.
    low_confidence_words: int = 0
    #: Every WORD block — the denominator for the two counts above.
    total_words: int = 0


class TextractPageCache:
    """Request-scoped store of page reads. Not thread-safe by design — a request's pages
    are read inside one thread pool whose futures are joined before the next stage."""

    def __init__(self, *, transcode_oversize: bool = False) -> None:
        self._by_hash: dict[str, PageRead] = {}
        self.calls = 0
        self.hits = 0
        #: Whether a payload over Textract's synchronous limit is transcoded to
        #: JPEG instead of being allowed to fail open. OFF by default because
        #: turning it on is a BEHAVIOUR CHANGE, not a repair: a page that used to
        #: come back witness-less now comes back with a witness, which moves the
        #: chunks the cross-read emits, moves the boxes the tightener produces,
        #: and bills a Textract call that previously failed. Those are all
        #: improvements — and all of them are things a customer who never asked
        #: for them would notice. The receipt lane, whose sealed stack was
        #: measured WITH the transcode, passes True per request.
        self.transcode_oversize = transcode_oversize

    def read(self, image: bytes, size: tuple[int, int]) -> PageRead:
        """Return the page's Textract read, calling the API at most once per image.

        ``size`` is (width, height) in pixels; Textract returns 0-1 page fractions, and the
        tightener needs page-pixel boxes.
        """
        key = hashlib.sha256(image).hexdigest()
        cached = self._by_hash.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        out = self._call(image, size)
        self._by_hash[key] = out
        return out

    def _call(self, image: bytes, size: tuple[int, int]) -> PageRead:
        try:
            import boto3

            # Textract's synchronous byte limit is 10MB; a large page rendered
            # to PNG can exceed it (a 2400x3200 receipt render is 19MB) and the
            # call then fails open — silently costing that page its witness.
            # Transcode oversized payloads to JPEG: same pixels, same read.
            # Per-request (see __init__): this changes what a page returns, so it
            # is the lane's lever rather than a deployment-wide repair.
            if self.transcode_oversize and len(image) > TEXTRACT_SYNC_LIMIT_BYTES:
                import io as _io

                from PIL import Image as _Image

                _buf = _io.BytesIO()
                with _Image.open(_io.BytesIO(image)) as _im:
                    # draft() lets JPEG-backed sources decode at reduced scale;
                    # for other formats it is a no-op. Keeps a 19MB PNG page from
                    # materialising ~150-200MB of pixels inside the prefetch's
                    # worker thread just to re-encode it.
                    _im.draft("RGB", _im.size)
                    _im.convert("RGB").save(_buf, format="JPEG", quality=90)
                transcoded = _buf.getvalue()
                if len(transcoded) > TEXTRACT_SYNC_LIMIT_BYTES:
                    # Still over the wire limit: the call would fail anyway, and
                    # saying so beats a confusing Textract error.
                    log.warning(
                        "textract payload still %.1fMB after transcode — skipping the read",
                        len(transcoded) / 1e6,
                    )
                    return PageRead(error="payload over Textract's synchronous limit")
                image = transcoded
                log.info("textract payload transcoded to JPEG (%.1fMB)", len(image) / 1e6)

            client = boto3.client(
                "textract", region_name=settings.AWS_REGION or "us-east-1"
            )
            self.calls += 1
            resp = client.detect_document_text(Document={"Bytes": image})
        except Exception as exc:  # noqa: BLE001 — a secondary source never breaks extraction
            log.warning("textract page read failed", exc_info=True)
            return PageRead(error=f"{type(exc).__name__}: {exc}"[:200])

        w, h = size
        lines: list[str] = []
        words: list[tuple[str, int, int, int, int]] = []
        handwriting = low_conf = 0
        for b in resp.get("Blocks", []):
            kind = b.get("BlockType")
            txt = b.get("Text") or ""
            if kind == "LINE":
                lines.append(txt)
            elif kind == "WORD" and txt:
                bb = (b.get("Geometry") or {}).get("BoundingBox") or {}
                words.append(
                    (
                        txt,
                        int(bb.get("Left", 0) * w),
                        int(bb.get("Top", 0) * h),
                        int(bb.get("Width", 0) * w),
                        int(bb.get("Height", 0) * h),
                    )
                )
                # TextType and Confidence come free on the same response. They are what
                # tells a later stage whether this page is worth a second opinion.
                if b.get("TextType") == "HANDWRITING":
                    handwriting += 1
                if (b.get("Confidence") or 100.0) < LOW_CONFIDENCE:
                    low_conf += 1
        return PageRead(
            text="\n".join(lines),
            words=words,
            handwriting_words=handwriting,
            low_confidence_words=low_conf,
            total_words=len(words),
        )
