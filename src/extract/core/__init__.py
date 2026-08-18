"""Pure extraction library. Depends on PyMuPDF / Pillow / numpy only —
no FastAPI, no boto3 at import time.
"""

from extract.core.errors import (
    DocumentTooLarge,
    ExtractionFailed,
    PageLimitExceeded,
    RemoteFetchError,
    UnsupportedInput,
    UpstreamTimeout,
)
from extract.core.extractor import Extractor
from extract.core.kind import detect_kind, detect_kind_from_magic, detect_kind_from_name
from extract.core.models import (
    Chunk,
    ChunkType,
    ExtractRequest,
    ExtractResponse,
    InputKind,
    PageDimensions,
    Segment,
    SegmentMember,
    TablePartRef,
    TextPartRef,
)

__all__ = [
    "Chunk",
    "ChunkType",
    "DocumentTooLarge",
    "ExtractRequest",
    "ExtractResponse",
    "ExtractionFailed",
    "Extractor",
    "InputKind",
    "PageDimensions",
    "PageLimitExceeded",
    "RemoteFetchError",
    "Segment",
    "SegmentMember",
    "TablePartRef",
    "TextPartRef",
    "UnsupportedInput",
    "UpstreamTimeout",
    "detect_kind",
    "detect_kind_from_magic",
    "detect_kind_from_name",
]
