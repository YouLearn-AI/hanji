"""In-memory storage — returns base64-encoded bytes in the response.

Zero external deps. Good default for CLIs, one-off scripts, and low-volume
API usage. For pipelines that extract thousands of images per document,
switch to the S3 or local backend to keep response payloads small.
"""

from __future__ import annotations

import base64

from extract.storage.base import Storage, StorageResult


class InlineStorage(Storage):
    name = "inline"

    async def put(
        self,
        data: bytes,
        *,
        mime: str = "image/webp",
        prefix: str = "images",
    ) -> StorageResult:
        return StorageResult(
            b64=base64.b64encode(data).decode("ascii"),
            mime=mime,
        )
