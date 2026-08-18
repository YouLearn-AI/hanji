"""Filesystem storage — writes to a directory, returns ``file://`` URLs."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from extract.storage.base import Storage, StorageResult

_MIME_TO_EXT = {
    "image/webp": "webp",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
}


class LocalStorage(Storage):
    name = "local"

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir).expanduser().resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    async def put(
        self,
        data: bytes,
        *,
        mime: str = "image/webp",
        prefix: str = "images",
    ) -> StorageResult:
        ext = _MIME_TO_EXT.get(mime, mime.split("/")[-1] or "bin")
        dest_dir = self._base / prefix
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.{ext}"
        dest = dest_dir / filename
        await asyncio.to_thread(dest.write_bytes, data)
        return StorageResult(url=dest.as_uri(), mime=mime)
