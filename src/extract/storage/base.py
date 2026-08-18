from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class StorageResult:
    """Where an uploaded blob landed. Either ``url`` or ``b64`` will be set."""

    url: str | None = None
    b64: str | None = None
    mime: str | None = None


class Storage(Protocol):
    name: str

    async def put(
        self,
        data: bytes,
        *,
        mime: str = "image/webp",
        prefix: str = "images",
    ) -> StorageResult: ...
