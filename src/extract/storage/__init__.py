"""Output storage backends for extracted images and thumbnails.

Default is ``inline`` — returns image bytes base64-encoded inside the
response. For higher-throughput deployments, swap in ``local``.
"""

from extract.storage.base import Storage, StorageResult
from extract.storage.inline import InlineStorage
from extract.storage.local import LocalStorage

__all__ = ["InlineStorage", "LocalStorage", "Storage", "StorageResult", "from_settings"]


def from_settings() -> Storage:
    """Build the configured storage backend from ``extract.config.settings``."""
    from extract.config import settings

    name = (settings.EXTRACT_STORAGE or "inline").lower()
    if name == "inline":
        return InlineStorage()
    if name == "local":
        return LocalStorage(settings.EXTRACT_LOCAL_STORAGE_DIR)
    raise ValueError(f"Unknown EXTRACT_STORAGE: {name}")
