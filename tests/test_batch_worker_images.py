"""Worker-level tests for image inputs in the async batch lane (plan 064).

``_process_item`` runs with a REAL ``Extractor`` (image→PDF conversion +
PDF parse under the conftest OCR stub) and fakes for the repo, result
store, and billing — so staged image files exercise the same
``aextract_from_bytes`` path production takes, without S3/Postgres.
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import UTC, datetime
from types import SimpleNamespace

from PIL import Image

from extract.core import Extractor
from extract.core.batch import ItemErrorCode
from extract.repos.batches import ClaimedItem
from extract.storage.inline import InlineStorage
from extract.workers.batch_worker import WorkerState, _process_item


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _tiff_bytes(n_frames: int) -> bytes:
    frames = [Image.new("RGB", (120, 90), (i * 50 % 255, 80, 40)) for i in range(n_frames)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def _gif_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, format="GIF")
    return buf.getvalue()


class _Repo:
    def __init__(self) -> None:
        self.succeeded: list[dict] = []
        self.failed: list[dict] = []
        self.rescheduled: list[dict] = []

    async def heartbeat_lease(self, **kwargs) -> None:
        pass

    async def update_item_succeeded(self, **kwargs) -> None:
        self.succeeded.append(kwargs)

    async def update_item_failed(self, **kwargs) -> None:
        self.failed.append(kwargs)

    async def reschedule_item(self, **kwargs) -> None:
        self.rescheduled.append(kwargs)


class _ResultStore:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.written: list[bytes] = []

    async def fetch_upload(self, *, bucket: str, key: str) -> bytes:
        return self.payload

    async def write_result(self, *, batch_id: str, item_id: str, body: bytes):
        self.written.append(body)
        return SimpleNamespace(
            bucket="results", key=f"{batch_id}/{item_id}.json", bytes_written=len(body)
        )


def _claimed(filename: str) -> ClaimedItem:
    return ClaimedItem(
        item_id="item_img",
        batch_id="batch_img",
        customer_id="cus_img",
        phi_safe=False,
        file_id="file_img",
        file_s3_bucket="uploads",
        file_s3_key=f"staged/{filename}",
        file_filename=filename,
        file_content_type=None,
        lease_token="lease",
        lease_expires_at=datetime.now(tz=UTC),
        attempts=1,
        extract_text=True,
        extract_images=False,
        ocr="auto",
        engine="baseline",
    )


def _state(payload: bytes) -> WorkerState:
    state = WorkerState()
    state.repo = _Repo()  # type: ignore[assignment]
    state.result_store = _ResultStore(payload)  # type: ignore[assignment]
    state.extractor = Extractor(storage=InlineStorage())
    state.stop_event = asyncio.Event()
    return state


async def test_staged_png_item_succeeds_with_one_page():
    state = _state(_png_bytes())
    await _process_item(state, _claimed("scan.png"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.failed == []
    assert len(repo.succeeded) == 1
    assert repo.succeeded[0]["page_count"] == 1

    result = json.loads(state.result_store.written[0])  # type: ignore[union-attr]
    assert result["page_count"] == 1
    assert result["batch_id"] == "batch_img"


async def test_staged_multiframe_tiff_counts_page_per_frame():
    state = _state(_tiff_bytes(3))
    await _process_item(state, _claimed("fax.tiff"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.failed == []
    assert repo.succeeded[0]["page_count"] == 3


async def test_unsupported_image_bytes_fail_with_unsupported_input():
    # GIF bytes staged under a .png name: magic is inconclusive, the filename
    # routes to the image converter, and the converter rejects GIF.
    state = _state(_gif_bytes())
    await _process_item(state, _claimed("photo.png"))

    repo: _Repo = state.repo  # type: ignore[assignment]
    assert repo.succeeded == []
    assert len(repo.failed) == 1
    assert repo.failed[0]["error_code"] == ItemErrorCode.UNSUPPORTED_INPUT
