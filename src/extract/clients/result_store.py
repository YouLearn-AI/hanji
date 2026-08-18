"""Worker-side reader/writer for the async batch S3 buckets.

Distinct from :mod:`extract.clients.upload_signer`:

- ``UploadSigner`` is **API-side** and only signs presigned URLs; bytes
  never transit the API.
- ``ResultStore`` is **worker-side**: it actually reads upload bytes from
  the uploads bucket and writes result JSON to the results bucket.

Works against AWS S3 or any S3-compatible store (MinIO etc.) via
``EXTRACT_S3_ENDPOINT_URL``. Server-side encryption is left to the bucket
default; we don't pass encryption parameters explicitly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from botocore.config import Config

from extract.config import settings


@dataclass(frozen=True)
class StoredResult:
    bucket: str
    key: str
    bytes_written: int


class ResultStore:
    """Worker boto3 wrapper. Constructed once per worker process."""

    def __init__(
        self,
        *,
        uploads_bucket: str | None,
        results_bucket: str | None,
        region: str | None,
        endpoint_url: str | None = None,
    ) -> None:
        self._uploads_bucket = uploads_bucket
        self._results_bucket = results_bucket
        self._region = region
        self._endpoint_url = endpoint_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=self._region,
                endpoint_url=self._endpoint_url,
                config=Config(
                    signature_version="s3v4",
                    max_pool_connections=50,
                    retries={"max_attempts": 3, "mode": "standard"},
                    # Path-style keeps custom endpoints (MinIO) working
                    # without per-bucket DNS.
                    s3={"addressing_style": "path"} if self._endpoint_url else {},
                ),
            )
        return self._client

    async def fetch_upload(self, *, bucket: str, key: str) -> bytes:
        """Download upload bytes into memory.

        We don't stream because the extractor expects a single ``bytes``
        argument. Large files (max 5 GB by S3 single-PUT cap) are bounded
        by the API's ``EXTRACT_UPLOAD_MAX_BYTES`` setting; we trust that
        cap rather than re-checking here.
        """

        def _get() -> bytes:
            response = self._get_client().get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()

        return await asyncio.to_thread(_get)

    async def write_result(
        self,
        *,
        batch_id: str,
        item_id: str,
        body: bytes,
        content_type: str = "application/json",
    ) -> StoredResult:
        if not self._results_bucket:
            raise RuntimeError("EXTRACT_RESULTS_BUCKET must be set for the worker")
        key = f"{batch_id}/{item_id}.json"

        def _put() -> None:
            self._get_client().put_object(
                Bucket=self._results_bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )

        await asyncio.to_thread(_put)
        return StoredResult(
            bucket=self._results_bucket,
            key=key,
            bytes_written=len(body),
        )


def from_settings() -> ResultStore:
    return ResultStore(
        uploads_bucket=settings.EXTRACT_UPLOADS_BUCKET,
        results_bucket=settings.EXTRACT_RESULTS_BUCKET,
        region=settings.AWS_REGION,
        endpoint_url=settings.EXTRACT_S3_ENDPOINT_URL,
    )


__all__ = ["ResultStore", "StoredResult", "from_settings"]
