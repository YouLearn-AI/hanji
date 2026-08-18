"""Presigned S3 URLs for the async batch lane.

Two operations:

- ``presign_upload`` — POST /v1/files returns a single presigned PUT URL.
  The customer streams bytes directly to S3; our API never proxies upload
  bytes. Multipart is intentionally not supported in v1 (single-PUT
  covers up to 5 GB which is more than enough for any realistic PDF).
- ``presign_download`` — GET /v1/batches/{id}/items/{id}/result issues
  a short-lived presigned GET so the client can fetch result JSON
  directly from the results bucket.

We do NOT pass ``ServerSideEncryption`` parameters in the presigned URL —
encryption is the bucket's default and S3 applies it on the PUT regardless.
Adding SSE headers here would force the client to set matching
``x-amz-server-side-encryption`` request headers or get a 400.

Works against AWS S3 or any S3-compatible store (MinIO etc.):
``EXTRACT_S3_ENDPOINT_URL`` points every operation at the store, and
``EXTRACT_S3_PUBLIC_ENDPOINT_URL`` (optional) presigns URLs against the
host CLIENTS reach — needed when the API talks to the store on an internal
hostname (compose service name) but callers PUT/GET from outside.

Boto3's ``generate_presigned_url`` is a synchronous, fast (<1ms) call
that produces a URL signed with the task role's credentials. No I/O.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from botocore.config import Config

from extract.config import settings


@dataclass(frozen=True)
class PresignedUpload:
    method: str  # always "PUT" for v1
    url: str
    expires_at: datetime
    bucket: str
    key: str


@dataclass(frozen=True)
class PresignedDownload:
    url: str
    expires_at: datetime


class UploadSigner:
    """Wrapper around boto3 S3's presigned-URL machinery.

    Constructed once per process; the boto3 client itself is thread-safe
    for read-only operations like signing.
    """

    def __init__(
        self,
        *,
        uploads_bucket: str | None,
        results_bucket: str | None,
        region: str | None,
        upload_ttl_seconds: int,
        download_ttl_seconds: int,
        endpoint_url: str | None = None,
        public_endpoint_url: str | None = None,
    ) -> None:
        self._uploads_bucket = uploads_bucket
        self._results_bucket = results_bucket
        self._upload_ttl = upload_ttl_seconds
        self._download_ttl = download_ttl_seconds
        # SigV4 explicitly — required for KMS-encrypted buckets.
        self._client = None
        self._sign_client = None
        self._region = region
        self._endpoint_url = endpoint_url
        self._public_endpoint_url = public_endpoint_url

    @property
    def uploads_bucket(self) -> str | None:
        return self._uploads_bucket

    @property
    def results_bucket(self) -> str | None:
        return self._results_bucket

    @property
    def configured_for_uploads(self) -> bool:
        return bool(self._uploads_bucket)

    @property
    def configured_for_results(self) -> bool:
        return bool(self._results_bucket)

    def _make_client(self, endpoint_url: str | None):
        import boto3

        return boto3.client(
            "s3",
            region_name=self._region,
            endpoint_url=endpoint_url,
            config=Config(
                signature_version="s3v4",
                # Path-style keeps custom endpoints (MinIO) working without
                # per-bucket DNS; AWS keeps virtual-host addressing.
                s3={"addressing_style": "path"} if endpoint_url else {},
            ),
        )

    def _get_client(self):
        if self._client is None:
            self._client = self._make_client(self._endpoint_url)
        return self._client

    def _get_sign_client(self):
        """The client used to MINT presigned URLs — signs against the
        public endpoint when one is configured, so the URL works from
        outside the deployment's network."""
        if self._public_endpoint_url is None:
            return self._get_client()
        if self._sign_client is None:
            self._sign_client = self._make_client(self._public_endpoint_url)
        return self._sign_client

    def upload_key_for(self, *, customer_id: str, file_id: str) -> str:
        """Stable upload key. Customer-scoped so cross-tenant collisions
        are impossible by construction; file_id is unique."""
        return f"{customer_id}/{file_id}"

    def result_key_for(self, *, batch_id: str, item_id: str) -> str:
        return f"{batch_id}/{item_id}.json"

    async def presign_upload(
        self,
        *,
        customer_id: str,
        file_id: str,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> PresignedUpload:
        if not self._uploads_bucket:
            raise RuntimeError("EXTRACT_UPLOADS_BUCKET must be set for /v1/files")
        key = self.upload_key_for(customer_id=customer_id, file_id=file_id)
        params: dict = {
            "Bucket": self._uploads_bucket,
            "Key": key,
        }
        if content_type:
            params["ContentType"] = content_type
        if content_length is not None:
            params["ContentLength"] = content_length

        def _sign() -> str:
            return self._get_sign_client().generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=self._upload_ttl,
                HttpMethod="PUT",
            )

        url = await asyncio.to_thread(_sign)
        return PresignedUpload(
            method="PUT",
            url=url,
            expires_at=datetime.now(tz=UTC) + timedelta(seconds=self._upload_ttl),
            bucket=self._uploads_bucket,
            key=key,
        )

    async def presign_download(
        self,
        *,
        bucket: str,
        key: str,
        ttl_seconds: int | None = None,
    ) -> PresignedDownload:
        ttl = ttl_seconds or self._download_ttl

        def _sign() -> str:
            return self._get_sign_client().generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl,
                HttpMethod="GET",
            )

        url = await asyncio.to_thread(_sign)
        return PresignedDownload(
            url=url,
            expires_at=datetime.now(tz=UTC) + timedelta(seconds=ttl),
        )

    async def head_upload(self, *, key: str) -> dict | None:
        """Cheap existence + size check after the client claims to have uploaded.

        Returns the boto3 ``head_object`` response dict, or ``None`` if the
        object is unavailable for any reason (missing, lifecycle-expired,
        access denied, transient error). Failing closed here would only
        surface as a 500 on GET /v1/files/{id}; returning None lets the
        route still respond with the row's last-known state.
        """
        if not self._uploads_bucket:
            return None
        try:
            return await asyncio.to_thread(
                self._get_client().head_object,
                Bucket=self._uploads_bucket,
                Key=key,
            )
        except Exception:  # noqa: BLE001 - intentionally swallow all
            return None

    async def put_bytes(
        self,
        *,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """Server-side PUT of object bytes.

        The public /v1/files flow hands the client a presigned PUT and never
        sees the bytes; this server-side PUT exists for tools and tests that
        already hold them. Encryption stays the bucket default (matching the
        presign path).
        """

        def _put() -> None:
            params = {"Bucket": bucket, "Key": key, "Body": data}
            if content_type:
                params["ContentType"] = content_type
            self._get_client().put_object(**params)

        await asyncio.to_thread(_put)


def from_settings() -> UploadSigner:
    return UploadSigner(
        uploads_bucket=settings.EXTRACT_UPLOADS_BUCKET,
        results_bucket=settings.EXTRACT_RESULTS_BUCKET,
        region=settings.AWS_REGION,
        upload_ttl_seconds=settings.EXTRACT_PRESIGNED_UPLOAD_TTL_SECONDS,
        download_ttl_seconds=settings.EXTRACT_PRESIGNED_DOWNLOAD_TTL_SECONDS,
        endpoint_url=settings.EXTRACT_S3_ENDPOINT_URL,
        public_endpoint_url=settings.EXTRACT_S3_PUBLIC_ENDPOINT_URL,
    )


__all__ = [
    "PresignedDownload",
    "PresignedUpload",
    "UploadSigner",
    "from_settings",
]
