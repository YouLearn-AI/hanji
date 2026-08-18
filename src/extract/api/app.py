"""FastAPI app: document parsing + schema extraction.

Sync routes (see ``routes/v1.py``):

- ``POST /v1/parse``          — parse a document by URL
- ``POST /v1/parse/file``     — parse an uploaded document
- ``POST /v1/extract/schema`` — fill a JSON schema from a document by URL
- ``POST /v1/extract/schema/file`` — fill a schema from an uploaded document

Async batch lane (needs ``DATABASE_URL`` + the S3 buckets; see
``docs/batch.md``):

- ``POST /v1/files`` / ``GET /v1/files/{id}`` — presigned uploads
- ``POST /v1/batches`` + poll/result/cancel/list — batch lifecycle
  (processed by ``python -m extract.workers.batch_worker``)

There is no auth, billing, or telemetry here — front this server with your own
gateway if you expose it publicly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from extract.config import settings
from extract.core import Extractor
from extract.gemini_extract import from_settings as gemini_extract_from_settings
from extract.storage import from_settings as storage_from_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.storage = storage_from_settings()
    app.state.extractor = Extractor(storage=app.state.storage)
    # Lazy: no Gemini round-trip until the first schema call, so the app boots
    # without Gemini credentials (parse still works; schema routes 503).
    app.state.schema_model = gemini_extract_from_settings()
    yield


app = FastAPI(
    title="extract",
    description="Document parsing and grounded schema extraction.",
    lifespan=lifespan,
)

if settings.CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}


from extract.api.errors import FlatAPIError, flat_api_error_handler  # noqa: E402
from extract.api.routes.batches import BatchUpdateEvent  # noqa: E402
from extract.api.routes.batches import router as batches_router  # noqa: E402
from extract.api.routes.files import router as files_router  # noqa: E402
from extract.api.routes.v1 import router as v1_router  # noqa: E402

app.include_router(v1_router)
app.include_router(files_router)
app.include_router(batches_router)
app.add_exception_handler(FlatAPIError, flat_api_error_handler)

# Documentation-only: register the completion webhook on `app.webhooks` so it
# lands in the generated OpenAPI `webhooks` object. This defines no callable
# route — the real delivery is the worker's dispatcher; the live serializer is
# ``extract.core.webhooks.build_batch_update_body``.
@app.webhooks.post("batch.update")
def batch_update_webhook(body: BatchUpdateEvent) -> None:
    """Sent to your registered endpoints (`webhook: {"mode": "svix"}`) or the
    inline `url` (`"mode": "direct"`) when a batch reaches a terminal status.
    Signed deliveries are Svix-wire-compatible — verify with the `svix`
    library. See docs/webhooks.md."""

