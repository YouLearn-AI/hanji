"""Settings loaded from environment variables via pydantic-settings.

Defaults are the production-effective configuration of the hosted service this
repo was extracted from: the values below are what the pipeline was measured
and shipped with. The parse pipeline runs with just ``PARSE_MODEL_URL`` set;
schema extraction additionally needs a Gemini transport (``GEMINI_API_KEY`` or
``GOOGLE_VERTEX_PROJECT``). Every enrichment beyond that (Textract cross-read,
Cloud Vision citation tightening) fails open when its credentials are absent.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

for file in (".env.local", ".env"):
    if os.path.exists(file):
        load_dotenv(file)
        break


class Settings(BaseSettings):
    # --- Parse model endpoint (the fine-tuned Qwen3-VL parse model) ---
    # Any endpoint implementing the /predict contract in serving/server.py:
    # POST {"image_b64", "prompt", "max_new_tokens", ...} -> {"raw", ...}.
    # See serving/SELF_HOSTING.md to run one from the published weights
    # (https://huggingface.co/hanji-dev/hanji-parse-4b).
    PARSE_MODEL_URL: str | None = None
    #: Optional. Sent as ``Authorization: Api-Key <value>`` (Baseten-style).
    PARSE_MODEL_API_KEY: str | None = None
    # Generous: a scaled-to-zero endpoint eats a cold start on the first page,
    # and a dense page can decode thousands of tokens on a 4B model.
    PARSE_MODEL_TIMEOUT_SECONDS: float = 180.0
    # First-pass output budget: serves legitimately-dense pages in one pass;
    # pages that finish early are unaffected.
    PARSE_MODEL_MAX_NEW_TOKENS: int = 8192

    # --- Storage backend (for extracted image chunks) ---
    EXTRACT_STORAGE: str = "inline"  # "inline" | "local"
    EXTRACT_LOCAL_STORAGE_DIR: str = "./extracted"

    # --- Gemini (schema extraction + OCR fallback), two transports ---
    #: Plain Gemini developer-API key — the simplest self-hosted setup.
    GEMINI_API_KEY: str | None = None
    #: Vertex AI project. When set, Vertex wins over GEMINI_API_KEY.
    GOOGLE_VERTEX_PROJECT: str | None = None
    GOOGLE_VERTEX_LOCATION: str = "global"
    #: Optional inline service-account JSON; ADC is used when unset.
    GOOGLE_VERTEX_SA_JSON: str | None = None

    # --- Gemini OCR fallback (per-page, when the parse model's read is unusable) ---
    GEMINI_OCR_MODEL: str = "gemini-3.5-flash"
    GEMINI_OCR_TIMEOUT_SECONDS: float = 120.0
    GEMINI_OCR_THINKING_BUDGET: int | None = 0
    GEMINI_OCR_MAX_OUTPUT_TOKENS: int = 65536
    #: Media resolution for Gemini image parts ("" = API default).
    EXTRACT_MEDIA_RESOLUTION: str = ""

    # --- Gemini schema extraction ---
    GEMINI_EXTRACT_MODEL: str | None = None  # None -> GEMINI_OCR_MODEL
    GEMINI_EXTRACT_TIMEOUT_SECONDS: float = 90.0
    GEMINI_EXTRACT_TOTAL_DEADLINE_SECONDS: float = 240.0
    GEMINI_EXTRACT_MAX_TIMEOUT_ATTEMPTS: int = 2
    GEMINI_EXTRACT_THINKING_BUDGET: int | None = 0
    GEMINI_EXTRACT_MAX_OUTPUT_TOKENS: int = 65536
    #: >1 -> request N candidates + grounding-gated union (recovers omissions).
    #: The hosted service shipped with 2.
    GEMINI_EXTRACT_CANDIDATE_COUNT: int = 2
    #: Sampling temperature used only when candidate_count > 1 (candidates must
    #: differ to be worth unioning).
    GEMINI_EXTRACT_CANDCOUNT_TEMP: float = 0.4
    #: Vertex scheduling tier ("standard" | "flex"). Flex is cheaper and
    #: shed-able; the per-call flex->standard fallback keeps answers unchanged.
    EXTRACT_SERVICE_TIER: str = "standard"
    EXTRACT_FLEX_TIMEOUT_SECONDS: float = 60.0

    # --- Parse pipeline ---
    #: Legal-document post-processing (transcript/multipanel/concordance
    #: repair). Default ON in production.
    OCR_LEGAL_POSTPROCESS: bool = True
    #: Per-chunk confidence from serving logprobs (feeds the low-confidence
    #: re-read; required by SCHEMA_REREAD_ENABLED).
    OCR_CHUNK_CONFIDENCE: bool = True
    #: Escalated-sampling retry ladder for unusable reads (default off).
    OCR_RETRY_ESCALATION: bool = False
    OCR_RETRY_RECOVERY: bool = False
    #: Numeric self-consistency vote over repeated reads of the same page.
    #: Production runs with the gate ON.
    EXTRACT_OCR_CONSISTENCY_GATE_ENABLED: bool = True
    EXTRACT_OCR_CONSISTENCY_ELIGIBILITY: Literal["risk", "numeric", "both"] = "risk"
    EXTRACT_OCR_CONSISTENCY_MIN_VALUES: int = 16
    EXTRACT_OCR_CONSISTENCY_MIN_AGREEMENT: float = 0.9
    EXTRACT_OCR_CONSISTENCY_MAX_PAGES_PER_DOC: int = 3
    EXTRACT_OCR_CONSISTENCY_VERIFY_TIMEOUT_S: float = 45.0
    EXTRACT_OCR_CONSISTENCY_RISK_CAP_FRACTION: float = 0.85
    EXTRACT_OCR_CONSISTENCY_RISK_EMPTY_RUN: int = 8
    #: Born-digital fast-path / native-text override / text-layer completeness
    #: heuristics (all default off — every page goes through the model).
    OCR_BORNDIGITAL_FASTPATH: bool = False
    OCR_NATIVE_TEXT_OVERRIDE: bool = False
    OCR_TEXTLAYER_COMPLETENESS: bool = False
    #: Page-coverage shadow metrics (default off).
    OCR_COVERAGE_MODE: Literal["off", "shadow", "route"] = "off"
    OCR_COVERAGE_MIN_BAND_FRAC: float = 0.5
    OCR_COVERAGE_MAX_UNCOVERED: float = 0.40
    #: Cross-page table assembly (default off — output byte-identical to the
    #: unassembled pipeline).
    EXTRACT_ASSEMBLY_MODE: Literal["off", "shadow", "on"] = "off"
    #: Kept for pipeline compatibility; the open-source build has no review
    #: artifact store, so leave this off.
    EXTRACT_REVIEW_ARTIFACTS_ENABLED: bool = False

    # --- Schema extraction: recipe + refinements ---
    #: The LAYOUT recipe ((x%,y%)+[SECTION] chunk annotation + entity-binding
    #: prompt) is the default schema pass. False -> plain pass.
    EXTRACT_LAYOUT_RECIPE_ENABLED: bool = True
    #: Cross-engine read (Textract) on handwriting-gated pages, fused into the
    #: extraction context. Needs AWS credentials; fails open without them.
    EXTRACT_XREF_GEMINI_MODEL: str = "gemini-3.5-flash"
    EXTRACT_XREF_RENDER_SCALE: float = 2.0
    #: Cloud Vision citation-box tightening (value-preserving; fails open).
    EXTRACT_VISION_TIGHTEN_ENABLED: bool = True
    EXTRACT_VISION_TIGHTEN_CONCURRENCY: int = 16
    #: Low-confidence high-stakes field re-read (agreement-gated, fail-open).
    SCHEMA_REREAD_ENABLED: bool = True
    SCHEMA_REREAD_GATE: float = 0.80
    SCHEMA_REREAD_MODEL: str = "gemini-2.5-flash-lite"
    #: Whole-document text on the schema response (`include_ocr_text`).
    EXTRACT_INCLUDE_OCR_TEXT: bool = True
    #: Pages attached to the extraction call / cross-read caps.
    EXTRACT_PAGE_IMAGES_MAX_PAGES: int = 3
    #: Output-shape knobs (deployment-wide defaults).
    EXTRACT_AUX_ARRAYS_INLINE: bool = False
    EXTRACT_OMIT_NULL_LEAVES: bool = False
    EXTRACT_SHORT_ROW_KEYS: bool = False
    EXTRACT_BARE_SCALARS: bool = False
    EXTRACT_CACHE_STAGGER_SECONDS: float = 2.0
    #: Value keys normalized as dates when quoted evidence disagrees on format.
    EXTRACT_NORMALIZED_DATE_KEYS: str = "issue_date,issue_time"

    # --- AWS (Textract cross-read; optional) ---
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_REGION: str | None = None

    # --- API server ---
    #: Hard wall-clock bound per request (0 disables).
    EXTRACT_REQUEST_TIMEOUT_SECONDS: float = 900.0
    #: Comma-separated CORS origins ("" = no CORS headers).
    CORS_ALLOWED_ORIGINS: str = ""

    # --- Async batch lane (POST /v1/files, POST /v1/batches) ---
    #: Postgres DSN for batch/webhook state. Unset -> the async routes 503
    #: and the worker refuses to start; the sync routes are unaffected.
    DATABASE_URL: str | None = None
    #: S3 buckets for uploaded source files and result JSON. Any
    #: S3-compatible store works (AWS, MinIO, ...).
    EXTRACT_UPLOADS_BUCKET: str | None = None
    EXTRACT_RESULTS_BUCKET: str | None = None
    #: Custom S3 endpoint (MinIO etc.); unset -> AWS.
    EXTRACT_S3_ENDPOINT_URL: str | None = None
    #: Endpoint presigned URLs are minted against, when clients reach the
    #: store on a different host than the API/worker do (e.g. compose:
    #: internal http://minio:9000, public http://localhost:9100).
    EXTRACT_S3_PUBLIC_ENDPOINT_URL: str | None = None
    #: Validity window of presigned upload/download URLs (seconds). Tighter
    #: than the object TTL so URLs handed out today are stale by the time a
    #: lifecycle rule deletes the object behind them.
    EXTRACT_PRESIGNED_UPLOAD_TTL_SECONDS: int = 60 * 30  # 30 min
    EXTRACT_PRESIGNED_DOWNLOAD_TTL_SECONDS: int = 60 * 15  # 15 min
    #: Hard limit applied at POST /v1/files. Falls back to MAX_SIZE_BYTES
    #: in `extract.core.pdf` if unset.
    EXTRACT_UPLOAD_MAX_BYTES: int | None = None
    #: Worker tuning. Each worker process polls Postgres for the next ready
    #: item; concurrency = simultaneous items this process will run (the
    #: hosted service ran 16 per 2-vCPU task).
    EXTRACT_WORKER_CONCURRENCY: int = 4
    EXTRACT_WORKER_POLL_INTERVAL_MS: int = 250
    EXTRACT_WORKER_QUEUE_METRIC_INTERVAL_SECONDS: int = 60
    EXTRACT_WORKER_LEASE_SECONDS: int = 300
    #: Batch/file/result retention window (3 days). Keep in sync with any
    #: object-lifecycle rule you configure on the buckets.
    EXTRACT_BATCH_DEFAULT_TTL_SECONDS: int = 3 * 24 * 60 * 60
    #: Batch status-polling rate limit (per second, in-process, 429 over it;
    #: 0 disables). Steers runaway pollers toward webhooks.
    EXTRACT_BATCH_POLL_RATE_LIMIT_PER_SECOND: int = 200

    # --- Batch completion webhooks ---
    #: Master switch for the in-worker dispatcher + reconciler coroutines.
    EXTRACT_WEBHOOKS_ENABLED: bool = True
    #: Registered endpoints cap (`python -m extract.webhooks add`).
    EXTRACT_WEBHOOK_MAX_ENDPOINTS: int = 10
    #: Dispatcher tuning. The claim loop mirrors the item worker: FOR UPDATE
    #: SKIP LOCKED over due pending rows OR delivering rows w/ expired leases.
    EXTRACT_WEBHOOK_DISPATCH_CONCURRENCY: int = 4
    EXTRACT_WEBHOOK_POLL_INTERVAL_MS: int = 500
    EXTRACT_WEBHOOK_LEASE_SECONDS: int = 60
    #: Reconciler: sweep opted-in terminal batches with no event row.
    #: Bounded lookback; "terminal for >N minutes" grace before reconciling.
    EXTRACT_WEBHOOK_RECONCILE_LOOKBACK_HOURS: int = 24
    EXTRACT_WEBHOOK_RECONCILE_GRACE_MINUTES: int = 15
    EXTRACT_WEBHOOK_RECONCILE_INTERVAL_SECONDS: int = 300
    #: Self-host escape hatch: allow http:// and private/loopback webhook
    #: destinations (delivery to services on your own network). Leave off
    #: whenever endpoint URLs come from untrusted users.
    EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS: bool = False

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=(".env", ".env.local"),
    )


settings = Settings()
