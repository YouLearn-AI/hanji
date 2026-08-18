"""One home for Google GenAI client construction.

Two transports, resolved from settings in this order:

1. **Vertex AI** — when ``GOOGLE_VERTEX_PROJECT`` is set. Credentials come from
   ``GOOGLE_VERTEX_SA_JSON`` (an inline service-account key) or Application
   Default Credentials. This is the transport hosted deployments with a Google
   Cloud project use.
2. **Gemini API key** — when ``GEMINI_API_KEY`` is set. The plain developer
   API; the simplest way to run schema extraction self-hosted.

Every Gemini consumer in this repo (schema extraction, the OCR fallback, the
cross-engine second reader, the low-confidence re-read) builds its client here
so the two transports work everywhere or nowhere.
"""

from __future__ import annotations

import json
from typing import Any


def genai_configured() -> bool:
    """Whether either Gemini transport is configured."""
    from extract.config import settings

    return bool(settings.GOOGLE_VERTEX_PROJECT or settings.GEMINI_API_KEY)


def make_genai_client(genai: Any | None = None) -> Any:
    """Build a ``google.genai.Client`` from settings.

    Raises ``RuntimeError`` when neither transport is configured — callers that
    prefer to fail open should check :func:`genai_configured` first.
    """
    from extract.config import settings

    if genai is None:
        from google import genai as genai_mod

        genai = genai_mod
    if settings.GOOGLE_VERTEX_PROJECT:
        credentials = None
        if settings.GOOGLE_VERTEX_SA_JSON:
            from google.oauth2.service_account import Credentials

            credentials = Credentials.from_service_account_info(
                json.loads(settings.GOOGLE_VERTEX_SA_JSON),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        return genai.Client(
            vertexai=True,
            project=settings.GOOGLE_VERTEX_PROJECT,
            location=settings.GOOGLE_VERTEX_LOCATION,
            credentials=credentials,
        )
    if settings.GEMINI_API_KEY:
        return genai.Client(api_key=settings.GEMINI_API_KEY)
    raise RuntimeError(
        "Gemini is not configured. Set GEMINI_API_KEY (developer API) or "
        "GOOGLE_VERTEX_PROJECT (Vertex AI)."
    )
