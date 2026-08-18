"""Public-API error shape.

FastAPI's default ``HTTPException`` handler nests ``detail`` under
``{"detail": ...}``. The public async-batch contract (see
``mintlify/introduction.mdx`` → *Error response body*) instead promises a
flat, switchable top-level ``error`` string, e.g.::

    {"error": "file_not_uploaded", "file_ids": ["file_abc123"]}

``FlatAPIError`` carries that body verbatim and ``flat_api_error_handler``
emits it un-nested. This is deliberately scoped to the *public* routes
(``/v1/files``, ``/v1/batches``): the internal demo lane keeps the default
nested shape, which its BFF already parses (``body.detail.error``).
"""

from __future__ import annotations

from typing import Any

import orjson
from fastapi import HTTPException, Request
from starlette.responses import Response


class FlatAPIError(HTTPException):
    """An ``HTTPException`` whose ``detail`` dict is the entire response body.

    Raise this instead of a bare ``HTTPException`` on public routes when the
    body should be a flat ``{"error": ...}`` object the caller can switch on.
    """

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        super().__init__(status_code=status_code, detail=body)


async def flat_api_error_handler(request: Request, exc: FlatAPIError) -> Response:
    """Emit ``exc.detail`` as the top-level body instead of nesting it."""
    return Response(
        content=orjson.dumps(exc.detail),
        status_code=exc.status_code,
        media_type="application/json",
        headers=exc.headers,
    )
