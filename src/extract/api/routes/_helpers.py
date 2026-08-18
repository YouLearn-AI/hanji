"""Shared FastAPI dependencies for /v1/files and /v1/batches.

The hosted service resolved a per-key ``RequestContext`` (customer id, PHI
lane) from API-key auth here. This build is single-tenant with no built-in
auth — front the server with your own gateway — so the context is a
constant. The shape is kept so the route code and the row schema stay
line-for-line comparable with the hosted service.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from extract.clients.upload_signer import UploadSigner
from extract.clients.upload_signer import from_settings as upload_signer_from_settings
from extract.repos.batches import BatchRepo
from extract.repos.batches import from_settings as batch_repo_from_settings

#: The one tenant every row is scoped to.
DEFAULT_CUSTOMER_ID = "default"


@dataclass(frozen=True)
class RequestContext:
    customer_id: str = DEFAULT_CUSTOMER_ID
    phi_safe: bool = False


_CTX = RequestContext()


def batch_repo_dep(request: Request) -> BatchRepo:
    repo = getattr(request.app.state, "batch_repo", None)
    if repo is None:
        repo = batch_repo_from_settings()
        request.app.state.batch_repo = repo
    return repo


def api_signer_dep(request: Request) -> UploadSigner:
    signer = getattr(request.app.state, "upload_signer", None)
    if signer is None:
        signer = upload_signer_from_settings()
        request.app.state.upload_signer = signer
    return signer


# Separate aliases per router so tests can override auth/context per-surface,
# mirroring the hosted service's dependency layout.
async def files_prevalidate_dep() -> RequestContext:
    return _CTX


async def batches_prevalidate_dep() -> RequestContext:
    return _CTX


__all__ = [
    "DEFAULT_CUSTOMER_ID",
    "RequestContext",
    "api_signer_dep",
    "batch_repo_dep",
    "batches_prevalidate_dep",
    "files_prevalidate_dep",
]
