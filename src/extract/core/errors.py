"""Typed exceptions raised by the core pipeline.

The API layer maps these to HTTP codes; the CLI prints them to stderr.
Core never raises ``fastapi.HTTPException`` directly.
"""


class ExtractError(Exception):
    """Base class for all extraction errors."""


class UnsupportedInput(ExtractError):
    """The supplied URL / path / bytes didn't match a supported input kind."""


class DocumentTooLarge(ExtractError):
    """Document bytes exceed ``ExtractRequest.max_size``."""


class PageLimitExceeded(ExtractError):
    """Document page count exceeds ``ExtractRequest.page_limit``."""


class ExtractionFailed(ExtractError):
    """Something in the pipeline failed that callers should see."""


class RemoteFetchError(ExtractionFailed):
    """A ``url``-sourced fetch failed (validation, SSRF-block, or HTTP error).

    ``retryable`` tells a caller with a retry loop (the async batch worker)
    whether trying again is worth it: ``False`` for scheme/host/SSRF
    validation failures and 4xx responses, which fail identically on every
    attempt; ``True`` for 5xx/408/409/429 and network-level errors (timeouts,
    connection resets), which may succeed on a later attempt."""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class UpstreamTimeout(ExtractError):
    """A model provider stopped answering and the request's time budget ran out.

    Distinct from ``ExtractionFailed`` (which is the caller's document or schema)
    — this failure is ours-and-the-provider's, and the caller can retry the exact
    same request later with a different outcome.

    Exists so the request ledger can name it. Before this, an exhausted retry
    loop raised a bare ``TimeoutError`` that matched no entry in
    ``_ERROR_CLASS_CODES`` and fell through to the ``internal_error`` catch-all,
    so provider stalls were indistinguishable from genuine bugs in the metrics
    (2026-07-30: a customer took two 738 s failures and both logged as
    ``internal_error``).

    ``elapsed_s`` / ``attempts`` are carried for the log line; the ledger stores
    only the canonical code."""

    def __init__(self, message: str, *, elapsed_s: float, attempts: int) -> None:
        super().__init__(message)
        self.elapsed_s = elapsed_s
        self.attempts = attempts
