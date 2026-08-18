"""Fail-closed legal-document post-processing over final OCR page reads."""


def apply_legal_postprocess(*args, **kwargs):
    """Lazy import keeps the pure transform modules independently testable."""
    from extract.core.legal_postprocess.runtime import apply_legal_postprocess as apply

    return apply(*args, **kwargs)


__all__ = ["apply_legal_postprocess"]
