"""OCR provider plug-in system.

Providers implement ``OCRProvider`` (``base.py``) and register themselves
via ``register()``. Lookups use ``get_provider(name)`` — this is the only
call site the extractor needs to know about.
"""

from extract.core.ocr.base import OCRBlock, OCRProvider

REGISTRY: dict[str, OCRProvider] = {}


def register(provider: OCRProvider) -> None:
    REGISTRY[provider.name] = provider


def get_provider(name: str) -> OCRProvider | None:
    if name in REGISTRY:
        return REGISTRY[name]
    # Lazy-load built-in providers on first request.
    if name == "qwen_lora":
        from extract.core.ocr.qwen_lora import QwenLoraProvider

        provider = QwenLoraProvider()
        register(provider)
        return provider
    if name == "gemini":
        from extract.core.ocr.gemini import GeminiOcrProvider

        provider = GeminiOcrProvider()
        register(provider)
        return provider
    return None


__all__ = ["OCRBlock", "OCRProvider", "REGISTRY", "register", "get_provider"]
