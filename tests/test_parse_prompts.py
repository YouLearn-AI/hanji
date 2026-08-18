"""The parse model is prompt-coupled: the served prompt must stay byte-identical
to the string the published weights were trained and gated on."""

import hashlib

from extract.parse_prompts import (
    PRODUCTION_BBOX_2D_JSON_PROMPT,
    PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE,
)


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def test_production_prompt_body_pinned():
    assert _md5(PRODUCTION_BBOX_2D_JSON_PROMPT) == "89eb909d296d16675d25806a8303e57a"


def test_production_prompt_with_image_pinned():
    assert PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE == (
        "<image>\n" + PRODUCTION_BBOX_2D_JSON_PROMPT
    )
    assert _md5(PRODUCTION_BBOX_2D_JSON_PROMPT_WITH_IMAGE) == "8a9f216d19ae4365506e8777f828a0e7"
