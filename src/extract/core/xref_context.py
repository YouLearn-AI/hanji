"""Cross-engine secondary OCR reads appended to the schema-extraction context.

The extractor only ever sees OUR parse. When our parse garbles an identifier, no schema
change and no prompt can recover it — the characters are simply not in the text the model
reads. An independently-trained recognizer frequently has them: measured on a healthcare-forms customer's
corpus, another engine holds the gold value where our parse does not on 14% of
``member_id`` and ~8% of ``patient_name`` / ``provider_name`` / ``npi``.

This is the single largest lever in campaign 104 — roughly +0.062 of the +0.094 total —
and unlike the schema changes it is pipeline code we own, so it applies to a customer's
EXISTING request schema without asking them to change anything.

The reads are appended as clearly-labelled EXTRA chunks, not merged into the page text, so
the model treats them as a secondary source to reconcile against rather than as more page
content. Bboxes are the full page, because a cross-engine read is evidence about the page,
not about a region — the grounding path will still cite our own chunks.

TRANSPORT: Textract only. It is AWS-BAA covered (mandatory — these pages carry PHI), it
bills per PAGE at $0.0015, and it is a genuinely different pretraining family from our
Qwen parser, which is the whole point (arXiv 2503.15195 measures ~4x lower CER for a
non-Qwen family on handwriting over identical images).
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from extract.config import settings
from extract.core.models import Chunk

log = logging.getLogger(__name__)

#: Chars of secondary text kept per page. The read is redundant with our parse for most of
#: the page; the budget keeps the prompt from doubling for the minority that differs.
_BUDGET = 4000

#: Handwritten WORD blocks on a page before it earns a cross-read. Measured over the
#: 36-page Clyde Health referral corpus: the one cursive fax cover scored 7, the
#: next-highest page scored 1, and 33 of 36 scored 0; over the 363-page production
#: gate corpus (printed EHR output) every page scored 0. Anything under a few words
#: is a signature squiggle or a stray mark, not written content.
_HANDWRITING_WORDS = 3

_LABEL = (
    "[SECONDARY OCR SOURCE — {engine}, an independent recognizer. Use ONLY to recover or "
    "correct a value the primary parse dropped or garbled; prefer the primary parse when "
    "they agree.]\n"
)

#: Prompt for the Gemini second reader. Values only, verbatim, no commentary — the model
#: is a RECOGNIZER here, not an extractor; the extraction call does the reasoning.
_GEMINI_READ_PROMPT = (
    "Transcribe every FILLED-IN value on this form, exactly as written.\n"
    "Rules:\n"
    "- Output only values that were filled in (typed or handwritten), one per line, with "
    "the printed label they sit next to when there is one.\n"
    "- Preserve characters exactly. Do not normalize, reformat or correct.\n"
    "- Do not guess. Output ??? for anything illegible.\n"
    "- No commentary, no numbering."
)


def page_needs_xref(read) -> bool:
    """Whether a page's cross-read earns its cost AND its weight in the prompt.

    The cross-read exists to rescue values our parse garbled, which happens on a
    minority of pages, and it is not free in either currency. Every page it covers
    adds a vendor call, and adds a full page of secondary text to the extraction
    context — measured on a 12-page referral fax, a 28-row problem list spanning a
    page break came back complete with the cross-read scoped to the one page that
    needed it, and truncated to 7 rows with all 12 pages injected.

    ``read`` is a :class:`~extract.core.textract_page.PageRead`. The signal rides
    along on the Textract response we have already paid for, so selection is free.

    HANDWRITING IS THE TRIGGER, and deliberately the only one. Textract's own
    low-confidence rate was tried as a second trigger and dropped: it selects pages
    that are merely *badly scanned*, which is not the same as pages holding a value
    at risk. On the Clyde corpus it added a washed-out insurance card that supplied
    no field the customer asked for, and adding it put the 28-row problem list back
    to truncating (1 of 2 runs, against 5 of 5 complete without it). Handwriting
    means a human entered data in that spot, which is a far better proxy for "a
    value we need, in the form our parser is worst at". ``low_confidence_words``
    stays on the read as a diagnostic; it just does not steer.

    Fail-open: a page we could not assess — Textract errored, or returned no words —
    is cross-read anyway. Only positive evidence that a page is clean, confident
    print buys a skip, because a page skipped in error is a value silently lost.
    """
    if read is None or getattr(read, "error", None) or not getattr(read, "total_words", 0):
        return True
    return read.handwriting_words >= _HANDWRITING_WORDS


def xref_chunks(
    page_images: dict[int, bytes] | None,
    *,
    budget: int = _BUDGET,
    reader=None,
    cache=None,
    page_sizes: dict[int, tuple[int, int]] | None = None,
    always: bool = False,
    second_reader: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> list[Chunk]:
    """Return labelled secondary-source chunks for the pages that need one.

    Which pages those are is decided by :func:`page_needs_xref` off the Textract read.
    An explicit ``reader`` (tests, alternate engines) bypasses selection entirely,
    because there is no Textract response to select on.

    ``always`` (per customer, ``EXTRACT_XREF_ALWAYS_CUSTOMERS``) runs BOTH readers on
    every page instead of only the selected ones. Meant for short documents, where the
    prompt-weight argument behind the gate (a 12-page packet's whole text restated) does
    not apply.

    The second (recognizer) read used to stay behind the handwriting gate even here, on
    the theory that only handwriting justifies its per-page bill. EXP-A measured that
    theory wrong on printed thermal receipts: the recognizer read is the STRONGEST single
    witness there (+0.035 dev micro, CI-strong), on pages the handwriting gate skips
    every time. So ``always`` now widens it too. Cost: +~$0.004/page on top of Textract's
    $0.0015. Latency stays bounded by the read itself — the whole cross-read is prefetched
    concurrently with parse, so it does not serialize behind it.

    ``second_reader=False`` (per request) drops the recognizer read and keeps Textract's,
    without touching ``EXTRACT_XREF_GEMINI_MODEL`` — which is a DEPLOYMENT setting, so
    emptying it to silence the read (what the bench does) silences it for everyone. The
    receipt lane needs exactly that per-lane shape, and it is a COST trade rather than an
    accuracy win: on the sealed receipts-lane stack the empty slot cost ~0.5-0.7 Nelson points
    (the internal measurement records, "TIES in combo with 3-call ... mean −0.5"; SEALED7 0.9554
    → SEALED8 0.9488) to save the read's ~$0.004/page and its latency. Other customers
    keep the read.

    ``should_stop`` is checked BETWEEN PAGES, in both reads. This runs in a worker
    thread on behalf of a request that may already be over (a failed parse, a
    client disconnect), and each page here is a paid call — Textract's $0.0015 and
    the recognizer's $0.00337. Checking once at the top would leave a 20-page
    document buying 19 more reads for an answer nobody will receive. Returns what
    it has: partial chunks are a valid degraded context, and the caller may still
    be alive and want them.

    Fail-open in every direction: unconfigured credentials, a provider error, or an empty
    read all yield fewer chunks (or none), never an exception into extraction.
    """
    if not page_images:
        return []
    if reader is None and cache is None:
        from extract.core.textract_page import TextractPageCache

        cache = TextractPageCache()
    out: list[Chunk] = []
    # Pages that cleared selection, carried to the second reader below so both
    # engines cover exactly the same set.
    selected: dict[int, bytes] = {}
    for page_no, img in sorted(page_images.items()):
        if should_stop is not None and should_stop():
            log.info("xref_context: request ended — stopping before page %s", page_no)
            return out
        read = None
        text = ""
        try:
            if reader is not None:
                text = reader(img)
            else:
                size = (page_sizes or {}).get(page_no, (1000, 1000))
                read = cache.read(img, size)
                text = read.text
        except Exception:  # noqa: BLE001 — a secondary source must never break extraction
            log.warning("xref_context: secondary read failed on page %s", page_no, exc_info=True)
            # Fall through rather than skip: a page we failed to read is exactly the
            # kind the second recognizer may still rescue, and it carries no evidence
            # that it is clean.
        # Selection decides TWO things: whether this page's text is injected, and
        # whether the (billed per page) second recognizer reads it. ``always``
        # overrides BOTH — on a printed receipt the gate never fires, and EXP-A
        # measured the recognizer read as the strongest witness on exactly those pages.
        if reader is not None or always or page_needs_xref(read):
            selected[page_no] = img
        else:
            continue
        if text and text.strip():
            out.append(
                Chunk(
                    page_content=_LABEL.format(engine="textract") + text.strip()[:budget],
                    page_no=page_no,
                    bbox=[0.0, 0.0, 1000.0, 1000.0],
                )
            )

    # SECOND reader. Two independently-trained recognizers make uncorrelated mistakes, and
    # the identifier fields are where that pays: measured over 40 production referral packets,
    # Member ID 88% -> 98% and NPI 86% -> 92%; EXP-A adds +0.035 dev micro (CI-strong) on
    # printed thermal receipts, which is why ``always`` widens this read to every page.
    # Costs $0.00337/page. Runs only when a model is configured, and never blocks
    # extraction on its own failure.
    if reader is None and second_reader and settings.EXTRACT_XREF_GEMINI_MODEL and selected:
        out.extend(_gemini_chunks(selected, budget, should_stop=should_stop))
    log.info(
        "xref_context: %d chunk(s) from %d page(s) (%d selected, always=%s, second_reader=%s)",
        len(out), len(page_images), len(selected), always, second_reader,
    )
    return out


def _gemini_chunks(
    page_images: dict[int, bytes],
    budget: int,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> list[Chunk]:
    """One Vertex read per page from the second recognizer. Fail-open per page,
    and stopped between pages when the request that asked for it is over."""
    model = settings.EXTRACT_XREF_GEMINI_MODEL
    out: list[Chunk] = []
    for page_no, img in sorted(page_images.items()):
        if should_stop is not None and should_stop():
            log.info("xref_context: request ended — %s read stopped before page %s", model, page_no)
            return out
        try:
            text = _gemini_read(img, model)
        except Exception:  # noqa: BLE001 — a secondary source must never break extraction
            log.warning("xref_context: %s read failed on page %s", model, page_no, exc_info=True)
            continue
        if text and text.strip():
            out.append(
                Chunk(
                    page_content=_LABEL.format(engine=model) + text.strip()[:budget],
                    page_no=page_no,
                    bbox=[0.0, 0.0, 1000.0, 1000.0],
                )
            )
    return out


def _gemini_read(img: bytes, model: str) -> str:
    """Vertex only — the BAA-covered Google transport, mandatory for PHI pages.

    Thinking is off: this is transcription, and the reasoning happens in the extraction
    call that consumes this text.
    """
    import json

    from google import genai
    from google.genai import types as gt

    from extract.genai_client import make_genai_client

    client = make_genai_client(genai)
    resp = client.models.generate_content(
        model=model,
        contents=[
            gt.Part.from_bytes(data=img, mime_type="image/png"),
            gt.Part.from_text(text=_GEMINI_READ_PROMPT),
        ],
        config=gt.GenerateContentConfig(
            temperature=0,
            max_output_tokens=8192,
            thinking_config=gt.ThinkingConfig(thinking_budget=0),
        ),
    )
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        # Per-PAGE token accounting for the cost bench — DEBUG for the same reason
        # as the extraction client's: it is one line per page, per request.
        log.debug(
            "xref_recognizer usage: prompt=%s cached=%s output=%s thoughts=%s",
            getattr(um, "prompt_token_count", None),
            getattr(um, "cached_content_token_count", None),
            getattr(um, "candidates_token_count", None),
            getattr(um, "thoughts_token_count", None),
        )
    return (resp.text or "").strip()


