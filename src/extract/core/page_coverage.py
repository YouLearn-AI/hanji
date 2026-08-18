"""One-pass omission detection: did the OCR output COVER the page's ink?

A production silent-failure audit's confirmed silent failures were all OMISSION — well-formed,
confident output that skipped part of the page (numeric recall 0.24-0.75 with
references agreeing at 0.94-0.99), invisible to the health guard (not empty, not
looping, not truncated) and immune to re-reads (the model is deterministic, so a
volatile re-read gate burns latency and a systematic page returns the same wrong
answer every time).

Skipped content has a physical fingerprint available in the SAME pass at ~zero
cost: the rasterized page image (already in memory) contains ink that no predicted
bounding box claims. This module measures that: the fraction of text-like ink
cells not covered by any output box (text blocks + tables + figures — figure
boxes count as covered even when figures aren't returned, so pictures don't
false-alarm). A high uncovered fraction means the model demonstrably did not read
part of the page; the caller treats the read as unusable and the existing per-page
fallback (Gemini on the live path) serves it instead.

Pure numpy + PIL on a coarse grid (fast, speckle-robust, no cv2/scipy); no regex;
thresholds are arguments. Calibrate against the consensus-labeled audit corpus
(target: zero false positives on clean pages) before enabling in prod.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

GRID = 48  # cells per side; coarse enough to be fast + robust, fine enough to localize
INK_CELL_FRACTION = 0.04  # a cell is "inky" when >=4% of its pixels are dark
DARK_RELATIVE = 0.62  # a pixel is "dark" when below 62% of the page's background level


@dataclass(frozen=True)
class CoverageResult:
    """Per-read coverage measurement (all values cheap, PHI-free scalars)."""

    inky_cells: int
    uncovered_cells: int
    uncovered_fraction: float  # uncovered / inky (0.0 when no ink)
    largest_uncovered_block: int  # max count of uncovered cells in any grid row band
    # the same band, normalized by the densest INKY band on the page — width-relative,
    # so narrow-column / half-width pages are measured against themselves rather than
    # an absolute cell count a narrow page can never reach.
    largest_uncovered_band_frac: float


def _ink_grid(image_bytes: bytes):
    """GRID x GRID boolean ink map of the page (True = text-like ink present)."""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    # downscale in two steps: keep enough resolution that thin glyph rows survive
    # the averaging, then pool to the grid.
    small = img.resize((GRID * 8, GRID * 8))
    a = np.asarray(small, dtype=np.float32)
    if float(np.median(a)) < 96.0:  # photo-negative / inverted scan: normalize so
        a = 255.0 - a  # "ink" is always the dark-on-light minority
    background = float(np.percentile(a, 80))  # robust paper-brightness estimate
    if background <= 1.0:  # all-black/degenerate image: nothing measurable
        return np.zeros((GRID, GRID), dtype=bool)
    dark = a < background * DARK_RELATIVE
    # pool 8x8 pixel blocks -> fraction of dark pixels per grid cell
    frac = dark.reshape(GRID, 8, GRID, 8).mean(axis=(1, 3))
    return frac >= INK_CELL_FRACTION


def _covered_grid(boxes, page_w: float, page_h: float):
    """GRID x GRID boolean map of cells touched by any predicted box (with one
    cell of dilation slack — box edges need not be glyph-tight)."""
    import numpy as np

    cov = np.zeros((GRID, GRID), dtype=bool)
    if page_w <= 0 or page_h <= 0:
        return cov
    for box in boxes:
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        cx0 = max(0, int(min(x0, x1) / page_w * GRID) - 1)
        cx1 = min(GRID, int(max(x0, x1) / page_w * GRID) + 2)
        cy0 = max(0, int(min(y0, y1) / page_h * GRID) - 1)
        cy1 = min(GRID, int(max(y0, y1) / page_h * GRID) + 2)
        if cx1 > cx0 and cy1 > cy0:
            cov[cy0:cy1, cx0:cx1] = True
    return cov


def page_result_boxes(
    page_result, *, page_w: float, page_h: float
) -> list[tuple[float, float, float, float]]:
    """Every bbox the provider claimed: text blocks + table regions + figures.
    Figures count as covered even when the request doesn't return them. Malformed
    bboxes (not exactly 4 finite numbers) are SKIPPED, never raised on — this feeds
    a telemetry shadow that must stay fail-open."""
    boxes: list[tuple[float, float, float, float]] = []
    if page_result is None:
        return boxes

    def _clean(bbox) -> tuple[float, float, float, float] | None:
        if not bbox or len(bbox) != 4:
            return None
        try:
            vals = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        # NaN/inf are valid floats but crash the grid int() casts downstream —
        # the stated contract is "exactly 4 FINITE numbers", so enforce it here.
        if not all(math.isfinite(v) for v in vals):
            return None
        return vals

    groups = [page_result.blocks, page_result.tables]
    for group in groups:
        for item in group:
            b = _clean(getattr(item, "bbox", None))
            if b is not None:
                boxes.append(b)
    # Figures count as covered (pictures must not false-alarm) — EXCEPT a
    # near-whole-page figure, which would mask ALL omission (the same
    # whole-page-dump exploit the eval scorer's giant-box guard rejects). A
    # whole-page TEXT block stays legitimate: it carries actual read text.
    for item in page_result.figures:
        b = _clean(getattr(item, "bbox", None))
        if b is None:
            continue
        x0, y0, x1, y1 = b
        if abs(x1 - x0) * abs(y1 - y0) >= 0.90 * (page_w * page_h):
            continue
        boxes.append(b)
    return boxes


def measure_coverage(
    image_bytes: bytes,
    boxes: list[tuple[float, float, float, float]],
    *,
    page_w: float,
    page_h: float,
) -> CoverageResult:
    """Measure how much of the page's ink the predicted boxes fail to claim."""
    import numpy as np

    ink = _ink_grid(image_bytes)
    cov = _covered_grid(boxes, page_w, page_h)
    inky = int(ink.sum())
    if inky == 0:
        return CoverageResult(0, 0, 0.0, 0, 0.0)
    uncovered = ink & ~cov
    n_unc = int(uncovered.sum())
    # contiguity proxy: the densest horizontal band of uncovered cells (a skipped
    # paragraph/table is a block; scattered specks are noise) — max over 4-row bands,
    # and the SAME band normalized by the densest inky band so the signal is
    # width-relative (a narrow page is measured against its own width).
    kernel = np.ones(4, dtype=int)
    unc_rows = uncovered.sum(axis=1)
    ink_rows = ink.sum(axis=1)
    if len(unc_rows) >= 4:
        band = int(max(np.convolve(unc_rows, kernel, mode="valid")))
        ink_band = int(max(np.convolve(ink_rows, kernel, mode="valid")))
    else:
        band, ink_band = n_unc, inky
    band_frac = band / ink_band if ink_band else 0.0
    return CoverageResult(inky, n_unc, n_unc / inky, band, band_frac)


def is_low_coverage(
    result: CoverageResult,
    *,
    max_uncovered_fraction: float,
    min_uncovered_band_frac: float,
) -> bool:
    """True when the read demonstrably skipped a meaningful, contiguous chunk of
    the page's ink — both conditions must hold so scattered marginalia/stamp
    noise cannot trip it. The contiguity condition is WIDTH-RELATIVE (the densest
    uncovered band vs the page's own densest inky band), so narrow-column and
    half-width pages are judged against themselves rather than an absolute cell
    count they structurally cannot reach."""
    return (
        result.uncovered_fraction > max_uncovered_fraction
        and result.largest_uncovered_band_frac >= min_uncovered_band_frac
    )
