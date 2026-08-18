"""Unit tests for the one-pass ink-coverage omission detector
(extract.core.page_coverage). Synthetic pages built with PIL so ground truth is
exact: ink placed deliberately, boxes covering (or not covering) it, thresholds
exercised on both sides. The real-page calibration lives in the diagnostics; these
pin the geometry/logic."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from extract.core.page_coverage import (
    CoverageResult,
    is_low_coverage,
    measure_coverage,
)


def _low(cov):
    """The geometry suite's explicit sensitivity (NOT the shipped thresholds —
    those are pinned from Settings in test_production_thresholds_*): mechanics
    like dilation slack and speck rejection are asserted at 0.20/0.5."""
    return is_low_coverage(cov, max_uncovered_fraction=0.20, min_uncovered_band_frac=0.5)

W, H = 800, 1000  # synthetic page pixels; page coords use the same scale


def _page(ink_rects: list[tuple[int, int, int, int]]) -> bytes:
    """White page with black filled rectangles (text-block stand-ins)."""
    img = Image.new("L", (W, H), 248)
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in ink_rects:
        d.rectangle([x0, y0, x1, y1], fill=12)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


TOP = (60, 80, 740, 280)  # a big top text block
MID = (60, 400, 740, 560)  # a middle block
BOT = (60, 700, 740, 920)  # a bottom block


def test_fully_covered_page_is_clean():
    img = _page([TOP, MID, BOT])
    cov = measure_coverage(img, [TOP, MID, BOT], page_w=W, page_h=H)
    assert cov.inky_cells > 0
    assert cov.uncovered_fraction < 0.05
    assert not _low(cov)


def test_skipped_region_is_flagged():
    # model "read" top+mid but skipped the bottom block — the audit's omission
    # signature (a contiguous region of unclaimed ink).
    img = _page([TOP, MID, BOT])
    cov = measure_coverage(img, [TOP, MID], page_w=W, page_h=H)
    assert cov.uncovered_fraction > 0.25
    assert cov.largest_uncovered_band_frac >= 0.5
    assert _low(cov)


def test_scattered_specks_do_not_flag():
    # tiny scattered marks (stamps/noise) uncovered, the real blocks covered:
    # fraction stays low AND no contiguous block forms -> never flagged.
    specks = [(x, y, x + 14, y + 10) for x, y in [(100, 350), (600, 620), (380, 960)]]
    img = _page([TOP, BOT, *specks])
    cov = measure_coverage(img, [TOP, BOT], page_w=W, page_h=H)
    assert not _low(cov)


def test_box_edge_slack_tolerated():
    # boxes 12px tighter than the ink on every side — dilation slack must absorb it
    img = _page([TOP, MID])
    tight = [(x0 + 12, y0 + 12, x1 - 12, y1 - 12) for x0, y0, x1, y1 in (TOP, MID)]
    cov = measure_coverage(img, tight, page_w=W, page_h=H)
    assert not _low(cov)


def test_blank_page_not_applicable():
    img = _page([])
    cov = measure_coverage(img, [], page_w=W, page_h=H)
    assert cov.inky_cells == 0
    assert cov.uncovered_fraction == 0.0
    assert not _low(cov)


def test_degenerate_all_black_page_does_not_flag():
    img = Image.new("L", (W, H), 0)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    cov = measure_coverage(buf.getvalue(), [], page_w=W, page_h=H)
    assert not _low(cov)  # nothing measurable, fail open


def test_empty_output_on_inky_page_flags():
    # the model claimed NOTHING on a page full of ink (the audit's empty class
    # is guard-caught, but a near-empty claim set must also trip coverage)
    img = _page([TOP, MID, BOT])
    cov = measure_coverage(img, [], page_w=W, page_h=H)
    assert cov.uncovered_fraction > 0.9
    assert _low(cov)


def test_zero_page_dims_fail_open():
    img = _page([TOP])
    cov = measure_coverage(img, [TOP], page_w=0, page_h=0)
    # boxes can't be mapped -> everything uncovered; the CALLER must guard dims,
    # but the function itself must not crash
    assert isinstance(cov, CoverageResult)


def test_thresholds_are_honored():
    img = _page([TOP, MID, BOT])
    cov = measure_coverage(img, [TOP, MID], page_w=W, page_h=H)
    assert is_low_coverage(cov, max_uncovered_fraction=0.10, min_uncovered_band_frac=0.2)
    assert not is_low_coverage(cov, max_uncovered_fraction=0.95, min_uncovered_band_frac=0.2)


# --- adversarial-workflow regression pins (commit d6d304a review) -------------


def test_malformed_bboxes_are_skipped_never_raised():
    # the shadow feeds on provider output; a malformed bbox (2 elements, None,
    # non-numeric) must be SKIPPED — an exception here killed the whole document
    # before the fail-open fix.
    class _B:
        def __init__(self, bbox):
            self.bbox = bbox

    class _R:
        blocks = [
            _B([10.0, 10.0]),
            _B(None),
            _B(["x", 1, 2, 3]),
            _B([float("nan"), 10, 50, 50]),
            _B([10, 10, float("inf"), 50]),
            _B([1, 2, 3, 4]),
        ]
        tables: list = []
        figures: list = []

    from extract.core.page_coverage import page_result_boxes

    boxes = page_result_boxes(_R(), page_w=W, page_h=H)
    assert boxes == [(1.0, 2.0, 3.0, 4.0)]  # only the well-formed box survives
    img = _page([TOP])
    cov = measure_coverage(img, [(10.0, 10.0)], page_w=W, page_h=H)  # 2-tuple input
    assert isinstance(cov, CoverageResult)  # skipped, not raised


def test_giant_figure_does_not_mask_omission():
    # a near-whole-page FIGURE box (the masking exploit) claims nothing; a
    # whole-page TEXT block stays legitimate (it carries actual read text).
    class _F:
        bbox = [0, 0, W, H]

    class _R:
        blocks: list = []
        tables: list = []
        figures = [_F()]

    from extract.core.page_coverage import page_result_boxes

    assert page_result_boxes(_R(), page_w=W, page_h=H) == []  # giant figure dropped
    img = _page([TOP, MID, BOT])
    cov = measure_coverage(img, [], page_w=W, page_h=H)
    assert _low(cov)  # nothing claimed -> omission visible


def test_inverted_page_measured_correctly():
    # photo-negative scan: light text on dark background must read as the SAME
    # ink layout, not as a page of solid ink.
    import io as _io

    from PIL import Image, ImageDraw

    img = Image.new("L", (W, H), 10)  # dark background
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in (TOP, MID):
        d.rectangle([x0, y0, x1, y1], fill=245)  # light "text" blocks
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    cov = measure_coverage(buf.getvalue(), [TOP, MID], page_w=W, page_h=H)
    assert not _low(cov)  # covered ink -> clean, not all-ink chaos
    cov2 = measure_coverage(buf.getvalue(), [TOP], page_w=W, page_h=H)
    assert cov2.uncovered_fraction > 0.25  # the skipped block is still visible


def test_production_thresholds_on_calibration_geometry():
    # finding #13: pin the PRODUCTION thresholds (0.40 / band 0.5) on geometry
    # shaped like the confirmed silent docs (roughly half the page's ink skipped
    # as one contiguous region) vs a clean page with scattered unclaimed specks.
    from extract.config import Settings

    cfg = Settings()
    skipped_half = _page([TOP, MID, BOT])
    cov_bad = measure_coverage(skipped_half, [TOP], page_w=W, page_h=H)  # ~2/3 skipped
    assert is_low_coverage(
        cov_bad,
        max_uncovered_fraction=cfg.OCR_COVERAGE_MAX_UNCOVERED,
        min_uncovered_band_frac=cfg.OCR_COVERAGE_MIN_BAND_FRAC,
    )
    specks = [(x, y, x + 14, y + 10) for x, y in [(100, 350), (600, 620), (380, 960)]]
    clean = _page([TOP, MID, BOT, *specks])
    cov_ok = measure_coverage(clean, [TOP, MID, BOT], page_w=W, page_h=H)
    assert not is_low_coverage(
        cov_ok,
        max_uncovered_fraction=cfg.OCR_COVERAGE_MAX_UNCOVERED,
        min_uncovered_band_frac=cfg.OCR_COVERAGE_MIN_BAND_FRAC,
    )


def test_narrow_page_omission_detectable():
    # finding #11: a half-width page's skipped column must be catchable — the
    # contiguity condition is relative to the page's own densest band, not an
    # absolute cell count narrow pages cannot reach.
    narrow_top = (300, 80, 500, 280)
    narrow_bot = (300, 700, 500, 920)
    img = _page([narrow_top, narrow_bot])
    cov = measure_coverage(img, [narrow_top], page_w=W, page_h=H)
    assert cov.largest_uncovered_band_frac >= 0.5  # relative band sees it
    assert _low(cov)


# --- figure-guard KEEP-side pins (review: only the DROP side was tested; a bug
# dropping ALL figures would have passed every test above) ---------------------


FIG = (250, 600, 550, 900)  # 300x300 picture region — ~11% of page area


def test_normal_figure_counts_as_covered():
    # a normal mid-size figure (well under the 0.90 guard) must be KEPT by
    # page_result_boxes and claim its ink, so pictures don't false-alarm.
    class _B:
        def __init__(self, bbox):
            self.bbox = bbox

    class _R:
        blocks = [_B(list(TOP)), _B(list(MID))]
        tables: list = []
        figures = [_B(list(FIG))]

    from extract.core.page_coverage import page_result_boxes

    boxes = page_result_boxes(_R(), page_w=W, page_h=H)
    assert (250.0, 600.0, 550.0, 900.0) in boxes  # the figure SURVIVES the guard
    assert len(boxes) == 3  # text blocks kept too
    img = _page([TOP, MID, FIG])
    cov = measure_coverage(img, boxes, page_w=W, page_h=H)
    assert not _low(cov)  # picture's ink is covered -> no false alarm


def test_figure_guard_boundary():
    # the guard is >= 0.90 of page area: a figure at exactly 0.89 is KEPT,
    # one at 0.95 is DROPPED.
    class _F:
        def __init__(self, bbox):
            self.bbox = bbox

    class _R:
        blocks: list = []
        tables: list = []

        def __init__(self, figures):
            self.figures = figures

    from extract.core.page_coverage import page_result_boxes

    kept = _F([0, 0, 712, H])  # 712*1000 / 800*1000 = 0.89 of page area
    dropped = _F([0, 0, 760, H])  # 760*1000 / 800*1000 = 0.95 of page area
    assert page_result_boxes(_R([kept]), page_w=W, page_h=H) == [(0.0, 0.0, 712.0, 1000.0)]
    assert page_result_boxes(_R([dropped]), page_w=W, page_h=H) == []
