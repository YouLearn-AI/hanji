"""Image quality filtering and WebP compression.

Port of reference repo's ``image_filter.py`` + ``image_compression.py``, with
the on-disk ``RejectedImageDebugger`` dropped. Runs entirely in-process; no
cloud dependencies.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass
class ImageFilterConfig:
    min_width: int = 30
    min_height: int = 30
    min_area: int = 5_000
    min_file_size: int = 300
    max_aspect_ratio: float = 20.0
    solid_color_std_threshold: float = 5.0
    min_color_range: int = 50
    min_unique_colors: int = 2
    color_sample_size: int = 1_000
    min_page_coverage_ratio: float = 0.005


@dataclass
class FilterResult:
    passed: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.passed


class ImageQualityFilter:
    """Fast filter that throws out obviously useless images (icons, spacers,
    near-solid rectangles, near-transparent layers, duplicates).
    """

    def __init__(self, config: ImageFilterConfig | None = None) -> None:
        self.config = config or ImageFilterConfig()
        self._seen_xrefs: set[int] = set()

    def check_duplicate(self, xref: int) -> FilterResult:
        if xref in self._seen_xrefs:
            return FilterResult(False, "duplicate_xref")
        self._seen_xrefs.add(xref)
        return FilterResult(True)

    def check_basic_metadata(
        self,
        *,
        width: int,
        height: int,
        file_size: int,
        page_width: float | None = None,
        page_height: float | None = None,
    ) -> FilterResult:
        if width < self.config.min_width or height < self.config.min_height:
            return FilterResult(False, f"dimensions_too_small:{width}x{height}")
        area = width * height
        if area < self.config.min_area:
            return FilterResult(False, f"area_too_small:{area}")
        if file_size < self.config.min_file_size:
            return FilterResult(False, f"file_too_small:{file_size}B")
        aspect = max(width / height, height / width) if height > 0 else float("inf")
        if aspect > self.config.max_aspect_ratio:
            return FilterResult(False, f"extreme_aspect_ratio:{aspect:.1f}")
        if page_width and page_height:
            page_area = page_width * page_height
            coverage = area / page_area if page_area > 0 else 0
            if coverage < self.config.min_page_coverage_ratio:
                return FilterResult(False, f"low_page_coverage:{coverage:.4f}")
        return FilterResult(True)

    def _adaptive_sample_size(self, width: int, height: int) -> int:
        total = width * height
        if total < 10_000:
            return min(total, 100)
        if total < 100_000:
            return 500
        if total < 1_000_000:
            return 750
        return self.config.color_sample_size

    def check_image_content(self, img_bytes: bytes) -> FilterResult:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size

            if img.mode == "RGBA":
                arr = np.array(img)
                alpha = arr[:, :, 3]
                sample = self._adaptive_sample_size(width, height)
                step_h = max(1, height // int(np.sqrt(sample)))
                step_w = max(1, width // int(np.sqrt(sample)))
                sampled_alpha = alpha[::step_h, ::step_w].flatten()
                if float(np.mean(sampled_alpha)) < 25:
                    return FilterResult(False, "mostly_transparent")

            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)

            sample = self._adaptive_sample_size(width, height)
            step_h = max(1, height // int(np.sqrt(sample)))
            step_w = max(1, width // int(np.sqrt(sample)))
            sampled = arr[::step_h, ::step_w].reshape(-1, 3)

            early = sampled[:100]
            unique = set(map(tuple, early))
            if len(unique) < self.config.min_unique_colors:
                for pixel in sampled[100:]:
                    unique.add(tuple(pixel))
                    if len(unique) >= self.config.min_unique_colors:
                        break
                if len(unique) < self.config.min_unique_colors:
                    return FilterResult(False, f"low_color_diversity:{len(unique)}")

            if len(sampled) > 0:
                std_per_ch = np.std(sampled, axis=0)
                max_std = float(np.max(std_per_ch))
                range_per_ch = np.ptp(sampled, axis=0)
                max_range = int(np.max(range_per_ch))
                if (
                    max_std < self.config.solid_color_std_threshold
                    and max_range < self.config.min_color_range
                ):
                    return FilterResult(
                        False, f"near_solid_color:std={max_std:.2f},range={max_range}"
                    )
            return FilterResult(True)
        except Exception:
            # On decode failure, don't block — let the upload path handle it.
            return FilterResult(True)


# ---------------------------------------------------------------------------
# Compression (WebP → 200 KB / 768 px target)
# ---------------------------------------------------------------------------


TARGET_BYTES = 200 * 1024
TARGET_SIDE = 768


def _open_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(im)


def _encode_webp(im: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    if im.mode not in ("RGB", "RGBA", "L", "LA"):
        im = im.convert("RGB")
    if im.mode == "LA":
        im = im.convert("RGBA")
    im.save(buf, format="WEBP", quality=quality, method=2)
    return buf.getvalue()


def compress_image_to_webp(data: bytes) -> tuple[bytes, int, int, int]:
    """Target 200 KB / 768 px. Returns (bytes, width, height, original_size)."""
    original_size = len(data)
    im = _open_image(data)
    if max(im.size) > TARGET_SIDE:
        im = im.copy()
        im.thumbnail((TARGET_SIDE, TARGET_SIDE), Image.BILINEAR)

    for quality in (60, 40):
        out = _encode_webp(im, quality)
        if len(out) <= TARGET_BYTES:
            return out, im.width, im.height, original_size

    im = im.copy()
    im.thumbnail((512, 512), Image.BILINEAR)
    out = _encode_webp(im, 35)
    return out, im.width, im.height, original_size


def should_compress(size_bytes: int, threshold: int = 500 * 1024) -> bool:
    return size_bytes > threshold
