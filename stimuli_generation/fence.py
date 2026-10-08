"""Regular black fence occlusion on RGB arrays, without file I/O.

Spacing is the fixed center-to-center distance between bars, in pixels,
measured perpendicular to the bars. It is not the remaining visible gap.
Increasing width keeps centers fixed, so occlusion masks are nested.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class FenceResult(NamedTuple):
    corrupted_rgb: np.ndarray
    occlusion_mask: np.ndarray
    image_occluded_fraction: float
    object_occluded_fraction: float | None


def generate_fence(
    image_rgb: np.ndarray,
    bar_width: float,
    bar_spacing: float,
    bar_angle: float = 90.0,
    seed: int | None = None,
    gt_mask: np.ndarray | None = None,
) -> FenceResult:
    """Cover pixels whose centers fall within regularly spaced fence bars.

    Args:
        image_rgb: Nonempty numeric (H, W, 3) array. Dtype is preserved;
            black occluders have value zero in every channel.
        bar_width: Width in pixels, between zero and bar_spacing inclusive.
        bar_spacing: Positive center-to-center bar distance in pixels.
        bar_angle: Degrees counterclockwise from horizontal in the displayed
            image: 0 is horizontal and 90 is vertical. Angles repeat every 180.
        seed: Optional integer controlling only the fence's spatial phase.
            None fixes a bar center at the top-left pixel center. Reusing the
            same seed and geometry fixes centers across width/severity levels.
        gt_mask: Optional (H, W) object mask; positive values are foreground.
            It is used only to measure coverage and is never modified.

    Returns:
        A tuple with named fields: a new corrupted RGB array, an (H, W)
        boolean mask (True means occluded), the covered image-pixel fraction,
        and the covered foreground-pixel fraction. Object coverage is None
        when no GT is supplied or the GT is empty (undefined denominator).

    Rasterization uses pixel centers and half-open bar intervals. Fractions
    therefore reflect actual discrete pixels, not simply width / spacing.
    This function does not change the inputs or global random state.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    height, width = image_rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("image_rgb must be nonempty")
    if image_rgb.dtype.kind not in "uif":
        raise TypeError("image_rgb must have an integer or floating dtype")

    bar_width, bar_spacing, bar_angle = map(
        float, (bar_width, bar_spacing, bar_angle)
    )
    if not all(map(math.isfinite, (bar_width, bar_spacing, bar_angle))):
        raise ValueError("fence parameters must be finite")
    if bar_spacing <= 0:
        raise ValueError("bar_spacing must be positive")
    if not 0 <= bar_width <= bar_spacing:
        raise ValueError("bar_width must be between zero and bar_spacing")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer or None")

    foreground = None
    if gt_mask is not None:
        if not isinstance(gt_mask, np.ndarray):
            raise TypeError("gt_mask must be a NumPy array")
        if gt_mask.shape != (height, width):
            raise ValueError("gt_mask must match the image's (H, W) dimensions")
        if gt_mask.dtype.kind not in "buif":
            raise TypeError("gt_mask must have a boolean or numeric dtype")
        if gt_mask.dtype.kind == "f" and not np.isfinite(gt_mask).all():
            raise ValueError("gt_mask must contain finite values")
        foreground = gt_mask > 0

    phase = 0.0 if seed is None else np.random.default_rng(seed).uniform(0, bar_spacing)
    angle = bar_angle % 180.0
    if bar_width == 0:
        occlusion = np.zeros((height, width), dtype=bool)
    elif bar_width == bar_spacing:
        occlusion = np.ones((height, width), dtype=bool)
    else:
        # Axis-aligned fences require only a 1-D pattern; oblique fences use
        # broadcasting, without constructing two full coordinate grids.
        if angle == 90.0:
            projection = np.arange(width, dtype=np.float64)[None, :]
        elif angle == 0.0:
            projection = np.arange(height, dtype=np.float64)[:, None]
        else:
            radians = math.radians(angle)
            projection = (
                np.arange(width, dtype=np.float64)[None, :] * math.sin(radians)
                + np.arange(height, dtype=np.float64)[:, None] * math.cos(radians)
            )
        distance = (projection - phase + bar_spacing / 2) % bar_spacing - bar_spacing / 2
        pattern = (distance >= -bar_width / 2) & (distance < bar_width / 2)
        occlusion = np.broadcast_to(pattern, (height, width)).copy()

    corrupted = image_rgb.copy()
    corrupted[occlusion] = 0
    image_fraction = float(np.count_nonzero(occlusion) / occlusion.size)
    object_fraction = None
    if foreground is not None:
        area = np.count_nonzero(foreground)
        if area:
            object_fraction = float(np.count_nonzero(occlusion & foreground) / area)
    return FenceResult(corrupted, occlusion, image_fraction, object_fraction)
