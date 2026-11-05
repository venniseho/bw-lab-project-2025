"""In-memory contour fragmentation with isolated, reproducible random streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from . import _contours


@dataclass(frozen=True)
class FragmentationConfig:
    """Generation defaults from the archive's fragment_one, without I/O options."""

    edge_len: int | None = -1
    grid: int = 40
    gap_factor: float = 0.4
    jitter_deg: int = 0
    noise_per_cell: int = 1
    thickness: int = 1
    noise_mode: str = "uniform"
    noise_count: int = 300
    sep_pad: int = 1
    target_frag_per_100px: float = 6
    outline_mode: str = "scan"
    max_outline_segments: int | None = None
    inside_noise_count: int | None = None
    outside_noise_count: int | None = None
    inside_noise_len: int | None = None
    outside_noise_len: int | None = None


class _RandomStreams:
    """Own independent RNG streams for one stimulus, without mutable globals."""

    def __init__(self, seed):
        self.sequence = None if seed is None else np.random.SeedSequence(seed)

    def __call__(self):
        if self.sequence is None:
            return np.random.default_rng()
        return np.random.default_rng(self.sequence.spawn(1)[0])


def _validate_inputs(
    image_rgb: np.ndarray, mask: np.ndarray, config: FragmentationConfig, seed: int | None
) -> np.ndarray:
    """Validate the array adapter's contract without altering legacy geometry."""
    from core.samples import binary_mask as _binary_mask

    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
        raise ValueError("image must be a uint8 RGB array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or 0 in image_rgb.shape:
        raise ValueError("image must have nonempty (H,W,3) shape")
    binary = _binary_mask(mask, "segmentation")
    if binary.shape != image_rgb.shape[:2]:
        raise ValueError("segmentation must align with image; no implicit resizing")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer or None")
    if config.outline_mode not in ("scan", "random") or config.noise_mode not in (
        "uniform",
        "grid",
    ):
        raise ValueError("unknown historical outline/noise mode")
    if config.thickness < 1 or config.grid < 1 or config.edge_len == 0:
        raise ValueError("thickness/grid must be positive; edge_len cannot be zero")
    if not np.isfinite(config.gap_factor) or config.gap_factor < 0:
        raise ValueError("gap_factor must be finite and nonnegative")
    return binary


def _outline_parameters(mask_u8: np.ndarray, config: FragmentationConfig) -> tuple[int, float]:
    """Resolve the historical automatic dash-length choice."""
    edge_len, gap_factor = config.edge_len, config.gap_factor
    if edge_len is None or edge_len < 0:
        edge_len, gap_factor = _contours.choose_edge_and_gap(
            _contours.approx_perimeter(mask_u8),
            config.target_frag_per_100px,
            min_edge=4,
            max_edge=12,
            gap_factor=gap_factor,
        )
    return edge_len, gap_factor


def _segments(
    mask_u8: np.ndarray, config: FragmentationConfig, edge_len: int, gap_factor: float, rng_factory
) -> tuple[np.ndarray, np.ndarray]:
    """Delegate outline and noise generation to unchanged archive helpers."""
    outline, occupied = _contours._generate_outline_segments(
        mask_u8,
        int(edge_len),
        float(gap_factor),
        int(config.jitter_deg),
        int(config.thickness),
        int(config.sep_pad),
        config.outline_mode,
        config.max_outline_segments,
        rng_factory=rng_factory,
    )
    noise, _ = _contours._generate_noise_segments(
        mask_u8,
        occupied,
        int(edge_len),
        int(config.thickness),
        int(config.sep_pad),
        config.noise_mode,
        int(config.noise_per_cell),
        int(config.grid),
        int(config.noise_count),
        config.inside_noise_count,
        config.outside_noise_count,
        config.inside_noise_len,
        config.outside_noise_len,
        outline,
        rng_factory=rng_factory,
    )
    return outline, noise


def generate_fragmentation(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: FragmentationConfig = FragmentationConfig(),
    seed: int | None = None,
) -> dict:
    """Preserve archive geometry/noise; seed=None keeps historical entropy RNGs.

    A seed opts into a separate reproducible RNG adapter. There is no claim that
    a seed reproduces an unseeded historical image or nests contour severities.
    The input image supplies only canvas shape; original texture is discarded.
    """
    binary = _validate_inputs(image_rgb, mask, config, seed)
    rng_factory = _RandomStreams(seed)
    mask_u8 = binary.astype(np.uint8) * 255
    edge_len, gap_factor = _outline_parameters(mask_u8, config)
    outline, noise = _segments(mask_u8, config, edge_len, gap_factor, rng_factory)
    stimulus = np.zeros_like(image_rgb)
    for segments in (outline, noise):
        _contours.draw_segments(stimulus, segments, thickness=int(config.thickness))
    return {
        "image": stimulus,
        "outline_segments": outline,
        "noise_segments": noise,
        "parameters": asdict(config),
        "edge_len_used": edge_len,
        "gap_factor_used": gap_factor,
        "seed": seed,
        "rng_protocol": "historical_entropy" if seed is None else "spawned_streams_v1",
    }
