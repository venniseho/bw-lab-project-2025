"""
mask_fragmenter_clean.py
--------------------------------------------------------------
Generate fragmented-contour stimuli from real object masks.

Goal (your experiment constraints)
----------------------------------
We want *no easy local cues* that could let a segmentation model cheat:
  - Outline dashes and background dashes must look identical locally:
      * same color/intensity
      * same thickness
      * same dash length distribution
      * similar orientation distribution (VERY important)
      * similar spacing / density (as much as feasible)

The *only* thing distinguishing "object" is the global arrangement:
  - some dashes lie along a coherent boundary (Gestalt closure cue)

What this file does
-------------------
Given an image + a binary mask (object=white, bg=black):
  1) Extract the outer contour of the mask (largest component)
  2) Convert that contour into short line segments with gaps (dashed outline)
  3) Generate background "noise" segments that match local statistics
  4) Render and save:
      - outline-only image
      - final stimulus image (outline + noise)
      - debug panel (orig | stimulus | mask)
      - metrics JSON + histograms

Requires: numpy, opencv-python, matplotlib (for histogram plots)
--------------------------------------------------------------
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# -------------------- Helper functions --------------------
# ==========================================================

def ensure_binary(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is binary uint8 with:
      object = 255, background = 0.

    We also try to auto-correct "inverted" masks:
      - if thresholding yields mostly-white, likely background is white → invert.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    uniques = np.unique(mask)
    if uniques.size <= 2 and set(uniques.tolist()).issubset({0, 255}):
        return (mask > 0).astype(np.uint8) * 255

    # Otsu to binarize "soft" masks
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8) * 255

    # If more than half the image is white, it's probably inverted.
    if (mask.mean() / 255.0) > 0.5:
        mask = 255 - mask

    return mask


def largest_external_contour(mask_u8: np.ndarray):
    """
    Find external contours and return the largest one by area.
    This is what we'll use for the "object boundary" outline.
    """
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def resample_polyline(poly: np.ndarray, step_px: int = 3) -> np.ndarray:
    """
    Resample a polyline (contour) to roughly equally spaced points.
    This stabilizes dash spacing and reduces sensitivity to contour sampling density.
    """
    pts = poly.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return pts

    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    total = s[-1]
    if total == 0:
        return pts[:1]

    new_s = np.arange(0, total, step_px, dtype=np.float32)
    out, j = [], 0
    for t in new_s:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 >= len(s):
            break
        r = (t - s[j]) / max(1e-6, (s[j + 1] - s[j]))
        p = (1 - r) * pts[j] + r * pts[j + 1]
        out.append(p)

    return np.array(out, dtype=np.float32)


def rasterize_line_mask(
    h: int,
    w: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thickness: int = 1,
) -> np.ndarray:
    """
    Create a binary image for a single segment (used for overlap/collision checking).
    """
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    return m


def mark_occupied(occ: np.ndarray, seg_mask: np.ndarray, pad: int = 0) -> np.ndarray:
    """
    Update an occupancy mask with the newly drawn segment.
    pad > 0 dilates the segment before marking occupied (enforces minimum spacing).
    """
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        seg_mask = cv2.dilate(seg_mask, k)

    occ = (occ > 0).astype(np.uint8, copy=False)
    seg = (seg_mask > 0).astype(np.uint8, copy=False)
    return np.maximum(occ, seg)


def intersects(occ: np.ndarray, seg_mask: np.ndarray, pad: int = 0) -> bool:
    """
    Check if this segment overlaps anything already occupied.
    pad is the same dilation concept as above.
    """
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        seg_mask = cv2.dilate(seg_mask, k)
    return np.any((occ > 0) & (seg_mask > 0))


def approx_perimeter(mask_u8: np.ndarray) -> float:
    """
    Approximate perimeter length of the largest contour.
    Used to choose an appropriate dash length if edge_len is not specified.
    """
    cnt = largest_external_contour(mask_u8)
    if cnt is None:
        return 0.0
    return float(cv2.arcLength(cnt, closed=True))


def choose_edge_and_gap(
    perimeter_px: float,
    target_frag_per_100px: float = 6,
    min_edge: int = 4,
    max_edge: int = 12,
    gap_factor: float = 0.35,
):
    """
    Choose dash length edge_len based on a target number of dashes per 100 pixels of perimeter.

    Note:
      - edge_len controls dash length
      - gap_factor controls typical gap as a fraction of edge_len (NOT grid)
    """
    desired = int(round(100.0 / max(1, target_frag_per_100px)))  # px per dash
    edge_len = int(np.clip(desired, min_edge, max_edge))
    return edge_len, gap_factor


# ==========================================================
# -------------------- Measures / stats --------------------
# ==========================================================

def segment_midpoints(segments: np.ndarray) -> np.ndarray:
    """Return (N,2) midpoints for segments shaped (N,4)."""
    if segments is None or len(segments) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    segs = segments.astype(np.float32)
    xs = 0.5 * (segs[:, 0] + segs[:, 2])
    ys = 0.5 * (segs[:, 1] + segs[:, 3])
    return np.stack([xs, ys], axis=1)


def segment_lengths(segments: np.ndarray) -> np.ndarray:
    """Return (N,) lengths for segments shaped (N,4)."""
    if segments is None or len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    segs = segments.astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    return np.sqrt(dx * dx + dy * dy)


def segment_orientations_deg(segments: np.ndarray) -> np.ndarray:
    """
    Orientation of each segment in degrees, modulo 180.
    (Line direction is symmetric: theta and theta+180 are same.)
    """
    if segments is None or len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    segs = segments.astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))   # [-180, 180]
    angles = np.mod(angles, 180.0)            # [0, 180)
    return angles.astype(np.float32)


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """
    For each point, compute distance to its nearest *other* point.
    Used as a proxy for spacing regularity.
    """
    if points is None or len(points) <= 1:
        return np.zeros((0,), dtype=np.float32)

    pts = points.astype(np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)
    return nn.astype(np.float32)


def save_histogram(data: np.ndarray, bins: int, title: str, xlabel: str, out_path: Path):
    """Save a simple histogram PNG for quick debugging/QA."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    if data is not None and len(data) > 0:
        plt.hist(data, bins=bins)
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_mask_stats(mask_u8: np.ndarray) -> dict:
    """Basic shape stats for sanity-checking mask quality."""
    H, W = mask_u8.shape[:2]
    area = int(np.count_nonzero(mask_u8 > 0))
    area_fraction = float(area) / float(H * W) if H * W > 0 else 0.0
    perim = approx_perimeter(mask_u8)
    apr = float(area) / float(perim) if perim > 1e-6 else 0.0
    return {
        "H": H,
        "W": W,
        "area_px": area,
        "area_fraction": area_fraction,
        "perimeter_px": float(perim),
        "area_perimeter_ratio": apr,
    }


def compute_and_save_metrics(
    contour_segs: np.ndarray,
    noise_segs: np.ndarray,
    mask_u8: np.ndarray,
    frag_canvas: np.ndarray,
    out_dir: Path,
    stem: str,
    metrics_dir: Optional[Path] = None,
):
    """
    Compute and save debug/QA metrics.

    IMPORTANT:
      These are for *debugging & analysis*, not for training inputs.
      Keep these separate from the stimulus images you feed into models.
    """
    H, W = mask_u8.shape[:2]
    img_area = float(H * W) if H > 0 and W > 0 else 1.0

    mp_outline = segment_midpoints(contour_segs)
    mp_noise = segment_midpoints(noise_segs)

    len_outline = segment_lengths(contour_segs)
    len_noise = segment_lengths(noise_segs)

    ori_outline = segment_orientations_deg(contour_segs)
    ori_noise = segment_orientations_deg(noise_segs)

    nn_outline = nearest_neighbor_distances(mp_outline)
    nn_noise = nearest_neighbor_distances(mp_noise)

    outline_density = (len(contour_segs) / img_area) * 10000.0
    noise_density = (len(noise_segs) / img_area) * 10000.0

    mask_stats = compute_mask_stats(mask_u8)

    # coverage: how many pixels got hit by any dash (outline+noise)
    frag_gray = cv2.cvtColor(frag_canvas, cv2.COLOR_BGR2GRAY) if frag_canvas.ndim == 3 else frag_canvas.copy()
    frag_cov_px = int(np.count_nonzero(frag_gray > 0))
    frag_cov_fraction = float(frag_cov_px) / img_area

    mask_bool = mask_u8 > 0
    inside_cov = int(np.count_nonzero((frag_gray > 0) & mask_bool))
    outside_cov = int(np.count_nonzero((frag_gray > 0) & (~mask_bool)))
    inside_cov_fraction = float(inside_cov) / img_area
    outside_cov_fraction = float(outside_cov) / img_area

    metrics_dir = Path(metrics_dir) if metrics_dir is not None else Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "image_size": {"H": H, "W": W, "area_px": img_area},
        "mask_stats": mask_stats,
        "outline": {
            "num_segments": int(len(contour_segs)),
            "density_per_10k_px": outline_density,
            "mean_length": float(len_outline.mean()) if len(len_outline) else 0.0,
            "median_length": float(np.median(len_outline)) if len(len_outline) else 0.0,
            "mean_nn_dist": float(nn_outline.mean()) if len(nn_outline) else 0.0,
            "median_nn_dist": float(np.median(nn_outline)) if len(nn_outline) else 0.0,
        },
        "noise": {
            "num_segments": int(len(noise_segs)),
            "density_per_10k_px": noise_density,
            "mean_length": float(len_noise.mean()) if len(len_noise) else 0.0,
            "median_length": float(np.median(len_noise)) if len(len_noise) else 0.0,
            "mean_nn_dist": float(nn_noise.mean()) if len(nn_noise) else 0.0,
            "median_nn_dist": float(np.median(nn_noise)) if len(nn_noise) else 0.0,
        },
        "coverage": {
            "total_cov_px": frag_cov_px,
            "total_cov_fraction": frag_cov_fraction,
            "inside_cov_px": inside_cov,
            "inside_cov_fraction": inside_cov_fraction,
            "outside_cov_px": outside_cov,
            "outside_cov_fraction": outside_cov_fraction,
        },
    }

    json_path = metrics_dir / f"{stem}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Histograms for distribution-matching debugging
    save_histogram(ori_outline, bins=18, title="Outline orientation (deg)", xlabel="degrees [0,180)", out_path=metrics_dir / f"{stem}_outline_orientation_hist.png")
    save_histogram(ori_noise, bins=18, title="Noise orientation (deg)", xlabel="degrees [0,180)", out_path=metrics_dir / f"{stem}_noise_orientation_hist.png")

    save_histogram(nn_outline, bins=20, title="Outline NN distances (px)", xlabel="pixels", out_path=metrics_dir / f"{stem}_outline_nn_hist.png")
    save_histogram(nn_noise, bins=20, title="Noise NN distances (px)", xlabel="pixels", out_path=metrics_dir / f"{stem}_noise_nn_hist.png")

    save_histogram(len_outline, bins=20, title="Outline segment lengths (px)", xlabel="pixels", out_path=metrics_dir / f"{stem}_outline_length_hist.png")
    save_histogram(len_noise, bins=20, title="Noise segment lengths (px)", xlabel="pixels", out_path=metrics_dir / f"{stem}_noise_length_hist.png")


# ==========================================================
# --------- Contour -> segments (scan & random) -------------
# ==========================================================

def contour_to_segments(
    pts: np.ndarray,
    edge_len: int = 18,
    gap_px: tuple[int, int] = (0, 6),
    jitter_deg: int = 0,
    shape=None,
    thickness: int = 1,
    sep_pad: int = 1,
    stick_to_contour: bool = True,
):
    """
    "Scan" mode:
      Walk along the contour polyline and convert it into dashed chords.

    The key thing for your setup:
      - If jitter_deg==0 and stick_to_contour=True,
        segments will closely follow the boundary (good for stimuli).
    """
    if len(pts) < 2:
        return np.zeros((0, 4), np.float32), np.zeros((1, 1), np.uint8)
    assert shape is not None, "Provide shape=(H,W) for collision checking"

    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    segs, i = [], 0
    rng = np.random.default_rng()

    # We repeatedly:
    #  - skip a random gap
    #  - take a run of length edge_len
    #  - add that dash if it doesn't collide too much
    while i + 1 < len(pts):
        # Randomly skip ahead to create gaps along the contour
        if gap_px and gap_px[1] > 0:
            i = min(i + int(rng.integers(gap_px[0], gap_px[1] + 1)), len(pts) - 2)

        run, j = 0.0, i + 1
        while j < len(pts) and run < edge_len:
            run += np.linalg.norm(pts[j] - pts[j - 1])
            j += 1
        if j >= len(pts):
            break

        p1, p2 = pts[i], pts[j - 1]

        # If no jitter: use the contour chord directly
        if stick_to_contour or not jitter_deg:
            q1, q2 = p1, p2
        else:
            # Optional: slightly rotate dash around its midpoint
            mid = 0.5 * (p1 + p2)
            v = p2 - p1
            th = np.deg2rad(rng.integers(jitter_deg - 5, jitter_deg + 6))
            th *= (1 if rng.random() < 0.5 else -1)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
            v2 = R @ v
            q1 = mid - 0.5 * v2
            q2 = mid + 0.5 * v2

        seg_mask = rasterize_line_mask(H, W, q1[0], q1[1], q2[0], q2[1], thickness=thickness)
        if not intersects(occ, seg_mask, pad=sep_pad):
            segs.append([q1[0], q1[1], q2[0], q2[1]])
            occ = mark_occupied(occ, seg_mask, pad=sep_pad)

        i = j

    return np.array(segs, dtype=np.float32), occ


def contour_to_segments_random(
    pts: np.ndarray,
    edge_len: int = 18,
    max_segments: int | None = None,
    jitter_deg: int = 15,
    shape=None,
    thickness: int = 1,
    sep_pad: int = 1,
    max_tries: int = 2000,
):
    """
    "Random" mode:
      Rejection-sample segments along the contour by picking random contour points
      and orienting segments along local tangent (+ jitter).

    You probably *don't* want this as your main stimulus mode, because
    it can produce less uniform gap structure. Keep for ablations.
    """
    if len(pts) < 2:
        return np.zeros((0, 4), np.float32), np.zeros((1, 1), np.uint8)
    assert shape is not None, "Provide shape=(H,W) for collision checking"

    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    rng = np.random.default_rng()

    d = np.gradient(pts, axis=0)
    tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)

    if max_segments is None:
        perim = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        max_segments = max(1, int(perim / max(edge_len, 1)))

    segs = []
    tries = 0
    half = 0.5 * edge_len

    while len(segs) < max_segments and tries < max_tries:
        tries += 1
        i = rng.integers(0, len(pts))
        c = pts[i]
        t = tang[i]

        th = np.deg2rad(rng.integers(-jitter_deg, jitter_deg + 1)) if jitter_deg else 0.0
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
        v = R @ t

        q1 = c - half * v
        q2 = c + half * v

        x1, y1 = int(np.clip(q1[0], 0, W - 1)), int(np.clip(q1[1], 0, H - 1))
        x2, y2 = int(np.clip(q2[0], 0, W - 1)), int(np.clip(q2[1], 0, H - 1))

        seg_mask = rasterize_line_mask(H, W, x1, y1, x2, y2, thickness=thickness)
        if not intersects(occ, seg_mask, pad=sep_pad):
            segs.append([x1, y1, x2, y2])
            occ = mark_occupied(occ, seg_mask, pad=sep_pad)

    return np.array(segs, dtype=np.float32), occ


# ==========================================================
# --------------- Background noise generators --------------
# ==========================================================

def sample_angles_from_segments(segments: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample 'n' angles (radians) from the orientation distribution of given segments.

    WHY:
      If noise has uniform random orientations but outline has boundary-tangent orientations,
      a model can cheat using orientation cues.

    This makes the *noise* orientation distribution match the *outline* orientation distribution.
    """
    if segments is None or len(segments) == 0:
        return rng.uniform(0, np.pi, size=n)  # fallback (no outline)

    angles_deg = segment_orientations_deg(segments)  # [0, 180)
    sampled_deg = rng.choice(angles_deg, size=n, replace=True)
    return np.deg2rad(sampled_deg)


def random_noise_segments(
    h: int,
    w: int,
    n_per_cell: int = 1,
    cell: int = 40,
    length: int = 18,
    thickness: int = 1,
    avoid=None,
    sep_pad: int = 1,
    occ=None,
    tries: int = 6,
    angle_source_segments: np.ndarray | None = None,
):
    """
    Grid-based noise (legacy).

    NOTE:
      If you want indistinguishable dashes, you should also feed angle_source_segments
      so that orientations match the outline.
    """
    rng = np.random.default_rng()
    segs = []

    if occ is None:
        occ = np.zeros((h, w), np.uint8)
    else:
        occ = (occ > 0).astype(np.uint8, copy=False)

    for y in range(cell // 2, h, cell):
        for x in range(cell // 2, w, cell):
            for _ in range(n_per_cell):
                for _try in range(tries):
                    # Match orientation distribution to outline if provided
                    if angle_source_segments is not None and len(angle_source_segments) > 0:
                        theta = float(sample_angles_from_segments(angle_source_segments, 1, rng)[0])
                        # Expand [0, pi) to [0, 2pi) direction for drawing
                        if rng.random() < 0.5:
                            theta += np.pi
                    else:
                        theta = rng.uniform(0, 2 * np.pi)

                    dx = 0.5 * length * np.cos(theta)
                    dy = 0.5 * length * np.sin(theta)

                    x1, y1 = int(x - dx), int(y - dy)
                    x2, y2 = int(x + dx), int(y + dy)

                    x1 = int(np.clip(x1, 0, w - 1))
                    x2 = int(np.clip(x2, 0, w - 1))
                    y1 = int(np.clip(y1, 0, h - 1))
                    y2 = int(np.clip(y2, 0, h - 1))

                    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    if avoid is not None and avoid[my, mx] > 0:
                        # For background noise, avoid placing inside the object mask
                        continue

                    seg_mask = rasterize_line_mask(h, w, x1, y1, x2, y2, thickness=thickness)
                    if not intersects(occ, seg_mask, pad=sep_pad):
                        segs.append([x1, y1, x2, y2])
                        occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                        break

    return np.array(segs, dtype=np.float32), occ


def random_noise_segments_uniform(
    h: int,
    w: int,
    count: int = 300,
    length: int = 18,
    thickness: int = 1,
    avoid=None,
    sep_pad: int = 1,
    occ=None,
    tries: int = 10,
    region: str = "any",
    angle_source_segments: np.ndarray | None = None,
):
    """
    Uniform random placement of short segments.

    region:
      - "any": place anywhere (except collision constraints)
      - "inside": only inside the avoid mask
      - "outside": only outside the avoid mask

    angle_source_segments:
      - if provided, we sample orientations from those segments
        (critical for matching outline vs noise local statistics).
    """
    rng = np.random.default_rng()
    segs = []

    if occ is None:
        occ = np.zeros((h, w), np.uint8)
    else:
        occ = (occ > 0).astype(np.uint8, copy=False)

    # Pre-sample angles for speed & to ensure we truly match the distribution
    if angle_source_segments is not None and len(angle_source_segments) > 0:
        angles = sample_angles_from_segments(angle_source_segments, int(count), rng)
    else:
        angles = None

    for i in range(int(count)):
        for _t in range(tries):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))

            if avoid is not None:
                inside = avoid[cy, cx] > 0
                if region == "inside" and not inside:
                    continue
                if region == "outside" and inside:
                    continue

            theta = float(angles[i]) if angles is not None else float(rng.uniform(0, np.pi))
            # Expand to both directions to avoid a subtle directional cue
            if rng.random() < 0.5:
                theta += np.pi

            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)

            x1, y1 = int(cx - dx), int(cy - dy)
            x2, y2 = int(cx + dx), int(cy + dy)

            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            seg_mask = rasterize_line_mask(h, w, x1, y1, x2, y2, thickness=thickness)
            if not intersects(occ, seg_mask, pad=sep_pad):
                segs.append([x1, y1, x2, y2])
                occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                break

    return np.array(segs, dtype=np.float32), occ


def draw_segments(canvas: np.ndarray, segments: np.ndarray, color=(255, 255, 255), thickness: int = 1) -> None:
    """Draw (N,4) segments onto the canvas."""
    if segments is None or len(segments) == 0:
        return
    for x1, y1, x2, y2 in segments.astype(int):
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)


def panel3(orig: np.ndarray, frag: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Debug panel ONLY.
    Do NOT feed this into any model/human trials (leaks strong cues).
    """
    m3 = 255 - cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([orig, frag, m3])


# ==========================================================
# -------------------- Main pipeline -----------------------
# ==========================================================

def _load_image_and_mask(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image + mask.
    Mask is resized to the image size (nearest neighbor) and binarized.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    mask = ensure_binary(cv2.imread(mask_path, 0))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask


def _generate_outline_segments(
    mask: np.ndarray,
    edge_len: int,
    gap_factor: float,
    jitter_deg: int,
    thickness: int,
    sep_pad: int,
    outline_mode: str,
    max_outline_segments: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the mask boundary into dashed line segments.

    IMPORTANT change vs your previous version:
      - gap is now tied to edge_len (dash length), not grid.
        That keeps outline/noise local geometry consistent and tunable.
    """
    cnt = largest_external_contour(mask)
    if cnt is None:
        raise RuntimeError("no contour found in mask")

    pts = resample_polyline(cnt, step_px=3)

    if outline_mode == "random":
        return contour_to_segments_random(
            pts,
            edge_len=edge_len,
            max_segments=max_outline_segments,
            jitter_deg=jitter_deg,
            shape=mask.shape[:2],
            thickness=thickness,
            sep_pad=max(1, sep_pad),
        )

    # Gap measured in pixels; scale to edge_len so "dash + gap" stays consistent across images.
    gap_hi = max(1, int(round(gap_factor * edge_len)))
    return contour_to_segments(
        pts,
        edge_len=edge_len,
        gap_px=(0, gap_hi),
        jitter_deg=jitter_deg,
        shape=mask.shape[:2],
        thickness=thickness,
        sep_pad=max(1, sep_pad),
        stick_to_contour=(jitter_deg == 0),
    )


def _generate_noise_segments(
    mask: np.ndarray,
    occ: np.ndarray,
    edge_len: int,
    thickness: int,
    sep_pad: int,
    noise_mode: str,
    noise_per_cell: int,
    grid: int,
    noise_count: int,
    inside_noise_count: int | None,
    outside_noise_count: int | None,
    inside_noise_len: int | None,
    outside_noise_len: int | None,
    contour_segs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate background noise segments.

    CRITICAL for your experiment:
      - use angle_source_segments=contour_segs to match orientation distribution
      - keep length=edge_len so dash length matches outline
      - draw with the same color & thickness later
    """
    H, W = mask.shape[:2]

    if noise_mode == "grid":
        return random_noise_segments(
            H,
            W,
            n_per_cell=noise_per_cell,
            cell=grid,
            length=edge_len,
            thickness=thickness,
            avoid=mask,
            sep_pad=max(1, sep_pad),
            occ=occ,
            tries=6,
            angle_source_segments=contour_segs,
        )

    # Default: put noise anywhere (not restricted to inside/outside),
    # but match orientation distribution to the outline.
    if inside_noise_count is None and outside_noise_count is None:
        return random_noise_segments_uniform(
            H,
            W,
            count=int(noise_count),
            length=edge_len,
            thickness=thickness,
            avoid=mask,
            sep_pad=max(1, sep_pad),
            occ=occ,
            tries=10,
            region="any",
            angle_source_segments=contour_segs,
        )

    # Optional: split noise counts inside vs outside object.
    # NOTE: Be careful—this can introduce subtle cues depending on your setup.
    all_noise = []

    if inside_noise_count is None:
        inside_noise_count = noise_count
    if outside_noise_count is None:
        outside_noise_count = noise_count
    if inside_noise_len is None:
        inside_noise_len = edge_len
    if outside_noise_len is None:
        outside_noise_len = edge_len

    if inside_noise_count > 0:
        segs_in, occ = random_noise_segments_uniform(
            H, W,
            count=int(inside_noise_count),
            length=int(inside_noise_len),
            thickness=thickness,
            avoid=mask,
            sep_pad=max(1, sep_pad),
            occ=occ,
            tries=10,
            region="inside",
            angle_source_segments=contour_segs,
        )
        all_noise.append(segs_in)

    if outside_noise_count > 0:
        segs_out, occ = random_noise_segments_uniform(
            H, W,
            count=int(outside_noise_count),
            length=int(outside_noise_len),
            thickness=thickness,
            avoid=mask,
            sep_pad=max(1, sep_pad),
            occ=occ,
            tries=10,
            region="outside",
            angle_source_segments=contour_segs,
        )
        all_noise.append(segs_out)

    noise_segs = np.vstack(all_noise) if all_noise else np.zeros((0, 4), np.float32)
    return noise_segs, occ


def fragment_one(
    image_path: str,
    mask_path: str,
    out_dir: str,
    output_stem: Optional[str] = None,
    edge_len: int = -1,
    grid: int = 40,
    gap_factor: float = 0.4,
    jitter_deg: int = 0,
    noise_per_cell: int = 1,
    thickness: int = 1,
    noise_mode: str = "uniform",      # "uniform" or "grid"
    noise_count: int = 300,           # used when noise_mode="uniform"
    sep_pad: int = 1,
    target_frag_per_100px: float = 6,
    outline_mode: str = "scan",       # "scan" or "random"
    max_outline_segments=None,
    inside_noise_count=None,
    outside_noise_count=None,
    inside_noise_len=None,
    outside_noise_len=None,
    # New: allow splitting outputs to prevent cue leakage
    stimuli_subdir: str = "fragments",
    debug_subdir: str = "debug",
):
    """
    Core entry point: one image + one mask.

    Output discipline (IMPORTANT):
      - "stimulus" image (outline+noise only) goes in out_dir/<stimuli_subdir>
      - debug images (outline-only + panel + metrics) can go elsewhere

    This helps ensure you don't accidentally train/evaluate on debug panels.
    """
    t0 = time.perf_counter()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Stimuli directory: only what humans/models should see
    stimuli_dir = out_root / stimuli_subdir
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    # Debug directory: outline-only, panels, etc. (do not feed to models)
    debug_dir = out_root / debug_subdir
    outlines_dir = debug_dir / "outlines"
    panels_dir = debug_dir / "panels"
    metrics_dir = debug_dir / "metrics"
    outlines_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    name = output_stem or Path(image_path).stem

    img, mask = _load_image_and_mask(image_path, mask_path)

    # If edge_len not specified, choose based on mask perimeter
    if edge_len is None or edge_len < 0:
        perim = approx_perimeter(mask)
        edge_len, gap_factor = choose_edge_and_gap(
            perimeter_px=perim,
            target_frag_per_100px=target_frag_per_100px,
            min_edge=10,
            max_edge=24,
            gap_factor=gap_factor,
        )

    # 1) outline segments along the object boundary
    contour_segs, occ = _generate_outline_segments(
        mask=mask,
        edge_len=edge_len,
        gap_factor=gap_factor,
        jitter_deg=jitter_deg,
        thickness=thickness,
        sep_pad=sep_pad,
        outline_mode=outline_mode,
        max_outline_segments=max_outline_segments,
    )

    # We render stimulus on a black canvas.
    # NOTE: we intentionally do NOT use the original image in the stimulus.
    stimulus = np.zeros_like(img)

    # Draw outline dashes (white)
    draw_segments(stimulus, contour_segs, color=(255, 255, 255), thickness=thickness)

    # Save outline-only debug view
    cv2.imwrite(str(outlines_dir / f"{name}_outline.png"), stimulus)

    # 2) background noise: match local stats to outline (length/orientation/thickness/color)
    noise_segs, occ = _generate_noise_segments(
        mask=mask,
        occ=occ,
        edge_len=edge_len,
        thickness=thickness,
        sep_pad=sep_pad,
        noise_mode=noise_mode,
        noise_per_cell=noise_per_cell,
        grid=grid,
        noise_count=noise_count,
        inside_noise_count=inside_noise_count,
        outside_noise_count=outside_noise_count,
        inside_noise_len=inside_noise_len,
        outside_noise_len=outside_noise_len,
        contour_segs=contour_segs,
    )

    # Draw noise with IDENTICAL appearance (same color + thickness)
    draw_segments(stimulus, noise_segs, color=(255, 255, 255), thickness=thickness)

    # Save the final stimulus image (this is what you should feed into models/humans)
    out_stimulus = stimuli_dir / f"{name}_fragmented.png"
    cv2.imwrite(str(out_stimulus), stimulus)

    # Debug panel (orig | stimulus | mask) — never use as model input
    out_panel = panels_dir / f"{name}_panel.png"
    cv2.imwrite(str(out_panel), panel3(img, stimulus, mask))

    t1 = time.perf_counter()
    print(
        f"saved stimulus: {out_stimulus} | debug panel: {out_panel} | "
        f"outline_mode={outline_mode} | edge_len={edge_len} | time={t1 - t0:.3f}s"
    )

    # Metrics/histograms (debug/analysis only)
    compute_and_save_metrics(
        contour_segs=contour_segs,
        noise_segs=noise_segs,
        mask_u8=mask,
        frag_canvas=stimulus,
        out_dir=debug_dir,
        stem=name,
        metrics_dir=metrics_dir,
    )


# ==========================================================
# ----------------------- Script entry ---------------------
# ==========================================================

def main():
    """
    Standalone mode for quick local testing:
      expects:
        images/   (input images)
        masks/    (binary masks named <stem>_mask.png)
      writes:
        outputs/mask_fragmenter/
          fragments/ (stimuli only)
          debug/     (outline/panels/metrics)

    This is NOT your COCO pipeline, just a quick sanity harness.
    """
    images_dir = Path("images")
    masks_dir = Path("masks")
    out_dir = Path("outputs") / "mask_fragmenter"

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        stem = Path(fname).stem
        img_p = images_dir / fname
        msk_p = masks_dir / f"{stem}_mask.png"
        if not msk_p.exists():
            print(f"⚠ skip {fname} — no mask found at {msk_p}")
            continue

        fragment_one(
            image_path=str(img_p),
            mask_path=str(msk_p),
            out_dir=str(out_dir),
            edge_len=-1,                 # auto from perimeter
            target_frag_per_100px=7,     # denser outline
            gap_factor=0.25,             # gaps relative to dash length
            jitter_deg=0,                # 0 = hug the contour (best for your goal)
            thickness=1,
            noise_mode="uniform",
            noise_count=400,
            sep_pad=1,
            outline_mode="scan",
            max_outline_segments=None,
            # If you turn these on, be mindful about cue leakage
            inside_noise_count=None,
            outside_noise_count=None,
            inside_noise_len=None,
            outside_noise_len=None,
        )


if __name__ == "__main__":
    main()
