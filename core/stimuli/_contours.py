"""Contour and noise geometry, extracted from the frozen 2025 algorithm.

No file I/O, plotting or global RNG mutation. The original is retained in
tests/fixtures/contour_reference.py for pixel-regression checks.
"""

import cv2
import numpy as np

DASH_LEN_MIN = 4
DASH_LEN_MAX = 12


def largest_external_contour(mask_u8: np.ndarray):
    """Find external contours and return the largest one by area."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def resample_polyline(poly: np.ndarray, step_px: int = 3) -> np.ndarray:
    """Resample a polyline (contour) to roughly equally spaced points."""
    pts = poly.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return pts
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    total = s[-1]
    if total == 0:
        return pts[:1]
    new_s = np.arange(0, total, step_px, dtype=np.float32)
    out, j = ([], 0)
    for t in new_s:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 >= len(s):
            break
        r = (t - s[j]) / max(1e-06, s[j + 1] - s[j])
        p = (1 - r) * pts[j] + r * pts[j + 1]
        out.append(p)
    return np.array(out, dtype=np.float32)


def rasterize_line_mask(
    h: int, w: int, x1: float, y1: float, x2: float, y2: float, thickness: int = 1
) -> np.ndarray:
    """Create a binary image for a single segment (used for overlap/collision checking)."""
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    return m


def mark_occupied(occ: np.ndarray, seg_mask: np.ndarray, pad: int = 0) -> np.ndarray:
    """Update an occupancy mask with the newly drawn segment."""
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        seg_mask = cv2.dilate(seg_mask, k)
    occ = (occ > 0).astype(np.uint8, copy=False)
    seg = (seg_mask > 0).astype(np.uint8, copy=False)
    return np.maximum(occ, seg)


def intersects(occ: np.ndarray, seg_mask: np.ndarray, pad: int = 0) -> bool:
    """Check if this segment overlaps anything already occupied."""
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        seg_mask = cv2.dilate(seg_mask, k)
    return np.any((occ > 0) & (seg_mask > 0))


def approx_perimeter(mask_u8: np.ndarray) -> float:
    """Approximate perimeter length of the largest contour."""
    cnt = largest_external_contour(mask_u8)
    if cnt is None:
        return 0.0
    return float(cv2.arcLength(cnt, closed=True))


def choose_edge_and_gap(
    perimeter_px: float,
    target_frag_per_100px: float = 6,
    min_edge: int = DASH_LEN_MIN,
    max_edge: int = DASH_LEN_MAX,
    gap_factor: float = 0.35,
):
    """Choose dash length edge_len based on a target number of dashes per 100 pixels of perimeter."""
    desired = int(round(100.0 / max(1, target_frag_per_100px)))
    edge_len = int(np.clip(desired, min_edge, max_edge))
    return (edge_len, gap_factor)


def segment_orientations_deg(segments: np.ndarray) -> np.ndarray:
    """Orientation of each segment in degrees, modulo 180."""
    if segments is None or len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    segs = segments.astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))
    angles = np.mod(angles, 180.0)
    return angles.astype(np.float32)


def contour_to_segments(
    pts: np.ndarray,
    edge_len: int = 18,
    gap_px: tuple[int, int] = (0, 6),
    jitter_deg: int = 0,
    shape=None,
    thickness: int = 1,
    sep_pad: int = 1,
    stick_to_contour: bool = True,
    *,
    rng_factory=None,
):
    """ "Scan" mode:"""
    if len(pts) < 2:
        return (np.zeros((0, 4), np.float32), np.zeros((1, 1), np.uint8))
    assert shape is not None, "Provide shape=(H,W) for collision checking"
    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    segs, i = ([], 0)
    rng = (rng_factory or np.random.default_rng)()
    while i + 1 < len(pts):
        if gap_px and gap_px[1] > 0:
            i = min(i + int(rng.integers(gap_px[0], gap_px[1] + 1)), len(pts) - 2)
        run, j = (0.0, i + 1)
        while j < len(pts) and run < edge_len:
            run += np.linalg.norm(pts[j] - pts[j - 1])
            j += 1
        if j >= len(pts):
            break
        p1, p2 = (pts[i], pts[j - 1])
        if stick_to_contour or not jitter_deg:
            q1, q2 = (p1, p2)
        else:
            mid = 0.5 * (p1 + p2)
            v = p2 - p1
            th = np.deg2rad(rng.integers(jitter_deg - 5, jitter_deg + 6))
            th *= 1 if rng.random() < 0.5 else -1
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
            v2 = R @ v
            q1 = mid - 0.5 * v2
            q2 = mid + 0.5 * v2
        seg_mask = rasterize_line_mask(H, W, q1[0], q1[1], q2[0], q2[1], thickness=thickness)
        if not intersects(occ, seg_mask, pad=sep_pad):
            segs.append([q1[0], q1[1], q2[0], q2[1]])
            occ = mark_occupied(occ, seg_mask, pad=sep_pad)
        i = j
    return (np.array(segs, dtype=np.float32), occ)


def contour_to_segments_random(
    pts: np.ndarray,
    edge_len: int = 18,
    max_segments: int | None = None,
    jitter_deg: int = 15,
    shape=None,
    thickness: int = 1,
    sep_pad: int = 1,
    max_tries: int = 2000,
    *,
    rng_factory=None,
):
    """ "Random" mode:"""
    if len(pts) < 2:
        return (np.zeros((0, 4), np.float32), np.zeros((1, 1), np.uint8))
    assert shape is not None, "Provide shape=(H,W) for collision checking"
    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    rng = (rng_factory or np.random.default_rng)()
    d = np.gradient(pts, axis=0)
    tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-06)
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
        x1, y1 = (int(np.clip(q1[0], 0, W - 1)), int(np.clip(q1[1], 0, H - 1)))
        x2, y2 = (int(np.clip(q2[0], 0, W - 1)), int(np.clip(q2[1], 0, H - 1)))
        seg_mask = rasterize_line_mask(H, W, x1, y1, x2, y2, thickness=thickness)
        if not intersects(occ, seg_mask, pad=sep_pad):
            segs.append([x1, y1, x2, y2])
            occ = mark_occupied(occ, seg_mask, pad=sep_pad)
    return (np.array(segs, dtype=np.float32), occ)


def sample_angles_from_segments(
    segments: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample 'n' angles (radians) from the orientation distribution of given segments."""
    if segments is None or len(segments) == 0:
        return rng.uniform(0, np.pi, size=n)
    angles_deg = segment_orientations_deg(segments)
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
    *,
    rng_factory=None,
):
    """Grid-based noise (legacy)."""
    rng = (rng_factory or np.random.default_rng)()
    segs = []
    if occ is None:
        occ = np.zeros((h, w), np.uint8)
    else:
        occ = (occ > 0).astype(np.uint8, copy=False)
    for y in range(cell // 2, h, cell):
        for x in range(cell // 2, w, cell):
            for _ in range(n_per_cell):
                for _try in range(tries):
                    if angle_source_segments is not None and len(angle_source_segments) > 0:
                        theta = float(sample_angles_from_segments(angle_source_segments, 1, rng)[0])
                        if rng.random() < 0.5:
                            theta += np.pi
                    else:
                        theta = rng.uniform(0, 2 * np.pi)
                    dx = 0.5 * length * np.cos(theta)
                    dy = 0.5 * length * np.sin(theta)
                    x1, y1 = (int(x - dx), int(y - dy))
                    x2, y2 = (int(x + dx), int(y + dy))
                    x1 = int(np.clip(x1, 0, w - 1))
                    x2 = int(np.clip(x2, 0, w - 1))
                    y1 = int(np.clip(y1, 0, h - 1))
                    y2 = int(np.clip(y2, 0, h - 1))
                    mx, my = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    if avoid is not None and avoid[my, mx] > 0:
                        continue
                    seg_mask = rasterize_line_mask(h, w, x1, y1, x2, y2, thickness=thickness)
                    if not intersects(occ, seg_mask, pad=sep_pad):
                        segs.append([x1, y1, x2, y2])
                        occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                        break
    return (np.array(segs, dtype=np.float32), occ)


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
    *,
    rng_factory=None,
):
    """Uniform random placement of short segments."""
    rng = (rng_factory or np.random.default_rng)()
    segs = []
    if occ is None:
        occ = np.zeros((h, w), np.uint8)
    else:
        occ = (occ > 0).astype(np.uint8, copy=False)
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
                if region == "inside" and (not inside):
                    continue
                if region == "outside" and inside:
                    continue
            theta = float(angles[i]) if angles is not None else float(rng.uniform(0, np.pi))
            if rng.random() < 0.5:
                theta += np.pi
            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)
            x1, y1 = (int(cx - dx), int(cy - dy))
            x2, y2 = (int(cx + dx), int(cy + dy))
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            seg_mask = rasterize_line_mask(h, w, x1, y1, x2, y2, thickness=thickness)
            if not intersects(occ, seg_mask, pad=sep_pad):
                segs.append([x1, y1, x2, y2])
                occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                break
    return (np.array(segs, dtype=np.float32), occ)


def draw_segments(
    canvas: np.ndarray, segments: np.ndarray, color=(255, 255, 255), thickness: int = 1
) -> None:
    """Draw (N,4) segments onto the canvas."""
    if segments is None or len(segments) == 0:
        return
    for x1, y1, x2, y2 in segments.astype(int):
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)


def _generate_outline_segments(
    mask: np.ndarray,
    edge_len: int,
    gap_factor: float,
    jitter_deg: int,
    thickness: int,
    sep_pad: int,
    outline_mode: str,
    max_outline_segments: int | None,
    *,
    rng_factory=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the mask boundary into dashed line segments."""
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
            rng_factory=rng_factory,
        )
    gap_hi = max(1, int(round(gap_factor * edge_len)))
    return contour_to_segments(
        pts,
        edge_len=edge_len,
        gap_px=(0, gap_hi),
        jitter_deg=jitter_deg,
        shape=mask.shape[:2],
        thickness=thickness,
        sep_pad=max(1, sep_pad),
        stick_to_contour=jitter_deg == 0,
        rng_factory=rng_factory,
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
    *,
    rng_factory=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noise in the historical order with shared rendering constraints."""
    height, width = mask.shape[:2]
    options = dict(
        thickness=thickness,
        avoid=mask,
        sep_pad=max(1, sep_pad),
        angle_source_segments=contour_segs,
        rng_factory=rng_factory,
    )
    if noise_mode == "grid":
        return random_noise_segments(
            height,
            width,
            n_per_cell=noise_per_cell,
            cell=grid,
            length=edge_len,
            occ=occ,
            tries=6,
            **options,
        )
    if inside_noise_count is None and outside_noise_count is None:
        return random_noise_segments_uniform(
            height, width, count=int(noise_count), length=edge_len, occ=occ, region="any", **options
        )
    regions = [
        (
            "inside",
            noise_count if inside_noise_count is None else inside_noise_count,
            edge_len if inside_noise_len is None else inside_noise_len,
        ),
        (
            "outside",
            noise_count if outside_noise_count is None else outside_noise_count,
            edge_len if outside_noise_len is None else outside_noise_len,
        ),
    ]
    all_noise = []
    for region, count, length in regions:
        if count > 0:
            segments, occ = random_noise_segments_uniform(
                height,
                width,
                count=int(count),
                length=int(length),
                occ=occ,
                region=region,
                **options,
            )
            all_noise.append(segments)
    noise = np.vstack(all_noise) if all_noise else np.zeros((0, 4), np.float32)
    return (noise, occ)
