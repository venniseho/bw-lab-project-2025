"""
mask_fragmenter_clean.py
--------------------------------------------------------------
Generate fragmented-contour stimuli from real object masks.

For each image + mask pair:
  • Extract the object's contour
  • Convert it into short line segments with gaps
  • Add background "noise" line segments
  • Save the fragmented result + comparison panel

Requires: numpy, opencv-python
--------------------------------------------------------------
"""

import os
import time
import cv2
import numpy as np

# ==========================================================
# -------------------- Helper functions --------------------
# ==========================================================

def ensure_binary(mask):
    """
    Make a clean binary mask where OBJECT is white (255) and
    background is black (0). If the initial threshold yields
    mostly-white image (likely background), auto-invert.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    uniques = np.unique(mask)
    if uniques.size <= 2 and set(uniques.tolist()).issubset({0, 255}):
        return (mask > 0).astype(np.uint8) * 255

    _, mask = cv2.threshold(mask, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8) * 255

    if (mask.mean() / 255.0) > 0.5:
        mask = 255 - mask
    return mask


def largest_external_contour(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def resample_polyline(poly, step_px=3):
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


def rasterize_line_mask(h, w, x1, y1, x2, y2, thickness=1):
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    return m


def mark_occupied(occ, seg_mask, pad=0):
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * pad + 1, 2 * pad + 1))
        seg_mask = cv2.dilate(seg_mask, k)
    occ = (occ > 0).astype(np.uint8, copy=False)
    seg = (seg_mask > 0).astype(np.uint8, copy=False)
    return np.maximum(occ, seg)


def intersects(occ, seg_mask, pad=0):
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * pad + 1, 2 * pad + 1))
        seg_mask = cv2.dilate(seg_mask, k)
    return np.any((occ > 0) & (seg_mask > 0))


def approx_perimeter(mask_u8):
    cnt = largest_external_contour(mask_u8)
    if cnt is None:
        return 0.0
    return float(cv2.arcLength(cnt, closed=True))


def choose_edge_and_gap(perimeter_px,
                        target_frag_per_100px=6,
                        min_edge=10,
                        max_edge=24,
                        gap_factor=0.35):
    desired = int(round(100.0 / max(1, target_frag_per_100px)))
    edge_len = int(np.clip(desired, min_edge, max_edge))
    return edge_len, gap_factor


# ==========================================================
# --------- Contour → segments (scan & random) -------------
# ==========================================================

def contour_to_segments(pts, edge_len=18, gap_px=(0, 6), jitter_deg=15,
                        shape=None, thickness=1, sep_pad=1,
                        stick_to_contour=False):
    """
    Walk along polyline and break it into short chords.
    - If stick_to_contour=True: NO rotation; segments hug boundary.
    - With jitter: chords are rotated slightly.
    """
    if len(pts) < 2:
        return np.zeros((0, 4), np.float32), np.zeros((1, 1), np.uint8)
    assert shape is not None, "Provide 'shape'=(H,W) for collision checking"

    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    segs, i = [], 0
    rng = np.random.default_rng()

    while i + 1 < len(pts):
        if gap_px and gap_px[1] > 0:
            i = min(i + int(rng.integers(gap_px[0], gap_px[1] + 1)),
                    len(pts) - 2)

        run, j = 0.0, i + 1
        while j < len(pts) and run < edge_len:
            run += np.linalg.norm(pts[j] - pts[j - 1])
            j += 1
        if j >= len(pts):
            break

        p1, p2 = pts[i], pts[j - 1]

        if stick_to_contour or not jitter_deg:
            q1, q2 = p1, p2
        else:
            mid = 0.5 * (p1 + p2)
            v = p2 - p1
            th = np.deg2rad(
                rng.integers(jitter_deg - 5, jitter_deg + 6)
            ) * (1 if rng.random() < 0.5 else -1)
            R = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th),  np.cos(th)]],
                         dtype=np.float32)
            v2 = R @ v
            q1 = mid - 0.5 * v2
            q2 = mid + 0.5 * v2

        seg_mask = rasterize_line_mask(H, W, q1[0], q1[1], q2[0], q2[1],
                                       thickness=thickness)
        if not intersects(occ, seg_mask, pad=sep_pad):
            segs.append([q1[0], q1[1], q2[0], q2[1]])
            occ = mark_occupied(occ, seg_mask, pad=sep_pad)

        i = j

    return np.array(segs, dtype=np.float32), occ


def contour_to_segments_random(
    pts, edge_len=18, max_segments=None, jitter_deg=15,
    shape=None, thickness=1, sep_pad=1, max_tries=2000
):
    """
    Rejection-sampling version:
      - pick random centers along contour
      - orient segment along local tangent (+ jitter)
      - reject if it overlaps with existing segments (with padding)
    """
    if len(pts) < 2:
        return np.zeros((0, 4), np.float32), np.zeros((1, 1), np.uint8)
    assert shape is not None, "Provide 'shape'=(H,W) for collision checking"

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

        th = 0.0
        if jitter_deg:
            th = np.deg2rad(rng.integers(-jitter_deg, jitter_deg + 1))
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]], dtype=np.float32)
        v = R @ t

        q1 = c - half * v
        q2 = c + half * v

        x1, y1 = int(np.clip(q1[0], 0, W - 1)), int(np.clip(q1[1], 0, H - 1))
        x2, y2 = int(np.clip(q2[0], 0, W - 1)), int(np.clip(q2[1], 0, H - 1))

        seg_mask = rasterize_line_mask(H, W, x1, y1, x2, y2,
                                       thickness=thickness)
        if not intersects(occ, seg_mask, pad=sep_pad):
            segs.append([x1, y1, x2, y2])
            occ = mark_occupied(occ, seg_mask, pad=sep_pad)

    return np.array(segs, dtype=np.float32), occ


# ==========================================================
# --------------- Background noise generators --------------
# ==========================================================

def random_noise_segments(h, w, n_per_cell=1, cell=40, length=18,
                          thickness=1, avoid=None, sep_pad=1,
                          occ=None, tries=6):
    """
    Old grid-based background noise (kept for compatibility).
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
                accepted = False
                for _try in range(tries):
                    theta = rng.uniform(0, 2 * np.pi)
                    dx = 0.5 * length * np.cos(theta)
                    dy = 0.5 * length * np.sin(theta)
                    x1, y1 = int(x - dx), int(y - dy)
                    x2, y2 = int(x + dx), int(y + dy)
                    x1 = np.clip(x1, 0, w - 1)
                    x2 = np.clip(x2, 0, w - 1)
                    y1 = np.clip(y1, 0, h - 1)
                    y2 = np.clip(y2, 0, h - 1)

                    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    if avoid is not None and avoid[my, mx] > 0:
                        continue

                    seg_mask = rasterize_line_mask(
                        h, w, x1, y1, x2, y2, thickness=thickness
                    )
                    if not intersects(occ, seg_mask, pad=sep_pad):
                        segs.append([x1, y1, x2, y2])
                        occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                        accepted = True
                        break
                # if not accepted: skip
    return np.array(segs, dtype=np.float32), occ


def random_noise_segments_uniform(h, w, count=300, length=18, thickness=1,
                                  avoid=None, sep_pad=1, occ=None, tries=10,
                                  region="any"):
    """
    Place 'count' random short lines.
    region: "any" | "inside" | "outside" with respect to 'avoid' mask.
    """
    rng = np.random.default_rng()
    segs = []
    if occ is None:
        occ = np.zeros((h, w), np.uint8)
    else:
        occ = (occ > 0).astype(np.uint8, copy=False)

    for _ in range(int(count)):
        accepted = False
        for _t in range(tries):
            cx = rng.integers(0, w)
            cy = rng.integers(0, h)

            if avoid is not None:
                inside = avoid[cy, cx] > 0
                if region == "inside" and not inside:
                    continue
                if region == "outside" and inside:
                    continue

            theta = rng.uniform(0, 2 * np.pi)
            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)
            x1, y1 = int(cx - dx), int(cy - dy)
            x2, y2 = int(cx + dx), int(cy + dy)
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h - 1)

            seg_mask = rasterize_line_mask(h, w, x1, y1, x2, y2,
                                           thickness=thickness)
            if not intersects(occ, seg_mask, pad=sep_pad):
                segs.append([x1, y1, x2, y2])
                occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                accepted = True
                break
        # if not accepted: skip
    return np.array(segs, dtype=np.float32), occ


def draw_segments(canvas, segments, color=(255, 255, 255), thickness=1):
    for x1, y1, x2, y2 in segments.astype(int):
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)


def panel3(orig, frag, mask):
    """
    Simple 1×3 panel: [original | fragmented | mask]
    """
    m3 = 255 - cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([orig, frag, m3])


# ==========================================================
# -------------------- Main pipeline -----------------------
# ==========================================================

def fragment_one(
    image_path,
    mask_path,
    out_dir,
    edge_len=-1,
    grid=40,
    gap_factor=0.4,
    jitter_deg=0,
    noise_per_cell=1,
    thickness=1,
    noise_mode="uniform",       # "uniform" or "grid"
    noise_count=300,            # used when noise_mode="uniform"
    sep_pad=1,
    target_frag_per_100px=6,
    outline_mode="scan",        # "scan" or "random"
    max_outline_segments=None,
    inside_noise_count=None,    # if None -> fall back to noise_count
    outside_noise_count=None,
    inside_noise_len=None,      # if None -> use edge_len
    outside_noise_len=None,
):
    """
    Core entry point: one image + one mask.
    """
    t0 = time.perf_counter()

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    H, W = img.shape[:2]

    mask = ensure_binary(cv2.imread(mask_path, 0))
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # auto density from outline if edge_len < 0
    if edge_len is None or edge_len < 0:
        perim = approx_perimeter(mask)
        edge_len, gap_factor = choose_edge_and_gap(
            perimeter_px=perim,
            target_frag_per_100px=target_frag_per_100px,
            min_edge=10, max_edge=24,
            gap_factor=gap_factor,
        )

    # 1) outline contour → segments
    cnt = largest_external_contour(mask)
    if cnt is None:
        raise RuntimeError(f"no contour found in mask: {mask_path}")

    pts = resample_polyline(cnt, step_px=3)

    if outline_mode == "random":
        contour_segs, occ = contour_to_segments_random(
            pts,
            edge_len=edge_len,
            max_segments=max_outline_segments,
            jitter_deg=jitter_deg,
            shape=(H, W),
            thickness=thickness,
            sep_pad=max(1, sep_pad),
        )
    else:  # "scan"
        contour_segs, occ = contour_to_segments(
            pts,
            edge_len=edge_len,
            gap_px=(0, int(gap_factor * grid)),
            jitter_deg=jitter_deg,
            shape=(H, W),
            thickness=thickness,
            sep_pad=max(1, sep_pad),
            stick_to_contour=(jitter_deg == 0),
        )

    frag = np.zeros_like(img)
    draw_segments(frag, contour_segs,
                  color=(255, 255, 255), thickness=thickness)
    cv2.imwrite(os.path.join(out_dir, f"{name}_outline.png"), frag)

    # 2) background noise
    if noise_mode == "grid":
        noise_segs, occ = random_noise_segments(
            H, W,
            n_per_cell=noise_per_cell,
            cell=grid,
            length=edge_len,
            thickness=thickness,
            avoid=mask,
            sep_pad=max(1, sep_pad),
            occ=occ,
            tries=6,
        )
    else:
        # uniform random; optionally separate inside vs outside
        if inside_noise_count is None and outside_noise_count is None:
            noise_segs, occ = random_noise_segments_uniform(
                H, W,
                count=int(noise_count),
                length=edge_len,
                thickness=thickness,
                avoid=mask,
                sep_pad=max(1, sep_pad),
                occ=occ,
                tries=10,
                region="any",
            )
        else:
            all_noise = []
            # default counts/lengths
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
                    length=inside_noise_len,
                    thickness=thickness,
                    avoid=mask,
                    sep_pad=max(1, sep_pad),
                    occ=occ,
                    tries=10,
                    region="inside",
                )
                all_noise.append(segs_in)

            if outside_noise_count > 0:
                segs_out, occ = random_noise_segments_uniform(
                    H, W,
                    count=int(outside_noise_count),
                    length=outside_noise_len,
                    thickness=thickness,
                    avoid=mask,
                    sep_pad=max(1, sep_pad),
                    occ=occ,
                    tries=10,
                    region="outside",
                )
                all_noise.append(segs_out)

            noise_segs = np.vstack(all_noise) if all_noise else np.zeros((0, 4),
                                                                         np.float32)

    draw_segments(frag, noise_segs,
                  color=(220, 220, 220), thickness=thickness)

    out_frag = os.path.join(out_dir, f"{name}_fragmented.png")
    out_panel = os.path.join(out_dir, f"{name}_panel.png")
    cv2.imwrite(out_frag, frag)
    cv2.imwrite(out_panel, panel3(img, frag, mask))

    t1 = time.perf_counter()
    print(f"saved: {out_frag} and {out_panel}  | "
          f"outline={outline_mode}  | time={t1 - t0:.3f}s")


# ==========================================================
# ----------------------- Script entry ---------------------
# ==========================================================

if __name__ == "__main__":
    IMAGES_DIR = "images"
    MASKS_DIR = "masks"
    OUT_DIR = "output"

    for fname in os.listdir(IMAGES_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        stem = os.path.splitext(fname)[0]
        img_p = os.path.join(IMAGES_DIR, fname)
        msk_p = os.path.join(MASKS_DIR, stem + "_mask.png")
        if not os.path.exists(msk_p):
            print(f"⚠ skip {fname} — no mask found at {msk_p}")
            continue

        fragment_one(
            img_p,
            msk_p,
            OUT_DIR,
            edge_len=-1,                  # auto from perimeter
            target_frag_per_100px=7,      # denser outline
            grid=40,
            gap_factor=0.25,
            jitter_deg=0,                 # 0 → chords stick to contour
            thickness=1,
            noise_mode="uniform",
            noise_count=400,
            sep_pad=1,
            outline_mode="scan",
            max_outline_segments=None,
            inside_noise_count=1100,
            outside_noise_count=600,
            inside_noise_len=12,
            outside_noise_len=16,
        )
