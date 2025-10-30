# mask_fragmenter_clean.py  (patched)
import os
import cv2
import numpy as np

# -------------------- Helpers --------------------

def ensure_binary(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    uniques = np.unique(mask)
    if uniques.size <= 2 and set(uniques.tolist()).issubset({0, 255}):
        return (mask > 0).astype(np.uint8) * 255
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8) * 255
    if (mask.mean()/255.0) > 0.5:
        mask = 255 - mask
    return mask

def largest_external_contour(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def resample_polyline(poly, step_px=3):
    pts = poly.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2: return pts
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    total = s[-1]
    if total == 0: return pts[:1]
    new_s = np.arange(0, total, step_px, dtype=np.float32)
    out, j = [], 0
    for t in new_s:
        while j + 1 < len(s) and s[j + 1] < t: j += 1
        if j + 1 >= len(s): break
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
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad+1, 2*pad+1))
        seg_mask = cv2.dilate(seg_mask, k)
    occ = (occ > 0).astype(np.uint8, copy=False)
    seg = (seg_mask > 0).astype(np.uint8, copy=False)
    return np.maximum(occ, seg)

def intersects(occ, seg_mask, pad=0):
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad+1, 2*pad+1))
        seg_mask = cv2.dilate(seg_mask, k)
    return np.any((occ > 0) & (seg_mask > 0))

def approx_perimeter(mask_u8):
    cnt = largest_external_contour(mask_u8)
    if cnt is None: return 0.0
    return float(cv2.arcLength(cnt, closed=True))

def choose_edge_and_gap(perimeter_px, target_frag_per_100px=6,
                        min_edge=10, max_edge=24, gap_factor=0.35):
    desired = int(round(100.0 / max(1, target_frag_per_100px)))
    edge_len = int(np.clip(desired, min_edge, max_edge))
    return edge_len, gap_factor

# -------------------- Key change #1: strict outline splitter --------------------

def contour_to_segments(pts, edge_len=18, gap_px=(0, 6), jitter_deg=15,
                        shape=None, thickness=1, sep_pad=1, stick_to_contour=False):
    """
    When stick_to_contour=True:
      - NO rotation/jitter
      - segments are chords along the original polyline, so they hug the outline.
    Otherwise: previous behavior (optional small jitter).
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
            i = min(i + int(rng.integers(gap_px[0], gap_px[1] + 1)), len(pts) - 2)
        run, j = 0.0, i + 1
        while j < len(pts) and run < edge_len:
            run += np.linalg.norm(pts[j] - pts[j - 1]); j += 1
        if j >= len(pts): break
        p1, p2 = pts[i], pts[j - 1]

        if stick_to_contour or not jitter_deg:
            q1, q2 = p1, p2  # exact chord on the boundary
        else:
            mid = 0.5 * (p1 + p2)
            v = p2 - p1
            th = np.deg2rad(rng.integers(jitter_deg - 5, jitter_deg + 6)) * (1 if rng.random() < 0.5 else -1)
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

# -------------------- Key change #2: uniform noise, fill inside/outside --------------------

def random_noise_segments_uniform(h, w, count=300, length=18, thickness=1,
                                  avoid=None, sep_pad=1, occ=None, tries=10,
                                  region="any"):
    """
    region: "any" | "inside" | "outside"
      - uses 'avoid' (binary mask, object=255) to bias sampling.
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
            cx = rng.integers(0, w); cy = rng.integers(0, h)
            if avoid is not None:
                inside = avoid[cy, cx] > 0
                if region == "inside" and not inside: continue
                if region == "outside" and inside:    continue
            theta = rng.uniform(0, 2*np.pi)
            dx = 0.5 * length * np.cos(theta); dy = 0.5 * length * np.sin(theta)
            x1, y1 = int(cx - dx), int(cy - dy)
            x2, y2 = int(cx + dx), int(cy + dy)
            x1 = np.clip(x1, 0, w-1); x2 = np.clip(x2, 0, w-1)
            y1 = np.clip(y1, 0, h-1); y2 = np.clip(y2, 0, h-1)
            seg_mask = rasterize_line_mask(h, w, x1, y1, x2, y2, thickness=thickness)
            if not intersects(occ, seg_mask, pad=sep_pad):
                segs.append([x1, y1, x2, y2])
                occ = mark_occupied(occ, seg_mask, pad=sep_pad)
                accepted = True
                break
        # if not accepted -> skip
    return np.array(segs, dtype=np.float32), occ

def draw_segments(canvas, segments, color=(255, 255, 255), thickness=1):
    for x1, y1, x2, y2 in segments.astype(int):
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

def panel3(orig, frag, mask):
    m3 = 255 - cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([orig, frag, m3])

# -------------------- Main pipeline --------------------

def fragment_one(image_path, mask_path, out_dir,
                 edge_len=-1, grid=40, gap_factor=0.35, jitter_deg=0,
                 noise_per_cell=1, thickness=1,
                 noise_mode="uniform", noise_count=300, sep_pad=1,
                 target_frag_per_100px=6,
                 outline_strict=True,
                 inside_noise_count=800, outside_noise_count=400,
                 inside_noise_len=12, outside_noise_len=16):
    """
    outline_strict=True => segments exactly follow the contour (no rotation).
    inside_noise_* / outside_noise_* control density + size of random fragments.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"cannot read image: {image_path}")
    H, W = img.shape[:2]

    mask = ensure_binary(cv2.imread(mask_path, 0))
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    if edge_len is None or edge_len < 0:
        perim = approx_perimeter(mask)
        edge_len, gap_factor = choose_edge_and_gap(perimeter_px=perim,
                                                   target_frag_per_100px=target_frag_per_100px,
                                                   min_edge=10, max_edge=24,
                                                   gap_factor=gap_factor)

    cnt = largest_external_contour(mask)
    if cnt is None: raise RuntimeError(f"no contour found in mask: {mask_path}")

    # resample more finely so chords track shape closely
    pts = resample_polyline(cnt, step_px=2)

    # make outline chords; disable jitter; tiny gaps
    contour_segs, occ = contour_to_segments(
        pts,
        edge_len=edge_len,
        gap_px=(0, max(1, int(0.15 * edge_len))),  # small breaks
        jitter_deg=0,
        shape=(H, W),
        thickness=thickness,
        sep_pad=max(1, sep_pad),
        stick_to_contour=bool(outline_strict),
    )

    frag = np.zeros_like(img)
    draw_segments(frag, contour_segs, color=(255, 255, 255), thickness=thickness)
    cv2.imwrite(os.path.join(out_dir, f"{name}_outline.png"), frag)

    # Random noise: fill INSIDE (many, shorter) and OUTSIDE (moderate)
    occ_inside = occ.copy()
    segs_in, occ_inside = random_noise_segments_uniform(
        H, W, count=int(inside_noise_count), length=int(inside_noise_len),
        thickness=thickness, avoid=mask, sep_pad=max(1, sep_pad),
        occ=occ_inside, tries=10, region="inside"
    )
    draw_segments(frag, segs_in, color=(220, 220, 220), thickness=thickness)

    segs_out, _ = random_noise_segments_uniform(
        H, W, count=int(outside_noise_count), length=int(outside_noise_len),
        thickness=thickness, avoid=mask, sep_pad=max(1, sep_pad),
        occ=occ_inside, tries=10, region="outside"
    )
    draw_segments(frag, segs_out, color=(220, 220, 220), thickness=thickness)

    out_frag  = os.path.join(out_dir, f"{name}_fragmented.png")
    out_panel = os.path.join(out_dir, f"{name}_panel.png")
    cv2.imwrite(out_frag, frag)
    cv2.imwrite(out_panel, panel3(img, frag, mask))
    print("saved:", out_frag, "and", out_panel)

# ----------------------- Script entry ----------------------

if __name__ == "__main__":
    IMAGES_DIR = "images"
    MASKS_DIR  = "masks"
    OUT_DIR    = "output_clean"

    for fname in os.listdir(IMAGES_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
        stem = os.path.splitext(fname)[0]
        img_p = os.path.join(IMAGES_DIR, fname)
        msk_p = os.path.join(MASKS_DIR, stem + "_mask.png")
        if not os.path.exists(msk_p):
            print(f"⚠ skip {fname} — no mask found at {msk_p}")
            continue

        fragment_one(
            img_p, msk_p, OUT_DIR,
            edge_len=-1, target_frag_per_100px=7,  # slightly denser boundary
            grid=40, gap_factor=0.25,
            jitter_deg=0, thickness=1,
            outline_strict=True,
            inside_noise_count=1100, outside_noise_count=600,
            inside_noise_len=12, outside_noise_len=16,
            sep_pad=1
        )
