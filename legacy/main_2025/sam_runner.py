"""
sam_runner.py
--------------------------------------------------------------
Utilities for running SAM on:

  • the original COCO image
  • the fragmented image

and saving nice overlays + a 3-panel bottom row:
  [COCO mask | fragmented | SAM on fragmented]

Now:
  - Uses a single point prompt at the centroid of the GT mask
--------------------------------------------------------------
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# ------------------------------------------------------------
# Global SAM state (so we only load the model once)
# ------------------------------------------------------------

_SAM = None
_PREDICTOR = None
_DEVICE = None


def init_sam(
    sam_checkpoint: str,
    model_type: str = "vit_h",
    device: str | None = None,
):
    """
    Lazy-init SAM + point-prompt predictor.
    Call this once before segmenting images.
    """
    global _SAM, _PREDICTOR, _DEVICE
    if _SAM is not None:
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _DEVICE = device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    _SAM = sam

    _PREDICTOR = SamPredictor(sam)


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------

def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _mask_centroid(mask_u8: np.ndarray):
    """
    Compute centroid (cx, cy) of a binary mask (0/255 or 0/1).
    Returns (cx, cy) as floats, or None if mask is empty.
    """
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    return cx, cy


def segment_with_sam_point(
    image_bgr: np.ndarray,
    gt_mask_u8: np.ndarray | None = None,
):
    """
    Run SAM with a single point prompt.

    If gt_mask_u8 is given:
      - Use its centroid as the prompt point.
      - Choose the SAM mask with best IoU vs GT.

    If gt_mask_u8 is None:
      - Use the image center as the prompt point.
      - Choose the mask with highest predicted score.

    Returns:
      seg_bool  (H, W) or None
      (cx, cy)  prompt coordinates (float, float) or None
    """
    assert _PREDICTOR is not None, "call init_sam(...) first"

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _PREDICTOR.set_image(image_rgb)

    # --- choose prompt point ---
    if gt_mask_u8 is not None:
        pt = _mask_centroid(gt_mask_u8)
        if pt is None:
            # fallback: center of image
            cx, cy = W / 2.0, H / 2.0
        else:
            cx, cy = pt
    else:
        cx, cy = W / 2.0, H / 2.0

    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    masks, scores, logits = _PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        return None, None

    # --- pick best mask ---
    if gt_mask_u8 is not None:
        gt_bool = gt_mask_u8 > 0
        best_idx = 0
        best_iou = -1.0
        for i, m in enumerate(masks):
            iou = _iou_bool(gt_bool, m)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        best_mask = masks[best_idx]
    else:
        # fallback: highest SAM score
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]

    return best_mask.astype(bool), (cx, cy)


def overlay_mask(
    image_bgr: np.ndarray,
    seg_bool: np.ndarray,
    color=(0, 255, 0),
    alpha: float = 0.5,
    point: tuple[float, float] | None = None,
    point_color=(0, 0, 255),
    point_radius: int = 4,
) -> np.ndarray:
    """
    Alpha-blend a single mask onto the image; optionally draw a dot
    at the given point (cx, cy) for the prompt location.
    """
    out = image_bgr.copy()
    if seg_bool is not None:
        if seg_bool.dtype != bool:
            seg_bool = seg_bool.astype(bool)
        overlay = out.copy()
        overlay[seg_bool] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    if point is not None:
        cx, cy = point
        cv2.circle(out, (int(round(cx)), int(round(cy))), point_radius, point_color, -1)

    return out


# ------------------------------------------------------------
# Public: run SAM on original + fragmented
# ------------------------------------------------------------

def run_sam_on_pair(
    orig_img_path: str,
    frag_img_path: str,
    gt_mask_path: str,
    out_dir: str,
    sam_checkpoint: str,
    model_type: str = "vit_h",
    device: str | None = None,
):
    """
    - loads SAM (if not already loaded)
    - runs SAM on the original & fragmented images, using
      a point prompt at the centroid of the GT mask
    - draws a dot at the prompt location on both overlays
    - saves:
        <stem>_sam_orig.png
        <stem>_sam_frag.png
        <stem>_sam_bottom_panel.png
      where bottom panel = [COCO mask | fragmented | SAM on fragmented]
    """
    init_sam(sam_checkpoint, model_type=model_type, device=device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(orig_img_path).stem

    orig = cv2.imread(orig_img_path)
    frag = cv2.imread(frag_img_path)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or frag is None or gt_mask is None:
        print(f"⚠ SAM: missing file(s) for {stem}, skipping")
        return

    # --- resize GT mask to each image as needed ---
    H, W = orig.shape[:2]
    gt_orig = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    Hf, Wf = frag.shape[:2]
    gt_frag = cv2.resize(gt_mask, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # --- SAM on original (with dot) ---
    seg_orig, pt_orig = segment_with_sam_point(orig, gt_orig)
    ov_orig = overlay_mask(orig, seg_orig, point=pt_orig)

    # --- SAM on fragmented (with dot) ---
    seg_frag, pt_frag = segment_with_sam_point(frag, gt_frag)
    ov_frag = overlay_mask(frag, seg_frag, point=pt_frag)

    # --- bottom 3-panel (COCO mask | fragmented | SAM on fragmented) ---
    mask_bgr = cv2.cvtColor(gt_frag, cv2.COLOR_GRAY2BGR)
    panel_bottom = np.hstack([mask_bgr, frag, ov_frag])

    # --- save ---
    cv2.imwrite(str(out_dir / f"{stem}_sam_orig.png"), ov_orig)
    cv2.imwrite(str(out_dir / f"{stem}_sam_frag.png"), ov_frag)
    cv2.imwrite(str(out_dir / f"{stem}_sam_bottom_panel.png"), panel_bottom)

    print(f"SAM overlays saved for {stem}")
