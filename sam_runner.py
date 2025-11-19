"""
sam_runner.py
--------------------------------------------------------------
Utilities for running SAM on:

  • the original COCO image
  • the fragmented image

and saving nice overlays + a 3-panel bottom row:
  [COCO mask | fragmented | SAM on fragmented]

Now uses SamPredictor with a point prompt at the
centre of the GT mask (union or instance).
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
_PREDICTOR: SamPredictor | None = None
_DEVICE = None


def init_sam(
    sam_checkpoint: str,
    model_type: str = "vit_h",
    device: str | None = None,
):
    """
    Lazy-init SAM + predictor.
    Call this once before segmenting images.
    """
    global _SAM, _PREDICTOR, _DEVICE
    if _SAM is not None and _PREDICTOR is not None:
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _DEVICE = device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    _SAM = sam
    _PREDICTOR = predictor


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------

def mask_centroid(mask_u8: np.ndarray) -> np.ndarray | None:
    """
    mask_u8: HxW uint8, object > 0.
    Returns coords as [[cx, cy]] in (x, y) order or None if empty.
    """
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) == 0:
        return None
    cx = xs.mean()
    cy = ys.mean()
    return np.array([[cx, cy]], dtype=np.float32)


def segment_with_sam_point(
    image_bgr: np.ndarray,
    gt_mask_u8: np.ndarray,
    multimask_output: bool = True,
):
    """
    Run SAM with a *point prompt at the centre of gt_mask_u8*.

    Returns:
      seg_bool  (H, W) or None if SAM failed
    """
    assert _PREDICTOR is not None, "call init_sam(...) first"

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # compute centroid of mask
    pt = mask_centroid(gt_mask_u8)
    if pt is None:
        return None

    _PREDICTOR.set_image(rgb)

    point_coords = pt  # shape (1, 2)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    masks, scores, logits = _PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
    )

    if masks is None or len(masks) == 0:
        return None

    # take best scoring mask
    best_idx = int(np.argmax(scores))
    seg = masks[best_idx].astype(bool)  # HxW bool
    return seg


def overlay_mask(
    image_bgr: np.ndarray,
    seg_bool: np.ndarray | None,
    color=(0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Alpha-blend a single mask onto the image.
    """
    out = image_bgr.copy()
    if seg_bool is None:
        return out

    if seg_bool.dtype != bool:
        seg_bool = seg_bool.astype(bool)

    overlay = out.copy()
    overlay[seg_bool] = color
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
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
    multimask_output: bool = True,
):
    """
    - loads SAM (if not already loaded)
    - runs SAM on the original & fragmented images using a point
      prompt at the centre of the GT mask
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

    # --- resize GT mask to original image size & run SAM ---
    H, W = orig.shape[:2]
    gt_mask_orig = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    seg_orig = segment_with_sam_point(orig, gt_mask_orig, multimask_output=multimask_output)
    ov_orig = overlay_mask(orig, seg_orig)

    # --- resize GT mask to fragmented size & run SAM ---
    Hf, Wf = frag.shape[:2]
    gt_for_frag = cv2.resize(gt_mask, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
    seg_frag = segment_with_sam_point(frag, gt_for_frag, multimask_output=multimask_output)
    ov_frag = overlay_mask(frag, seg_frag)

    # --- bottom 3-panel (COCO mask | fragmented | SAM on fragmented) ---
    mask_bgr = cv2.cvtColor(gt_for_frag, cv2.COLOR_GRAY2BGR)
    panel_bottom = np.hstack([mask_bgr, frag, ov_frag])

    # --- save ---
    cv2.imwrite(str(out_dir / f"{stem}_sam_orig.png"), ov_orig)
    cv2.imwrite(str(out_dir / f"{stem}_sam_frag.png"), ov_frag)
    cv2.imwrite(str(out_dir / f"{stem}_sam_bottom_panel.png"), panel_bottom)

    print(f"SAM overlays saved for {stem}")
