"""
sam_runner.py
--------------------------------------------------------------
Utilities for running SAM on:
  - the original COCO image
  - the fragmented stimulus image

Prompting (for now):
  - Single positive point at the centroid of the GT mask.

IMPORTANT (evaluation validity):
  - We use GT ONLY to place the point.
  - We do NOT use GT to choose among SAM's multimask outputs.
    (Choosing the best IoU mask would be a form of leakage/cheating.)

Saves DEBUG outputs:
  - overlay on original (GT + pred + point)
  - overlay on fragmented (GT + pred + point)
  - 3×2 panel:
        row1: original | fragmented | GT mask
        row2: overlay(orig) | overlay(frag) | GT mask outline (frag-sized)

Outputs are DEBUG/ANALYSIS ONLY (do not feed overlays or panels to models/humans).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


# ------------------------------------------------------------
# Global SAM state (load once)
# ------------------------------------------------------------
_SAM = None
_PREDICTOR = None
_DEVICE = None
_SAM_CKPT = None
_SAM_TYPE = None


def init_sam(
    sam_checkpoint: str,
    model_type: str = "vit_h",
    device: str | None = None,
):
    """
    Lazy-init SAM + predictor once.

    If called with a different checkpoint/model_type later,
    we re-load to avoid mismatches.
    """
    global _SAM, _PREDICTOR, _DEVICE, _SAM_CKPT, _SAM_TYPE

    if _SAM is not None and _SAM_CKPT == sam_checkpoint and _SAM_TYPE == model_type:
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _DEVICE = device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    _SAM = sam
    _PREDICTOR = SamPredictor(sam)
    _SAM_CKPT = sam_checkpoint
    _SAM_TYPE = model_type


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------
def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _mask_centroid(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Centroid (cx, cy) of a binary-ish mask.
    Returns None if empty.
    """
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def iou_u8(gt_mask_u8: np.ndarray, pred_bool: np.ndarray | None) -> float:
    """IoU where GT is uint8 and pred is bool."""
    if pred_bool is None:
        return 0.0
    return _iou_bool(gt_mask_u8 > 0, pred_bool.astype(bool))


def chance_iou_full_image(gt_mask_u8: np.ndarray) -> float:
    """
    Chance baseline if a model predicts the entire image as foreground.
    IoU(GT, AllOnes) = area(GT) / area(image)
    """
    gt = (gt_mask_u8 > 0)
    return float(gt.sum()) / float(gt.size) if gt.size > 0 else 0.0


def segment_with_sam_centroid_point(
    image_bgr: np.ndarray,
    gt_mask_u8: np.ndarray,
):
    """
    Run SAM with a single positive point prompt at the GT centroid.

    VALIDITY NOTE:
      - GT is used ONLY to compute the point location.
      - We select SAM's output using SAM's own confidence score
        (highest score), not IoU-vs-GT.
    """
    assert _PREDICTOR is not None, "call init_sam(...) first"

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _PREDICTOR.set_image(image_rgb)

    pt = _mask_centroid(gt_mask_u8)
    cx, cy = pt if pt is not None else (W / 2.0, H / 2.0)

    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    masks, scores, _ = _PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        return None, None, None

    best_idx = int(np.argmax(scores))  # IMPORTANT: no GT-based selection
    return masks[best_idx].astype(bool), (cx, cy), float(scores[best_idx])


def overlay_pred_and_gt(
    image_bgr: np.ndarray,
    pred_bool: np.ndarray | None,
    gt_mask_u8: np.ndarray | None,
    *,
    pred_color=(255, 255, 0),   # cyan-ish (BGR)
    pred_alpha: float = 0.35,
    gt_color=(0, 0, 255),       # red (BGR)
    gt_alpha: float = 0.25,
    point: tuple[float, float] | None = None,
    point_color=(255, 0, 0),    # bright blue dot (BGR)
    point_radius: int = 5,
) -> np.ndarray:
    """
    Overlay GT + prediction on top of an image.

    Draw order:
      1) GT overlay (reddish)
      2) Pred overlay (cyan-ish)
      3) Prompt point dot (bright blue)
    """
    out = image_bgr.copy()

    # GT overlay first
    if gt_mask_u8 is not None:
        gt_bool = (gt_mask_u8 > 0)
        if np.any(gt_bool):
            ov = out.copy()
            ov[gt_bool] = gt_color
            out = cv2.addWeighted(ov, gt_alpha, out, 1.0 - gt_alpha, 0)

    # Pred overlay second
    if pred_bool is not None:
        pred_bool = pred_bool.astype(bool)
        if np.any(pred_bool):
            ov = out.copy()
            ov[pred_bool] = pred_color
            out = cv2.addWeighted(ov, pred_alpha, out, 1.0 - pred_alpha, 0)

    # Prompt point last
    if point is not None:
        cx, cy = point
        cv2.circle(out, (int(round(cx)), int(round(cy))), point_radius, point_color, -1)

    return out


def mask_outline_image(mask_u8: np.ndarray, *, size_hw: Tuple[int, int], color=(255, 255, 255), thickness: int = 2) -> np.ndarray:
    """
    Make a black image with the GT mask *outline* drawn on it.
    This is useful as the "fragmented mask outline" panel cell.

    size_hw: (H, W) for output resolution (match frag)
    """
    H, W = size_hw
    m = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W, 3), dtype=np.uint8)

    cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        # draw the largest external contour
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [cnt], -1, color, thickness)
    return out


def _resize_to(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize to (H,W) using area for images, nearest for masks-like."""
    H, W = size_hw
    if img.shape[0] == H and img.shape[1] == W:
        return img
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def _mask_to_bgr(mask_u8: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize GT mask to (H,W) and convert to BGR for panels."""
    H, W = size_hw
    m = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def make_panel_3x2(
    orig_bgr: np.ndarray,
    frag_bgr: np.ndarray,
    gt_mask_full_u8: np.ndarray,
    ov_orig: np.ndarray,
    ov_frag: np.ndarray,
) -> np.ndarray:
    """
    3 columns × 2 rows:

      row1: original | fragmented | GT mask
      row2: overlay(orig) | overlay(frag) | GT mask outline (frag-sized)

    We standardize all panel cells to the FRAG resolution.
    """
    Hf, Wf = frag_bgr.shape[:2]
    size_hw = (Hf, Wf)

    orig_r = _resize_to(orig_bgr, size_hw)
    ov_orig_r = _resize_to(ov_orig, size_hw)

    frag_r = frag_bgr
    ov_frag_r = ov_frag

    gt_mask_bgr = _mask_to_bgr(gt_mask_full_u8, size_hw)
    gt_outline = mask_outline_image(gt_mask_full_u8, size_hw=size_hw, color=(255, 255, 255), thickness=2)

    row1 = np.hstack([orig_r, frag_r, gt_mask_bgr])
    row2 = np.hstack([ov_orig_r, ov_frag_r, gt_outline])
    panel = np.vstack([row1, row2])
    return panel


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
    Runs SAM on original + fragmented using centroid point prompt.

    Writes (DEBUG):
      <stem>_overlay_orig.png
      <stem>_overlay_frag.png
      <stem>_panel_3x2.png

    Returns dict with IoU + scores.
    """
    init_sam(sam_checkpoint, model_type=model_type, device=device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use fragment filename to avoid overwriting per-instance outputs
    stem = Path(frag_img_path).stem.replace("_fragmented", "")

    orig = cv2.imread(orig_img_path)
    frag = cv2.imread(frag_img_path)
    gt_mask_full = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or frag is None or gt_mask_full is None:
        print(f"⚠ SAM: missing file(s) for {stem}, skipping")
        return {
            "stem": stem,
            "iou_orig": 0.0,
            "iou_frag": 0.0,
            "chance_iou": 0.0,
            "score_orig": None,
            "score_frag": None,
        }

    # Resize GT to match each image (for prompting + IoU)
    H, W = orig.shape[:2]
    gt_orig = cv2.resize(gt_mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

    Hf, Wf = frag.shape[:2]
    gt_frag = cv2.resize(gt_mask_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # SAM on original
    seg_orig, pt_orig, score_orig = segment_with_sam_centroid_point(orig, gt_orig)
    ov_orig = overlay_pred_and_gt(orig, pred_bool=seg_orig, gt_mask_u8=gt_orig, point=pt_orig)

    # SAM on fragmented
    seg_frag, pt_frag, score_frag = segment_with_sam_centroid_point(frag, gt_frag)
    ov_frag = overlay_pred_and_gt(frag, pred_bool=seg_frag, gt_mask_u8=gt_frag, point=pt_frag)

    # IoUs
    iou_orig = iou_u8(gt_orig, seg_orig)
    iou_frag = iou_u8(gt_frag, seg_frag)

    # Chance baseline (predict full image)
    chance = chance_iou_full_image(gt_frag)

    # 3×2 panel (standardized to frag resolution)
    panel = make_panel_3x2(
        orig_bgr=orig,
        frag_bgr=frag,
        gt_mask_full_u8=gt_mask_full,
        ov_orig=ov_orig,
        ov_frag=ov_frag,
    )

    # Save
    cv2.imwrite(str(out_dir / f"{stem}_overlay_orig.png"), ov_orig)
    cv2.imwrite(str(out_dir / f"{stem}_overlay_frag.png"), ov_frag)
    cv2.imwrite(str(out_dir / f"{stem}_panel_3x2.png"), panel)

    print(
        f"SAM overlays saved for {stem} | "
        f"IoU(orig)={iou_orig:.3f} IoU(frag)={iou_frag:.3f} chance={chance:.3f}"
    )

    return {
        "stem": stem,
        "iou_orig": float(iou_orig),
        "iou_frag": float(iou_frag),
        "chance_iou": float(chance),
        "score_orig": score_orig,
        "score_frag": score_frag,
    }
