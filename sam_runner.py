"""
sam_runner.py
--------------------------------------------------------------
Utilities for running SAM on:
  • the original COCO image
  • the fragmented image

Saves overlays + a 3-panel bottom row:
  [COCO mask | fragmented | SAM on fragmented]

Now:
  - Uses a single point prompt at the centroid of the GT mask
  - Draws a dot on the prompt location
  - Returns IoU on clean vs IoU on fragmented
--------------------------------------------------------------
"""

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
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _mask_centroid(mask_u8: np.ndarray):
    """
    Centroid (cx, cy) of a binary mask.
    Returns (cx, cy) floats, or None if empty.
    """
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def segment_with_sam_point(
    image_bgr: np.ndarray,
    gt_mask_u8: np.ndarray | None = None,
):
    """
    Run SAM with a single point prompt.

    If gt_mask_u8 is given:
      - Use GT centroid as prompt point.
      - Choose the SAM mask with best IoU vs GT.

    If gt_mask_u8 is None:
      - Use image center as prompt point.
      - Choose highest predicted score.

    Returns:
      seg_bool (H,W) or None
      (cx, cy) prompt coords or None
    """
    assert _PREDICTOR is not None, "call init_sam(...) first"

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _PREDICTOR.set_image(image_rgb)

    # prompt point
    if gt_mask_u8 is not None:
        pt = _mask_centroid(gt_mask_u8)
        cx, cy = pt if pt is not None else (W / 2.0, H / 2.0)
    else:
        cx, cy = W / 2.0, H / 2.0

    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # foreground

    masks, scores, _ = _PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        return None, None

    if gt_mask_u8 is not None:
        gt_bool = gt_mask_u8 > 0
        best_idx = int(
            np.argmax([_iou_bool(gt_bool, m.astype(bool)) for m in masks])
        )
    else:
        best_idx = int(np.argmax(scores))

    return masks[best_idx].astype(bool), (cx, cy)


def overlay_mask(
    image_bgr: np.ndarray,
    seg_bool: np.ndarray | None,
    color=(0, 255, 0),
    alpha: float = 0.5,
    point: tuple[float, float] | None = None,
    point_color=(0, 0, 255),
    point_radius: int = 4,
) -> np.ndarray:
    """
    Alpha-blend a mask onto the image; optionally draw a dot at the prompt.
    """
    out = image_bgr.copy()

    if seg_bool is not None:
        seg_bool = seg_bool.astype(bool)
        overlay = out.copy()
        overlay[seg_bool] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    if point is not None:
        cx, cy = point
        cv2.circle(out, (int(round(cx)), int(round(cy))), point_radius, point_color, -1)

    return out


def iou_u8(gt_mask_u8: np.ndarray, pred_bool: np.ndarray | None) -> float:
    if pred_bool is None:
        return 0.0
    return _iou_bool(gt_mask_u8 > 0, pred_bool.astype(bool))


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
    Runs SAM on original + fragmented using a centroid point prompt.

    Saves:
      <stem>_sam_orig.png
      <stem>_sam_frag.png
      <stem>_sam_bottom_panel.png  [COCO mask | fragmented | SAM on fragmented]

    Returns dict:
      {"stem": ..., "iou_orig": ..., "iou_frag": ...}
    """
    init_sam(sam_checkpoint, model_type=model_type, device=device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(orig_img_path).stem

    orig = cv2.imread(orig_img_path)
    frag = cv2.imread(frag_img_path)
    gt_mask_full = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or frag is None or gt_mask_full is None:
        print(f"⚠ SAM: missing file(s) for {stem}, skipping")
        return {"stem": stem, "iou_orig": 0.0, "iou_frag": 0.0}

    # resize GT for each image
    H, W = orig.shape[:2]
    gt_orig = cv2.resize(gt_mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

    Hf, Wf = frag.shape[:2]
    gt_frag = cv2.resize(gt_mask_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # SAM on original
    seg_orig, pt_orig = segment_with_sam_point(orig, gt_orig)
    ov_orig = overlay_mask(orig, seg_orig, point=pt_orig)

    # SAM on fragmented
    seg_frag, pt_frag = segment_with_sam_point(frag, gt_frag)
    ov_frag = overlay_mask(frag, seg_frag, point=pt_frag)

    # IoUs (IMPORTANT: use the correctly resized GTs)
    iou_orig = iou_u8(gt_orig, seg_orig)
    iou_frag = iou_u8(gt_frag, seg_frag)

    # bottom panel: [COCO mask | fragmented | SAM on fragmented]
    mask_bgr = cv2.cvtColor(gt_frag, cv2.COLOR_GRAY2BGR)
    panel_bottom = np.hstack([mask_bgr, frag, ov_frag])

    # save
    cv2.imwrite(str(out_dir / f"{stem}_sam_orig.png"), ov_orig)
    cv2.imwrite(str(out_dir / f"{stem}_sam_frag.png"), ov_frag)
    cv2.imwrite(str(out_dir / f"{stem}_sam_bottom_panel.png"), panel_bottom)

    print(f"SAM overlays saved for {stem} | IoU(orig)={iou_orig:.3f} IoU(frag)={iou_frag:.3f}")

    return {"stem": stem, "iou_orig": iou_orig, "iou_frag": iou_frag}
