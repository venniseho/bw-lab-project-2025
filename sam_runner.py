"""
sam_runner.py
--------------------------------------------------------------
Utilities for running SAM with versatile prompting and 
hierarchical evaluation.

Prompt Modes:
  - 'centroid': Single point at the mass center of GT.
  - 'box': Bounding box of the GT.
  - 'random_point': A random single point inside the GT.

Hierarchical Eval:
  - Returns "Model IoU" (mask with highest internal SAM score).
  - Returns "Oracle IoU" (mask with best IoU vs GT, regardless of score).
  - Returns Adjusted Rand Index (ARI).
  - Returns Chance IoU (baseline).

Outputs are DEBUG/ANALYSIS ONLY.
--------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from sklearn.metrics import adjusted_rand_score


# ------------------------------------------------------------
# Global SAM state
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
# Helpers
# ------------------------------------------------------------
def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def _mask_centroid(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())

def compute_ari(gt_mask_u8: np.ndarray, pred_bool: np.ndarray | None) -> float:
    """
    Compute Adjusted Rand Index (ARI) between binary masks.
    We flatten the images to 1D arrays of labels (0 or 1).
    """
    if pred_bool is None:
        return 0.0
    
    # Downsample for speed if images are huge? usually 512x512 is fast enough.
    gt_flat = (gt_mask_u8 > 0).astype(np.int8).ravel()
    pred_flat = pred_bool.astype(np.int8).ravel()
    
    return float(adjusted_rand_score(gt_flat, pred_flat))

def chance_iou_full_image(gt_mask_u8: np.ndarray) -> float:
    """
    Chance baseline if a model predicts the entire image as foreground.
    """
    gt = (gt_mask_u8 > 0)
    return float(gt.sum()) / float(gt.size) if gt.size > 0 else 0.0

def normalized_iou(iou: float, chance: float) -> float:
    """
    Normalize IoU: (IoU - chance) / (1 - chance).
    """
    denom = (1.0 - chance)
    if denom <= 1e-9:
        # GT fills whole image -> treat as perfect if IoU==1 else 0
        return 1.0 if iou >= 1.0 - 1e-9 else 0.0
    n = (iou - chance) / denom
    return float(np.clip(n, 0.0, 1.0))


# ------------------------------------------------------------
# Versatile Segmentation
# ------------------------------------------------------------

def segment_with_sam_versatile(
    image_bgr: np.ndarray,
    gt_mask_u8: np.ndarray,
    prompt_mode: str = "centroid", 
    n_points: int = 1,  
):
    """
    Runs SAM prediction with flexible prompting.
    Modes: 'centroid', 'box', 'random_point' (supports n_points)
    """
    assert _PREDICTOR is not None, "call init_sam(...) first"

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _PREDICTOR.set_image(image_rgb)

    # --- 1. Generate Prompt ---
    point_coords = None
    point_labels = None
    box = None
    prompt_vis = None

    ys, xs = np.where(gt_mask_u8 > 0)
    
    # Handle empty masks gracefully
    if xs.size == 0:
        return None, None, 0.0, {"oracle_iou": 0.0, "ari": 0.0, "model_iou": 0.0}

    if prompt_mode == "box":
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max, y_max])
        prompt_vis = box

    elif prompt_mode == "random_point":
        # Sample N unique points if possible
        count = min(len(xs), n_points)
        indices = random.sample(range(len(xs)), count)
        
        coords = []
        labels = []
        vis_points = []
        
        for idx in indices:
            cx, cy = float(xs[idx]), float(ys[idx])
            coords.append([cx, cy])
            labels.append(1) # 1 = foreground point
            vis_points.append((cx, cy))
            
        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
        prompt_vis = vis_points

    else: # Default: Centroid (Single Point)
        cx, cy = float(xs.mean()), float(ys.mean())
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        prompt_vis = [(cx, cy)]

    # --- 2. Predict ---
    masks, scores, _ = _PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True, 
    )

    if masks is None or len(masks) == 0:
         return None, None, 0.0, {"oracle_iou": 0.0, "ari": 0.0, "model_iou": 0.0}

    # --- 3. Evaluation ---
    gt_bool = gt_mask_u8 > 0
    ious = [_iou_bool(gt_bool, m) for m in masks]
    
    # Model Selection
    model_idx = int(np.argmax(scores))
    pred_mask = masks[model_idx].astype(bool)
    pred_score = float(scores[model_idx])
    model_iou = ious[model_idx]

    # Oracle Selection
    oracle_idx = int(np.argmax(ious))
    oracle_iou = float(ious[oracle_idx])
    
    # ARI
    ari = compute_ari(gt_mask_u8, pred_mask)

    extra_stats = {
        "model_iou": model_iou,
        "oracle_iou": oracle_iou,
        "ari": ari,
    }

    return pred_mask, prompt_vis, pred_score, extra_stats

# ------------------------------------------------------------
# Visualization Helper
# ------------------------------------------------------------
def overlay_vis(
    image_bgr: np.ndarray,
    pred_bool: np.ndarray | None,
    gt_mask_u8: np.ndarray | None,
    prompt_vis: Any,
    prompt_mode: str
) -> np.ndarray:
    """
    Draws GT (red), Pred (cyan), and Prompt (blue point or box).
    """
    out = image_bgr.copy()

    # GT
    if gt_mask_u8 is not None:
        gt_bool = gt_mask_u8 > 0
        if np.any(gt_bool):
            ov = out.copy()
            ov[gt_bool] = (0, 0, 255) # Red
            out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)

    # Pred
    if pred_bool is not None:
        if np.any(pred_bool):
            ov = out.copy()
            ov[pred_bool] = (255, 255, 0) # Cyan
            out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)

    # Prompt
    if prompt_vis is not None:
        if prompt_mode == "box":
            x1, y1, x2, y2 = prompt_vis.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            # Handle both single tuple (cx, cy) and list of tuples [(cx, cy), ...]
            points = prompt_vis if isinstance(prompt_vis, list) else [prompt_vis]
            for pt in points:
                cx, cy = pt
                cv2.circle(out, (int(cx), int(cy)), 6, (255, 0, 0), -1)
                cv2.circle(out, (int(cx), int(cy)), 3, (255, 255, 255), -1)

    return out

def mask_outline_image(mask_u8: np.ndarray, size_hw: Tuple[int, int], color=(255, 255, 255), thickness: int = 2) -> np.ndarray:
    H, W = size_hw
    m = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [cnt], -1, color, thickness)
    return out

def _resize_to(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    H, W = size_hw
    if img.shape[0] == H and img.shape[1] == W:
        return img
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

def _mask_to_bgr(mask_u8: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
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
    Creates the 6-grid panel:
    [ Orig    | Frag    | GT Mask ]
    [ Ov Orig | Ov Frag | Outline ]
    """
    # Standardize to Fragment size
    Hf, Wf = frag_bgr.shape[:2]
    size_hw = (Hf, Wf)

    orig_r = _resize_to(orig_bgr, size_hw)
    ov_orig_r = _resize_to(ov_orig, size_hw)
    gt_mask_bgr = _mask_to_bgr(gt_mask_full_u8, size_hw)
    gt_outline = mask_outline_image(gt_mask_full_u8, size_hw=size_hw)

    row1 = np.hstack([orig_r, frag_bgr, gt_mask_bgr])
    row2 = np.hstack([ov_orig_r, ov_frag, gt_outline])
    
    return np.vstack([row1, row2])
# ------------------------------------------------------------
# Main Wrapper
# ------------------------------------------------------------

def run_sam_on_pair(
    orig_img_path: str,
    frag_img_path: str,
    gt_mask_path: str,
    out_dir: str,
    sam_checkpoint: str,
    model_type: str = "vit_h",
    device: str | None = None,
    prompt_mode: str = "centroid",
    n_points: int = 1, 
    outline_img_path: str | None = None,
):
    init_sam(sam_checkpoint, model_type=model_type, device=device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(frag_img_path).stem.replace("_fragmented", "")

    # Load
    orig = cv2.imread(orig_img_path)
    frag = cv2.imread(frag_img_path)
    gt_mask_full = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or frag is None or gt_mask_full is None:
        print(f"⚠ SAM: missing file(s) for {stem}")
        return None

    # Resize GT to match
    H, W = orig.shape[:2]
    gt_orig = cv2.resize(gt_mask_full, (W, H), interpolation=cv2.INTER_NEAREST)
    Hf, Wf = frag.shape[:2]
    gt_frag = cv2.resize(gt_mask_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # --- Run SAM on Original ---
    seg_o, p_vis_o, score_o, stats_o = segment_with_sam_versatile(
        orig, gt_orig, prompt_mode=prompt_mode, n_points=n_points
    )
    
    # --- Run SAM on Fragmented ---
    seg_f, p_vis_f, score_f, stats_f = segment_with_sam_versatile(
        frag, gt_frag, prompt_mode=prompt_mode, n_points=n_points
    )

    # Visuals
    ov_orig = overlay_vis(orig, seg_o, gt_orig, p_vis_o, prompt_mode)
    ov_frag = overlay_vis(frag, seg_f, gt_frag, p_vis_f, prompt_mode)
    
    panel_3x2 = make_panel_3x2(orig, frag, gt_mask_full, ov_orig, ov_frag)
    cv2.imwrite(str(out_dir / f"{stem}_panel_3x2.png"), panel_3x2)
    
    cv2.imwrite(str(out_dir / f"{stem}_sam_orig.png"), ov_orig)
    cv2.imwrite(str(out_dir / f"{stem}_sam_frag.png"), ov_frag)

    # --- Metrics ---
    chance = chance_iou_full_image(gt_frag)
    niou_orig = normalized_iou(stats_o["model_iou"], chance)
    niou_frag = normalized_iou(stats_f["model_iou"], chance)

    return {
        "stem": stem,
        # Standard IoU (Model selected)
        "iou_orig": stats_o["model_iou"],
        "iou_frag": stats_f["model_iou"],
        # Baselines
        "chance_iou": chance,
        "niou_orig": niou_orig,
        "niou_frag": niou_frag,
        # Oracle IoU (Best of 3)
        "oracle_iou_orig": stats_o["oracle_iou"],
        "oracle_iou_frag": stats_f["oracle_iou"],
        # Adjusted Rand Index
        "ari_orig": stats_o["ari"],
        "ari_frag": stats_f["ari"],
        # Confidence Score
        "score_orig": score_o,
        "score_frag": score_f,
    }