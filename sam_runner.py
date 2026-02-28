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

"""
Sam3Tracker performs Promptable Visual Segmentation (PVS) on images, taking interactive visual prompts (points, boxes, masks) to segment a specific object instance per prompt. It is an updated version of SAM2 that maintains the same API while providing improved performance, making it a drop-in replacement for SAM2 workflows.
"""
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

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
    
    if model_type == "sam3":
        # Load stable HF version
        _SAM = Sam3TrackerModel.from_pretrained("facebook/sam3").to(_DEVICE)
        _PREDICTOR = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    else:
        # Standard SAM 1 fallback
        _SAM = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        _SAM.to(device=device)
        _PREDICTOR = SamPredictor(_SAM)

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
    assert _PREDICTOR is not None, "call init_sam(...) first"

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # _PREDICTOR.set_image(image_rgb)

    # --- 1. Generate Prompt ---
    point_coords = None
    point_labels = None
    box = None
    prompt_vis = None

    ys, xs = np.where(gt_mask_u8 > 0)
    
    if xs.size == 0:
        # Return empty structure compatible with new signature
        empty_stats = {"oracle_iou": 0.0, "ari": 0.0, "model_iou": 0.0, "model_idx": 0, "oracle_idx": 0}
        return None, None, None, 0.0, empty_stats, [], [], []

    if prompt_mode == "box":
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max, y_max])
        prompt_vis = box

    elif prompt_mode == "random_point":
        count = min(len(xs), n_points)
        indices = random.sample(range(len(xs)), count)
        coords = []
        labels = []
        vis_points = []
        for idx in indices:
            cx, cy = float(xs[idx]), float(ys[idx])
            coords.append([cx, cy])
            labels.append(1) 
            vis_points.append((cx, cy))
        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
        prompt_vis = vis_points

    else: # Centroid
        cx, cy = float(xs.mean()), float(ys.mean())
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        prompt_vis = [(cx, cy)]

    # --- 2. Predict ---
    if _SAM_TYPE == "sam3":
        # Prepare inputs for Transformers API (expects 4D inputs: [batch, objects, points, xy])
        # We use [1, 1, N, 2] for a single image/single object inference
        in_pts = [[point_coords.tolist()]] if point_coords is not None else None
        in_lbs = [[point_labels.tolist()]] if point_labels is not None else None
        in_boxes = [[box.tolist()]] if box is not None else None

        inputs = _PREDICTOR(
            images=image_rgb, 
            input_points=in_pts, 
            input_labels=in_lbs, 
            input_boxes=in_boxes, 
            return_tensors="pt"
        ).to(_DEVICE)

        with torch.no_grad():
            outputs = _SAM(**inputs, multimask_output=True)

        # Post-process to original resolution
        masks_post = _PREDICTOR.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
        
        # Pull out the first batch, first object results
        masks = masks_post[0][0].numpy()  # Results in [3, H, W]
        scores = outputs.iou_scores[0][0].cpu().numpy() # Results in [3]
    else:
        # Standard SAM 1 logic
        _PREDICTOR.set_image(image_rgb)
        masks, scores, _ = _PREDICTOR.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True, 
        )
    # masks, scores, _ = _PREDICTOR.predict(
    #     point_coords=point_coords,
    #     point_labels=point_labels,
    #     box=box,
    #     multimask_output=True, 
    # )

    # if masks is None or len(masks) == 0:
    #      empty_stats = {"oracle_iou": 0.0, "ari": 0.0, "model_iou": 0.0, "model_idx": 0, "oracle_idx": 0}
    #      return None, None, None, 0.0, empty_stats, [], [], []

    # --- 3. Evaluation ---
    gt_bool = gt_mask_u8 > 0
    ious = [_iou_bool(gt_bool, m) for m in masks]
    
    # Model Selection (Highest Score)
    model_idx = int(np.argmax(scores))
    pred_mask = masks[model_idx].astype(bool)
    pred_score = float(scores[model_idx])
    model_iou = ious[model_idx]

    # Oracle Selection (Highest IoU)
    oracle_idx = int(np.argmax(ious))
    oracle_mask = masks[oracle_idx].astype(bool)
    oracle_iou = float(ious[oracle_idx])
    
    ari = compute_ari(gt_mask_u8, pred_mask)

    extra_stats = {
        "model_iou": model_iou,
        "oracle_iou": oracle_iou,
        "ari": ari,
        "model_idx": model_idx,
        "oracle_idx": oracle_idx
    }

    # RETURN EVERYTHING: Best Mask, Oracle Mask, Visuals, Score, Stats, ALL Masks, ALL Scores, ALL IoUs
    return pred_mask, oracle_mask, prompt_vis, pred_score, extra_stats, masks, scores, ious

# ------------------------------------------------------------
# Visualization Helper
# ------------------------------------------------------------
def overlay_vis(
    image_bgr: np.ndarray,
    pred_bool: np.ndarray | None,
    gt_mask_u8: np.ndarray | None,
    prompt_vis: Any,
    prompt_mode: str,
    color=(255, 255, 0) # Default Cyan
) -> np.ndarray:
    out = image_bgr.copy()
    
    # GT (Red)
    if gt_mask_u8 is not None:
        gt_bool = gt_mask_u8 > 0
        if np.any(gt_bool):
            ov = out.copy()
            ov[gt_bool] = (0, 0, 255)
            out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)

    # Pred (Custom Color)
    if pred_bool is not None and np.any(pred_bool):
        ov = out.copy()
        ov[pred_bool] = color
        out = cv2.addWeighted(ov, 0.45, out, 0.55, 0)

    # Prompt
    if prompt_vis is not None:
        if prompt_mode == "box":
            x1, y1, x2, y2 = prompt_vis.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            points = prompt_vis if isinstance(prompt_vis, list) else [prompt_vis]
            for pt in points:
                cx, cy = pt
                cv2.circle(out, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                cv2.circle(out, (int(cx), int(cy)), 2, (255, 255, 255), -1)
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

def make_oracle_panel(
    frag_bgr: np.ndarray,
    model_mask: np.ndarray,
    oracle_mask: np.ndarray,
    gt_mask: np.ndarray,
    prompt_vis: Any,
    prompt_mode: str,
    stats: Dict[str, float],
    score: float
) -> np.ndarray:
    """
    Creates a comparison panel:
    [ SAM Choice (Cyan) ]  |  [ Oracle Choice (Green) ]
    Includes text stats.
    """
    H, W = frag_bgr.shape[:2]
    
    # Left: Model Choice (Cyan)
    vis_model = overlay_vis(frag_bgr, model_mask, gt_mask, prompt_vis, prompt_mode, color=(255, 255, 0))
    
    # Right: Oracle Choice (Green)
    vis_oracle = overlay_vis(frag_bgr, oracle_mask, gt_mask, prompt_vis, prompt_mode, color=(0, 255, 0))
    
    # Stack Side by Side
    panel = np.hstack([vis_model, vis_oracle])
    
    # Add Text Overlay
    text_lines = [
        f"SAM Conf: {score:.3f} | Model IoU: {stats['model_iou']:.3f}",
        f"Oracle IoU: {stats['oracle_iou']:.3f} | ARI: {stats['ari']:.3f}"
    ]
    
    # Draw text background
    ph = 60
    header = np.zeros((ph, panel.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(header, "LEFT: SAM Choice (Cyan)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(header, "RIGHT: Oracle Best (Green)", (W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(header, text_lines[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(header, text_lines[1], (W + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return np.vstack([header, panel])

def make_5_panel_debug(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    ious: List[float],
    gt_mask: np.ndarray,
    prompt_vis: Any,
    prompt_mode: str,
    model_idx: int,
    oracle_idx: int
) -> np.ndarray:
    """
    Creates a 5-column panel:
    [Mask 0] [Mask 1] [Mask 2] [SAM Choice] [Oracle Choice]
    """
    panels = []

    # 1. Generate panels for the 3 raw masks
    for i in range(3):
        # Handle case if SAM returns fewer than 3 masks
        if i < len(masks):
            m = masks[i]
            s = scores[i]
            iou = ious[i]
            
            # Color logic: Purple for raw candidates
            vis = overlay_vis(image_bgr, m, gt_mask, prompt_vis, prompt_mode, color=(255, 0, 255))
            
            # Add text
            header = f"M{i} | Conf: {s:.2f} | IoU: {iou:.2f}"
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panels.append(vis)
        else:
            panels.append(np.zeros_like(image_bgr))

    # 2. SAM Choice Panel (Cyan)
    sam_mask = masks[model_idx]
    vis_sam = overlay_vis(image_bgr, sam_mask, gt_mask, prompt_vis, prompt_mode, color=(255, 255, 0))
    cv2.putText(vis_sam, f"SAM Choice (M{model_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    panels.append(vis_sam)

    # 3. Oracle Choice Panel (Green)
    oracle_mask = masks[oracle_idx]
    vis_oracle = overlay_vis(image_bgr, oracle_mask, gt_mask, prompt_vis, prompt_mode, color=(0, 255, 0))
    cv2.putText(vis_oracle, f"Oracle Best (M{oracle_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    panels.append(vis_oracle)

    return np.hstack(panels)
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

    # Load images
    orig = cv2.imread(orig_img_path)
    frag = cv2.imread(frag_img_path)
    gt_mask_full = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or frag is None or gt_mask_full is None:
        return None

    # Resize GT
    H, W = orig.shape[:2]
    gt_orig = cv2.resize(gt_mask_full, (W, H), interpolation=cv2.INTER_NEAREST)
    Hf, Wf = frag.shape[:2]
    gt_frag = cv2.resize(gt_mask_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # --- Run SAM on Original ---
    seg_o, _, p_vis_o, score_o, stats_o, _, _, _ = segment_with_sam_versatile(
        orig, gt_orig, prompt_mode=prompt_mode, n_points=n_points
    )
    
    # --- Run SAM on Fragmented ---
    seg_f, oracle_f, p_vis_f, score_f, stats_f, all_masks, all_scores, all_ious = segment_with_sam_versatile(
        frag, gt_frag, prompt_mode=prompt_mode, n_points=n_points
    )

    # --- Generate 5-GRID Debug Panel ---
    if seg_f is not None:
        panel_5 = make_5_panel_debug(
            image_bgr=frag,
            masks=all_masks,
            scores=all_scores,
            ious=all_ious,
            gt_mask=gt_frag,
            prompt_vis=p_vis_f,
            prompt_mode=prompt_mode,
            model_idx=stats_f["model_idx"],
            oracle_idx=stats_f["oracle_idx"]
        )
        # Save as a wide image
        cv2.imwrite(str(out_dir / f"{stem}_debug_5grid.png"), panel_5)

    # Standard Overlays
    ov_orig = overlay_vis(orig, seg_o, gt_orig, p_vis_o, prompt_mode)
    ov_frag = overlay_vis(frag, seg_f, gt_frag, p_vis_f, prompt_mode)
    cv2.imwrite(str(out_dir / f"{stem}_sam_orig.png"), ov_orig)
    cv2.imwrite(str(out_dir / f"{stem}_sam_frag.png"), ov_frag)

    # Metrics
    chance = chance_iou_full_image(gt_frag)
    niou_orig = normalized_iou(stats_o["model_iou"], chance)
    niou_frag = normalized_iou(stats_f["model_iou"], chance)

    return {
        "stem": stem,
        "iou_orig": stats_o["model_iou"],
        "iou_frag": stats_f["model_iou"],
        "chance_iou": chance,
        "niou_orig": niou_orig,
        "niou_frag": niou_frag,
        "oracle_iou_orig": stats_o["oracle_iou"],
        "oracle_iou_frag": stats_f["oracle_iou"],
        "ari_orig": stats_o["ari"],
        "ari_frag": stats_f["ari"],
        "score_orig": score_o,
        "score_frag": score_f,
        
        # Save masks and indices for further analysis
        "model_idx_frag": stats_f["model_idx"],
        "oracle_idx_frag": stats_f["oracle_idx"],
    }