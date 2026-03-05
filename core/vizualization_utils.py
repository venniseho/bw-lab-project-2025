"""
vizualition_utils.py
---------------------
Visualization and debugging utilities for segmentation results.

This module provides functions to overlay masks onto images and 
construct multi-panel diagnostic grids. It is designed to help 
visualize the difference between SAM's chosen mask and the 
'Oracle' (best possible) mask.

Main Functions:
    - apply_overlay: Blends a boolean mask onto a BGR image.
    - make_5_panel_debug: Creates a row showing all 3 SAM masks plus 
      the Model and Oracle selections.
    - make_comparison_panel: A 3x2 grid comparing Original vs Fragmented 
      states and their respective segmentation results.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple, List, Any

def apply_overlay(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    gt_mask_u8: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (255, 255, 0),  # Default: Cyan
    alpha: float = 0.45
) -> np.ndarray:
    """
    Blends a segmentation mask and optionally a GT outline onto an image.
    
    Args:
        image_bgr: The background image.
        mask_bool: The predicted mask to overlay.
        gt_mask_u8: Optional ground truth mask for red-tinted background.
        color: BGR color for the prediction overlay.
        alpha: Transparency of the overlay (0.0 to 1.0).
        
    Returns:
        np.ndarray: The image with overlays applied.
    """
    out = image_bgr.copy()
    
    # 1. Apply Ground Truth (Subtle Red Tint)
    if gt_mask_u8 is not None:
        gt_bool = gt_mask_u8 > 0
        if np.any(gt_bool):
            ov = out.copy()
            ov[gt_bool] = (0, 0, 255) # Red
            out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)

    # 2. Apply Prediction (Custom Color)
    if mask_bool is not None and np.any(mask_bool):
        ov = out.copy()
        ov[mask_bool] = color
        out = cv2.addWeighted(ov, alpha, out, 1.0 - alpha, 0)
        
    return out

def draw_prompts(image, prompt_vis, prompt_mode):
    """
    Overlays prompts onto a visualization image.
    """
    vis = image.copy()
    if prompt_vis is None:
        return vis

    if prompt_mode == "box":
        # prompt_vis is [x1, y1, x2, y2]
        cv2.rectangle(vis, (int(prompt_vis[0]), int(prompt_vis[1])), 
                      (int(prompt_vis[2]), int(prompt_vis[3])), (0, 255, 0), 2)
    else:
        # Handle single point or array of points [[x, y], [x, y]...]
        pts = np.atleast_2d(prompt_vis) 
        for pt in pts:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1) # Red dot
            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 1) # White outline
            
    return vis

def make_5_panel_debug(
    image_bgr: np.ndarray,
    res: dict,
    prompt_vis: Any,
    prompt_mode: str
) -> np.ndarray:
    """
    Constructs a 5-column diagnostic panel:
    [Mask 0] [Mask 1] [Mask 2] [SAM Choice] [Oracle Best]

    Args:
        image_bgr: The source image (usually the fragmented version).
        res: Results dictionary from SAMInferenceEngine.segment_and_evaluate.
        prompt_vis: Coordinates for prompt visualization.
        prompt_mode: 'centroid', 'box', etc.
    """
    panels = []
    masks = res["all_masks"]
    ious = res["all_ious"]
    scores = res["all_scores"]

    # 1. The 3 raw multimask candidates
    for i in range(3):
        if i < len(masks):
            # Overlay candidate in Purple
            vis = apply_overlay(image_bgr, masks[i], color=(255, 0, 255))
            vis = draw_prompts(vis, prompt_vis, prompt_mode)
            header = f"M{i} | Conf: {scores[i]:.2f} | IoU: {ious[i]:.2f}"
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            panels.append(vis)
        else:
            panels.append(np.zeros_like(image_bgr))

    # 2. SAM's official choice (Cyan)
    vis_sam = apply_overlay(image_bgr, res["model_mask"], color=(255, 255, 0))
    cv2.putText(vis_sam, f"SAM Choice (M{res['model_idx']})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    panels.append(vis_sam)

    # 3. The Oracle Best (Green)
    vis_oracle = apply_overlay(image_bgr, res["oracle_mask"], color=(0, 255, 0))
    cv2.putText(vis_oracle, f"Oracle Best (M{res['oracle_idx']})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    panels.append(vis_oracle)

    return np.hstack(panels)

def make_panel_3x2(
    orig_bgr: np.ndarray,
    frag_bgr: np.ndarray,
    gt_mask: np.ndarray,
    res_orig: dict,
    res_frag: dict,
    prompt_vis: Any,
    prompt_mode: str
) -> np.ndarray:
    """
    Creates a 3x2 grid summary panel as requested:
    [ OG Image          | Frag Image        | Mask (Binary) ]
    [ OG + GT(R)+SAM(B) | Frag + SAM(Blue)  | Outline       ]
    """
    h, w = frag_bgr.shape[:2]
    orig_r = cv2.resize(orig_bgr, (w, h))
    
    # --- ROW 1 ---
    # Col 1: Original Image
    r1c1 = orig_r.copy()
    # Col 2: Fragmented Image
    r1c2 = frag_bgr.copy()
    # Col 3: Binary Mask (Filled)
    r1c3 = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    r1c3[gt_mask > 0] = (255, 255, 255)

    # --- ROW 2 ---
    # Col 1: OG Image + GT (Red) + SAM (Blue)
    # BGR for Blue is (255, 0, 0)
    r2c1 = apply_overlay(orig_r, res_orig["model_mask"], gt_mask_u8=gt_mask, color=(255, 0, 0))
    r2c1 = draw_prompts(r2c1, prompt_vis, prompt_mode)

    # Col 2: Frag Image + SAM (Blue)
    r2c2 = apply_overlay(frag_bgr, res_frag["model_mask"], gt_mask_u8=gt_mask, color=(255, 0, 0))
    r2c2 = draw_prompts(r2c2, prompt_vis, prompt_mode)

    # Col 3: Outline
    r2c3 = np.zeros((h, w, 3), dtype=np.uint8)
    cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(r2c3, cnts, -1, (255, 255, 255), 2)

    # Assemble
    row1 = np.hstack([r1c1, r1c2, r1c3])
    row2 = np.hstack([r2c1, r2c2, r2c3])
    
    return np.vstack([row1, row2])