"""
prompting.py
-------------
SAM Prompt Generation Utilities.

This module provides logic for generating visual prompts based on 
ground truth (GT) masks. It supports various experimental modes 
to test SAM's robustness under different levels of guidance.

Main Functions:
    - generate_sam_prompt: The primary entry point for creating 
      point-based or box-based prompts.
"""

import random
import numpy as np
from typing import Tuple, Optional, Any

def generate_sam_prompt(
    gt_mask_u8: np.ndarray, 
    mode: str = "centroid", 
    n_points: int = 1
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Creates SAM-compatible prompts from a binary mask.

    Args:
        gt_mask_u8: The ground truth mask (uint8, 0 or 255).
        mode: The strategy ('centroid', 'box', 'random_point').
        n_points: Number of points to sample if mode is 'random_point'.

    Returns:
        Tuple of (coords, labels, box):
            - coords: Nx2 array of [x, y] coordinates.
            - labels: N array of 1s (foreground).
            - box: [x1, y1, x2, y2] array or None.
    """
    ys, xs = np.where(gt_mask_u8 > 0)
    
    # Handle empty masks
    if xs.size == 0:
        return None, None, None

    if mode == "box":
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        return None, None, box

    elif mode == "random_point":
        # Sample N unique indices from the foreground pixel coordinates
        count = min(len(xs), n_points)
        indices = random.sample(range(len(xs)), count)
        coords = np.array([[xs[i], ys[i]] for i in indices], dtype=np.float32)
        labels = np.ones(count, dtype=np.int32)
        return coords, labels, None

    else:  # Default to 'centroid'
        cx, cy = float(xs.mean()), float(ys.mean())
        coords = np.array([[cx, cy]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        return coords, labels, None