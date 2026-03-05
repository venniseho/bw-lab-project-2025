"""
metrics.py
----------
Evaluation metrics for image segmentation.

This module provides functions to quantify the similarity between 
predicted masks and ground truth (GT) masks. It includes standard 
overlap metrics and normalization techniques to account for 
object size relative to the image.

Main Functions:
    - compute_iou: Standard Intersection over Union.
    - compute_ari: Adjusted Rand Index for cluster-based similarity.
    - get_chance_iou: Baseline IoU if foreground filled the entire frame.
    - get_normalized_iou: Corrects IoU scores based on the chance baseline.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) of two boolean masks.
    
    Args:
        mask_a: First mask (boolean or 0/1).
        mask_b: Second mask (boolean or 0/1).
        
    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def compute_ari(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculates the Adjusted Rand Index (ARI) between two masks.
    ARI measures similarity by considering all pairs of pixels and 
    adjusting for the grouping that would happen by chance.
    
    Args:
        gt_mask: Ground truth mask (uint8 or bool).
        pred_mask: Predicted mask (bool).
        
    Returns:
        float: ARI score.
    """
    if pred_mask is None:
        return 0.0
    
    # Flatten to 1D label arrays for sklearn
    gt_flat = (gt_mask > 0).astype(np.int8).ravel()
    pred_flat = pred_mask.astype(np.int8).ravel()
    
    return float(adjusted_rand_score(gt_flat, pred_flat))

def get_chance_iou(gt_mask: np.ndarray) -> float:
    """
    Calculates the 'Chance IoU'—the score achieved if the model 
    predicted every pixel as foreground. This represents the object 
    density in the image.
    
    Args:
        gt_mask: Ground truth mask.
        
    Returns:
        float: Density of the mask relative to image size.
    """
    gt_bool = (gt_mask > 0)
    return float(gt_bool.sum()) / float(gt_bool.size) if gt_bool.size > 0 else 0.0

def get_normalized_iou(iou: float, chance: float) -> float:
    """
    Calculates the Normalized IoU (nIoU).
    nIoU = (IoU - chance) / (1 - chance).
    
    This metric penalizes large-object masks that gain high IoU 
    simply by covering more area.
    
    Args:
        iou: The actual IoU score achieved.
        chance: The chance baseline (object density).
        
    Returns:
        float: Normalized score clipped between 0.0 and 1.0.
    """
    denom = (1.0 - chance)
    if denom <= 1e-9:
        return 1.0 if iou >= 1.0 - 1e-9 else 0.0
    
    n = (iou - chance) / denom
    return float(np.clip(n, 0.0, 1.0))