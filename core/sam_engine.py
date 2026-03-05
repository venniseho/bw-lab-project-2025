"""
sam_engine.py
-------------
Provides a unified interface for Segment Anything Models (SAM) and 
hierarchical evaluation (Model vs. Oracle).

This module manages model inference and the selection logic for 
identifying the "Model Best" (highest confidence) and "Oracle Best" 
(highest overlap with GT) masks.

Main Classes:
    - SAMInferenceEngine: Manages model lifecycle and prediction.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from .metrics import compute_iou, compute_ari

from segment_anything import sam_model_registry, SamPredictor
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

class SAMInferenceEngine:
    """
    An orchestrator for SAM inference that supports multiple model versions.
    
    Attributes:
        checkpoint (str): Path to the model weights (.pth).
        model_type (str): Type of architecture (e.g., 'vit_h', 'sam3').
        device (str): The device ('cuda' or 'cpu') the model is loaded onto.
    """

    def __init__(self, 
                 checkpoint: str, 
                 model_type: str = "vit_h", 
                 device: Optional[str] = None
                 ):
        """
        Initializes the SAM engine and loads weights into memory.
        
        Args:
            checkpoint: Path to the .pth or HuggingFace model string.
            model_type: Architecture identifier ('vit_h', 'vit_l', 'vit_b', 'sam3').
            device: Computing device. Defaults to 'cuda' if available.
        """
        self.checkpoint = checkpoint
        self.model_label = model_type
        self.model_type_map = {
            "sam1": "vit_h",
            "sam3": "sam3",
            "vit_h": "vit_h",
            "vit_l": "vit_l",
            "vit_b": "vit_b"
        }
        
        self.model_type = self.model_type_map.get(model_type, model_type)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None
        self._setup_model()

    def _setup_model(self):
        """
        Internal method to route model initialization based on self.model_type.
        Supports standard SAM-1 (Meta's implementation) and SAM-3 (HF implementation).
        """
        if self.model_type == "sam3":
            # Load SAM 3 from HuggingFace
            self.model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self.device)
            self.predictor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        else:
            # Load SAM 1 using the official segment_anything registry
            self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)

    def predict(self, 
                image_rgb: np.ndarray, 
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes a prompt-based segmentation prediction.

        Args:
            image_rgb (np.ndarray): The input image in RGB format.
            point_coords (np.ndarray): Nx2 array of [x, y] coordinates.
            point_labels (np.ndarray): N array of labels (1=fg, 0=bg).
            box (np.ndarray): [x1, y1, x2, y2] bounding box.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - masks: Boolean array of shape [M, H, W] for M generated masks.
                - scores: Array of shape [M] containing SAM's internal confidence scores.
        """
        if self.model_type == "sam3":
            # SAM 3 requires specific nesting: [batch, objects, points, xy]
            # We use [1, 1, N, 2] for single image inference
            in_pts = [[point_coords.tolist()]] if point_coords is not None else None
            in_lbs = [[point_labels.tolist()]] if point_labels is not None else None
            in_boxes = [[box.tolist()]] if box is not None else None

            inputs = self.predictor(
                images=image_rgb, 
                input_points=in_pts, 
                input_labels=in_lbs, 
                input_boxes=in_boxes, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=True)

            masks_post = self.predictor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"]
            )
            
            # Ensure output is a boolean numpy array
            masks = masks_post[0][0].numpy().astype(bool) 
            scores = outputs.iou_scores[0][0].cpu().numpy()
            return masks, scores

        else:
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
            # Ensure masks are boolean for IoU calculations
            return masks.astype(bool), scores
        
    def segment_and_evaluate(self, 
                            image_rgb: np.ndarray, 
                            gt_mask: np.ndarray,
                            point_coords: Optional[np.ndarray] = None,
                            point_labels: Optional[np.ndarray] = None,
                            box: Optional[np.ndarray] = None
                            ) -> Dict[str, Any]:
        """
        Runs inference and computes both Model and Oracle metrics.
        
        The 'Oracle' is the mask among SAM's multimask outputs that 
        yields the highest IoU with the Ground Truth.

        Returns:
            Dict containing:
                - model_mask/iou/score: The mask SAM officially chose.
                - oracle_mask/iou: The best possible mask SAM generated.
                - ari: Adjusted Rand Index for the model's choice.
                - all_masks/all_scores/all_ious: Full raw data for debugging.
        """
        masks, scores = self.predict(image_rgb, point_coords, point_labels, box)
        
        gt_bool = gt_mask > 0
        ious = [compute_iou(gt_bool, m) for m in masks]
        
        # 1. Model Selection: What SAM thinks is best (highest score)
        model_idx = int(np.argmax(scores))
        model_mask = masks[model_idx]
        
        # 2. Oracle Selection: What is actually best (highest IoU)
        oracle_idx = int(np.argmax(ious))
        oracle_mask = masks[oracle_idx]

        return {
            "model_mask": model_mask,
            "model_iou": ious[model_idx],
            "model_score": float(scores[model_idx]),
            "model_idx": model_idx,
            
            "oracle_mask": oracle_mask,
            "oracle_iou": ious[oracle_idx],
            "oracle_idx": oracle_idx,
            
            "ari": compute_ari(gt_mask, model_mask),
            "all_masks": masks,
            "all_scores": scores,
            "all_ious": ious
        }