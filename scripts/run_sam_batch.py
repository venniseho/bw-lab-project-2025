"""
run_sam_batch.py
-----------------
STAGE 2: Batch evaluation of SAM on fragmented stimuli.
This script orchestrates the evaluation process by:
  1. Loading the SAMInferenceEngine (supporting SAM 1 or SAM 3).
  2. Identifying image triples (Original, Fragmented, GT Mask).
  3. Running hierarchical evaluation (Model vs. Oracle).
  4. Saving metrics (including box/gt areas) to CSV for analysis.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add project root to path so 'core' and 'utils' are visible
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np

# Core logic imports
from core.sam_engine import SAMInferenceEngine
from core.metrics import get_chance_iou, get_normalized_iou
from core.vizualization_utils import apply_overlay, make_5_panel_debug, make_panel_3x2
from core.prompting import generate_sam_prompt
from utils.io_utils import read_manifest

def get_worklist(out_root: Path, manifest_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Identifies the set of images to process via manifest."""
    work = []
    if manifest_path and manifest_path.exists():
        data_rows = read_manifest(manifest_path)
        for row in data_rows:
            work.append({
                "stem": row["stem"],
                "orig": Path(row["orig_img_path"]),
                "frag": Path(row["frag_img_path"]),
                "mask": Path(row["gt_mask_path"])
            })
    # Filter to ensure files exist before starting
    return [w for w in work if w["orig"].exists() and w["frag"].exists()]

def main():
    parser = argparse.ArgumentParser(description="Run SAM Batch Evaluation")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    # NEW: Accepts a list of prompts (e.g. --prompt_modes centroid 3 box)
    parser.add_argument("--prompt_modes", nargs="+", default=["centroid"], 
                        help="List of modes: 'centroid', 'box', or an integer for point count")
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args()

    # 1. Resolve Model Type
    requested_model = args.sam_model_type.lower()
    if requested_model == "sam1":
        actual_model_type = "vit_h"
    elif requested_model == "sam3":
        actual_model_type = "sam3"
    else:
        actual_model_type = requested_model

    # 2. Initialize SAM Engine ONCE
    print(f"Initializing SAM Engine: {actual_model_type}")
    engine = SAMInferenceEngine(args.sam_ckpt, actual_model_type)

    # 3. Load Worklist ONCE
    manifest_p = Path(args.manifest) if args.manifest else Path(args.out_root) / "indexes" / "manifest.jsonl"
    worklist = get_worklist(Path(args.out_root), manifest_p)
    
    if not worklist:
        print(f"Error: No images found to process. Check manifest: {manifest_p}")
        return

    # 4. Loop through each prompt mode
    for mode_key in args.prompt_modes:
        # Parse the prompt key
        if mode_key == "centroid":
            current_mode, n_points = "centroid", 1
        elif mode_key == "box":
            current_mode, n_points = "box", 0
        elif mode_key.isdigit():
            current_mode, n_points = "random_point", int(mode_key)
        else:
            current_mode, n_points = mode_key, 1

        print(f"\n>>> TESTING: {actual_model_type} with {mode_key} prompt")
        
        # Setup specific output directories for this mode
        mode_out_root = Path(args.out_root) / requested_model / mode_key
        metrics_dir = mode_out_root / "metrics"
        debug_root = mode_out_root / "debug" / "sam_overlays"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        results_csv = []

        # 5. Inference Loop
        for i, item in enumerate(worklist):
            orig = cv2.imread(str(item["orig"]))
            frag = cv2.imread(str(item["frag"]))
            mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
            
            if orig is None or frag is None or mask is None:
                continue

            instance_debug_dir = debug_root / item["stem"]
            instance_debug_dir.mkdir(parents=True, exist_ok=True)

            # Generate Prompt
            p_coords, p_labels, p_box = generate_sam_prompt(
                mask, mode=current_mode, n_points=n_points
            )
            
            # Metrics Pre-calc
            H, W = mask.shape[:2]
            gt_area = np.count_nonzero(mask > 0)
            box_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1]) if p_box is not None else 0
            
            # Run Inference
            res_o = engine.segment_and_evaluate(orig, mask, p_coords, p_labels, p_box)
            res_f = engine.segment_and_evaluate(frag, mask, p_coords, p_labels, p_box)

            # Extra Metrics
            chance = get_chance_iou(mask)
            niou_f = get_normalized_iou(res_f["model_iou"], chance)

            # 6. Visualization (FIXED: matching updated vizualization_utils)
            prompt_vis = p_box if current_mode == "box" else p_coords
            
            # 5-Panel View
            panel_5 = make_5_panel_debug(frag, res_f, prompt_vis, current_mode)
            cv2.imwrite(str(instance_debug_dir / "panel_5.png"), panel_5)

            # 3x2 Comparison View (Now passes full result dicts and prompts)
            panel_3x2 = make_panel_3x2(
                orig, frag, mask, 
                res_o, res_f, 
                prompt_vis, current_mode
            )
            cv2.imwrite(str(instance_debug_dir / "panel_3x2.png"), panel_3x2)

            # Record Data
            results_csv.append({
                "stem": item["stem"],
                "iou_orig": round(res_o["model_iou"], 4),
                "iou_frag": round(res_f["model_iou"], 4),
                "niou_frag": round(niou_f, 4),
                "oracle_iou_frag": round(res_f["oracle_iou"], 4),
                "ari_frag": round(res_f["ari"], 4),
                "chance_iou": round(chance, 6),
                "gt_area": gt_area,
                "box_area": box_area,
                "image_area": H * W,
                "prompt_mode": current_mode,
                "point_count": n_points
            })

            if i % 10 == 0:
                print(f"  [{i}/{len(worklist)}] Processed {item['stem']}")

        # 7. Save CSV for this prompt mode
        csv_path = metrics_dir / "sam_results.csv"
        if results_csv:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results_csv[0].keys())
                writer.writeheader()
                writer.writerows(results_csv)
            print(f"Metrics saved to: {csv_path}")

    print("\nBatch suite complete.")

if __name__ == "__main__":
    main()