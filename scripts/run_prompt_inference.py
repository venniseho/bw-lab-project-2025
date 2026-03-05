"""
run_prompt_inference.py
-----------------------
INFERENCE RUNNER: Executes SAM on pre-generated stimuli.

This script is designed to be flexible for testing different prompting strategies (box, centroid, random points) on the same set of fragmented stimuli. It relies on an existing manifest.jsonl to ensure consistency across runs.

Execution Steps:
  1. SAM Inference (run_sam_batch.py) using an existing manifest.
  2. Performance Analysis (run_analysis.py)
  3. Oracle Debugging (run_oracle_audit.py)
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

def run_cmd(cmd: list[str]):
    """Helper to execute shell commands and log them clearly."""
    print(f"\n[PIPELINE STEP] {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run SAM inference on existing stimuli.")
    
    # Path Arguments
    parser.add_argument("--out_root", required=True, help="Folder containing 'indexes/manifest.jsonl'")
    parser.add_argument("--sam_ckpt", required=True, help="Path to SAM model checkpoint")
    
    # SAM Configuration
    parser.add_argument("--prompt_mode", choices=["box", "centroid", "random_point"], required=True)
    parser.add_argument("--n_points", type=int, default=1, help="Points for random_point mode")
    parser.add_argument("--model_type", default="vit_h", help="SAM model architecture (e.g., vit_h, sam3)")
    
    # Analysis Configuration
    parser.add_argument("--suffix", help="Optional suffix for the plots folder (e.g., 'v2', 'sam3')")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    manifest_path = out_root / "indexes" / "manifest.jsonl"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Generate stimuli first.")

    # 1. Stage 2: SAM Batch Inference
    # We run this on the existing out_root; it will update metrics/sam_iou.csv
    sam_cmd = [
        "python3", "run_sam_batch.py",
        "--out_root", str(out_root),
        "--manifest", str(manifest_path),
        "--sam_ckpt", args.sam_ckpt,
        "--prompt_mode", args.prompt_mode,
        "--model_type", args.model_type
    ]
    if args.prompt_mode == "random_point":
        sam_cmd += ["--n_points", str(args.n_points)]
    
    run_cmd(sam_cmd)

    # 2. Stage 3: Analysis & Visualization
    # We save plots to a specific subfolder to avoid overwriting previous prompt results
    plot_folder_name = f"plots_{args.prompt_mode}"
    if args.suffix:
        plot_folder_name += f"_{args.suffix}"
        
    plot_dir = out_root / plot_folder_name
    csv_path = out_root / "metrics" / "sam_iou.csv"
    
    # Run standardized analysis
    run_cmd([
        "python3", "scripts/run_analysis.py",
        "--csv", str(csv_path),
        "--out_dir", str(plot_dir)
    ])
    
    # Run Oracle Audit (Hierarchy check)
    run_cmd([
        "python3", "scripts/run_oracle_audit.py",
        "--csv", str(csv_path),
        "--out_dir", str(plot_dir / "oracle_debug")
    ])

    print(f"\n[COMPLETE] Inference and Analysis finished.")
    print(f"Results available in: {plot_dir}")

if __name__ == "__main__":
    main()