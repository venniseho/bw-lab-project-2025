"""
run_noise_experiment.py
-----------------------
Executes a sweep across noise densities.

This script ensures stimuli exist for each noise level and then triggers
standardized batch inference and analysis for each density.
"""

from __future__ import annotations
import argparse
import subprocess
import os
from pathlib import Path

import os

def run_cmd(cmd: list[str]):
    """Helper to execute shell commands and log them clearly."""
    print(f"\n[PIPELINE STEP] {' '.join(cmd)}")
    
    # Inject the project root into PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + ":" + env.get("PYTHONPATH", "")
    
    # Use the new env with the subprocess call
    subprocess.check_call(cmd, env=env)

def main():
    parser = argparse.ArgumentParser(description="Run a sweep across noise densities.")
    
    # Path Arguments
    parser.add_argument("--stimuli_root", required=True, help="Folder to store/read stimuli")
    parser.add_argument("--sam_ckpt", required=True, help="Path to SAM checkpoint")
    
    # Data Generation Configuration
    parser.add_argument("--coco_ann", required=True)
    parser.add_argument("--coco_imgdir", required=True)
    parser.add_argument("--noise_levels", type=int, nargs="+", default=[0, 25, 50, 75, 100, 200, 400, 800])
    parser.add_argument("--limit", type=int, default=10)

    # SAM Configuration
    parser.add_argument("--prompt_mode", default="centroid")
    parser.add_argument("--model_type", default="sam1")
    
    args = parser.parse_args()
    stim_root = Path(args.stimuli_root)

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    for n in args.noise_levels:
        print(f"\n{'='*60}\n PROCESSING NOISE LEVEL: n{n}\n{'='*60}")

        n_dir = project_root / "outputs" / "noise_sweep_experiment" / f"n{n}"
        manifest_path = stim_root / f"n{n}" / "indexes" / "manifest.jsonl"

        # SAM Batch Inference
        # Outputs to n_dir/metrics/sam_iou.csv
        run_cmd([
            "python3", "scripts/run_sam_batch.py",
            "--out_root", str(n_dir),
            "--manifest", str(manifest_path),
            "--sam_ckpt", args.sam_ckpt,
            "--sam_model_type", args.model_type,
            "--prompt_modes", args.prompt_mode
        ])
    print(f"\n[COMPLETE] Noise sweep finished. Results in {args.stimuli_root}")

if __name__ == "__main__":
    main()