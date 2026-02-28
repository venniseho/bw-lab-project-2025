"""
analysis/test_sam3_centroid.py
---------------------------------
Quick check to see if SAM 3 fixes the "Selection Gap" 
using a single centroid prompt.
"""
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam_ckpt", required=True) 
    args = ap.parse_args()

    out_root = Path("outputs/tests/sam3_centroid_debug")
    
    # 1. Generate Stimuli (Limit 50)
    # run_cmd([
    #     "python3", "make_stimuli.py",
    #     "--coco_ann", args.coco_ann,
    #     "--coco_imgdir", args.coco_imgdir,
    #     "--out_root", str(out_root),
    #     "--limit", "50",
    #     "--noise_count", "300"
    # ])

    # 2. Run SAM 3 (Centroid Mode)
    run_cmd([
        "python3", "run_sam_batch.py",
        "--out_root", str(out_root),
        "--manifest", "outputs/experiments/centroid_prompt_exp/indexes/manifest.jsonl",  # <-- Using the same stimuli as the centroid experiment
        "--sam_ckpt", args.sam_ckpt,
        "--sam_model_type", "sam3",       # <-- SAM 3 logic
        "--prompt_mode", "centroid",  # <-- Centroid prompt mode
    ])

    # 5. Analysis: Plot IoU Results
    run_cmd([
        "python3", "analysis/plot_sam_results.py",
        "--csv", str(out_root / "metrics/sam_iou.csv"),
        "--out_dir", str(out_root / "plots")
    ])

    # 6. Analysis: Plot ARI Results
    run_cmd([
        "python3", "analysis/ari_scatterplot.py",
        "--csv", str(out_root / "metrics/sam_iou.csv"),
        "--out_dir", str(out_root / "plots")
    ])

    print(f"\nDONE. Outputs in:\n{out_root}/debug/sam_overlays/")

if __name__ == "__main__":
    main()