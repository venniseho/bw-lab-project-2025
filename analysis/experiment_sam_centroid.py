"""
analysis/experiment_sam_centroid.py
--------------------------------------
Runs the full pipeline using a Single Centroid Point prompt.
"""
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--coco_imgdir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--limit", type=str, default="50")
    args = ap.parse_args()

    # 1. Setup
    exp_name = "centroid_prompt_exp"
    out_root = Path(f"outputs/experiments/{exp_name}")
    
    # 2. Stage 1: Make Stimuli
    run_cmd([
        "python3", "make_stimuli.py",
        "--coco_ann", args.coco_ann,
        "--coco_imgdir", args.coco_imgdir,
        "--out_root", str(out_root),
        "--limit", args.limit,
        "--noise_count", "300"
    ])

    # 3. Stage 1.5: Verify Stats (No Local Cues)
    run_cmd([
        "python3", "analysis/stimuli_stats_check.py",
        "--metrics_dir", str(out_root / "debug/metrics"),
        "--out_dir", str(out_root / "debug/metrics/stats_check")
    ])

    # 4. Stage 2: SAM with CENTROID prompts
    run_cmd([
        "python3", "run_sam_batch.py",
        "--out_root", str(out_root),
        "--manifest", str(out_root / "indexes/manifest.jsonl"),
        "--sam_ckpt", args.sam_ckpt,
        "--prompt_mode", "centroid"  # <--- Centroid Mode
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

if __name__ == "__main__":
    main()