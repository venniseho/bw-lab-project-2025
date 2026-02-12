"""
analysis/experiment_sam_random.py
------------------------------------------
Runs the full pipeline using Random Point prompts (Single or Multi-point).
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
    
    # NEW ARGUMENT: Controls single vs multi-point
    ap.add_argument("--n_points", type=str, default="1", help="Number of random points to sample")

    args = ap.parse_args()

    # Dynamic folder name based on number of points (e.g. random_points_1_exp, random_points_5_exp)
    exp_name = f"random_points_{args.n_points}_exp"
    out_root = Path(f"outputs/experiments/{exp_name}")
    
    # 1. Generate Stimuli (Stage 1)
    run_cmd([
        "python3", "make_stimuli.py",
        "--coco_ann", args.coco_ann,
        "--coco_imgdir", args.coco_imgdir,
        "--out_root", str(out_root),
        "--limit", args.limit,
        "--noise_count", "200"
    ])

    # 2. Stage 1.5: Verify Stats (No Local Cues)
    run_cmd([
        "python3", "analysis/stimuli_stats_check.py",
        "--metrics_dir", str(out_root / "debug/metrics"),
        "--out_dir", str(out_root / "debug/metrics/stats_check")
    ])

    # 3. Run SAM with RANDOM POINT prompts (Stage 2)
    # Note: We pass --n_points here
    run_cmd([
        "python3", "run_sam_batch.py",
        "--out_root", str(out_root),
        "--manifest", str(out_root / "indexes/manifest.jsonl"),
        "--sam_ckpt", args.sam_ckpt,
        "--prompt_mode", "random_point",
        "--n_points", args.n_points 
    ])

    # 4. Generate Plots (Stage 3)
    run_cmd([
        "python3", "analysis/plot_sam_results.py",
        "--csv", str(out_root / "metrics/sam_iou.csv"),
        "--out_dir", str(out_root / "plots")
    ])

if __name__ == "__main__":
    main()