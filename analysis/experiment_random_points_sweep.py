"""
analysis/experiment_random_points_sweep.py
---------------------------------------------
1. Runs the full pipeline for increasing numbers of random points.
2. Aggregates results across all runs.
3. Generates a summary plot (Performance vs N Points) with error bars.
"""
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.check_call(cmd)

def load_and_aggregate(csv_path, n_points):
    """Reads a single experiment CSV and calculates Mean/SEM."""
    df = pd.read_csv(csv_path)
    
    # Define metrics to aggregate
    metrics = {
        "niou": ("niou_orig", "niou_frag"),
        "ari": ("ari_orig", "ari_frag"),
        "oracle": ("oracle_iou_orig", "oracle_iou_frag")
    }
    
    stats = {"n_points": n_points, "n_samples": len(df)}
    
    for name, (col_orig, col_frag) in metrics.items():
        # Original
        stats[f"{name}_orig_mean"] = df[col_orig].mean()
        stats[f"{name}_orig_sem"] = df[col_orig].std() / np.sqrt(len(df))
        
        # Fragmented
        stats[f"{name}_frag_mean"] = df[col_frag].mean()
        stats[f"{name}_frag_sem"] = df[col_frag].std() / np.sqrt(len(df))
        
    return stats

def plot_sweep(summary_df, out_dir):
    """Generates the line plot with error bars (like your noise density plot)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting configuration for different metrics
    plot_configs = [
        ("niou", "Normalized IoU (0 = Chance)", "sweep_niou.png"),
        ("ari", "Adjusted Rand Index", "sweep_ari.png"),
        ("oracle", "Oracle IoU (Best of 3)", "sweep_oracle.png")
    ]
    
    for metric, ylabel, filename in plot_configs:
        plt.figure(figsize=(8, 6))
        
        # Original Line
        plt.errorbar(
            summary_df["n_points"], 
            summary_df[f"{metric}_orig_mean"], 
            yerr=summary_df[f"{metric}_orig_sem"],
            fmt='-o', label=f"Original ({ylabel})", capsize=3, color="#1f77b4"
        )
        
        # Fragmented Line
        plt.errorbar(
            summary_df["n_points"], 
            summary_df[f"{metric}_frag_mean"], 
            yerr=summary_df[f"{metric}_frag_sem"],
            fmt='-D', label=f"Fragmented ({ylabel})", capsize=3, color="#ff7f0e"
        )
        
        # Styling
        plt.xlabel("Number of Random Points")
        plt.ylabel(ylabel)
        plt.title(f"SAM Performance vs Number of Prompts")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(summary_df["n_points"]) # Ensure all Ns are shown on axis
        
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()
        print(f"Saved plot: {out_dir / filename}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--coco_imgdir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--limit", type=str, default="50")
    
    # Sweep configuration
    sweep_points = [1, 3, 5, 10]

    args = ap.parse_args()
    
    base_out = Path("outputs/experiments/random_points_sweep")
    base_out.mkdir(parents=True, exist_ok=True)
    
    results = []

    # --- 1. Run Experiments Loop ---
    for n in sweep_points:
        print(f"\n{'='*40}\nSTARTING EXPERIMENT FOR N={n} POINTS\n{'='*40}")
        
        # Sub-experiment folder
        exp_name = f"run_n{n}"
        out_root = base_out / exp_name
        
        # Step A: Generate Stimuli
        run_cmd([
            "python3", "make_stimuli.py",
            "--coco_ann", args.coco_ann,
            "--coco_imgdir", args.coco_imgdir,
            "--out_root", str(out_root),
            "--limit", args.limit,
            "--noise_count", "300"
        ])

        # Step B: Run SAM (Multi-point)
        run_cmd([
            "python3", "run_sam_batch.py",
            "--out_root", str(out_root),
            "--manifest", str(out_root / "indexes/manifest.jsonl"),
            "--sam_ckpt", args.sam_ckpt,
            "--prompt_mode", "random_point",
            "--n_points", str(n)
        ])
        
        # Step C: Collect Data
        csv_path = out_root / "metrics/sam_iou.csv"
        if csv_path.exists():
            row = load_and_aggregate(csv_path, n)
            results.append(row)
        else:
            print(f"Warning: Results missing for n={n}")

    # --- 2. Aggregation & Plotting ---
    print(f"\n{'='*40}\nGENERATING SUMMARY PLOTS\n{'='*40}")
    
    if results:
        summary_df = pd.DataFrame(results).sort_values("n_points")
        summary_csv = base_out / "sweep_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary CSV saved to: {summary_csv}")
        
        plot_sweep(summary_df, base_out)
    else:
        print("No results collected to plot.")

if __name__ == "__main__":
    main()