"""
analysis/plot_noise_density_summary.py
--------------------------------------
Reads existing sam_iou.csv files from noise sweep folders,
calculates Mean & SEM, and generates the summary plot.
DOES NOT run SAM or generate images.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def normalise_iou(iou: pd.Series, chance: pd.Series) -> pd.Series:
    denom = (1.0 - chance).replace(0, pd.NA)
    return (iou - chance) / denom

def main():
    ap = argparse.ArgumentParser()
    # Adjust these defaults if your folder structure is different
    ap.add_argument("--out_root", default="outputs/experiments/noise_density", help="Root folder containing n50, n100, etc.")
    ap.add_argument("--noise_counts", type=int, nargs="+", default=[50, 100, 200, 300, 500, 800], 
                    help="List of noise counts to look for (must match folder names)")
    args = ap.parse_args()

    base_out = Path(args.out_root)
    summary_rows = []

    print(f"Reading results from: {base_out}")

    for nc in args.noise_counts:
        exp_dir = base_out / f"n{nc}"
        sam_csv = exp_dir / "metrics" / "sam_iou.csv"

        if not sam_csv.exists():
            print(f"Skipping n={nc}: CSV not found at {sam_csv}")
            continue

        # Load Data
        df = pd.read_csv(sam_csv)

        # Re-calculate Normalised IoU to ensure consistency
        df["norm_iou_orig"] = normalise_iou(df["iou_orig"], df["chance_iou"])
        df["norm_iou_frag"] = normalise_iou(df["iou_frag"], df["chance_iou"])

        # Calculate Stats (Mean & SEM)
        n = len(df)
        row = {
            "noise_count": nc,
            "n_instances": n,
            "mean_iou_orig": df["iou_orig"].mean(),
            "sem_iou_orig": df["iou_orig"].std() / np.sqrt(n),
            "mean_iou_frag": df["iou_frag"].mean(),
            "sem_iou_frag": df["iou_frag"].std() / np.sqrt(n),
            "mean_norm_iou_orig": df["norm_iou_orig"].mean(),
            "sem_norm_iou_orig": df["norm_iou_orig"].std() / np.sqrt(n),
            "mean_norm_iou_frag": df["norm_iou_frag"].mean(),
            "sem_norm_iou_frag": df["norm_iou_frag"].std() / np.sqrt(n),
        }
        summary_rows.append(row)
        print(f"Loaded n={nc} ({n} images)")

    if not summary_rows:
        print("No data found!")
        return

    # Create Summary DataFrame
    summary = pd.DataFrame(summary_rows).sort_values("noise_count")
    
    # Save Summary CSV
    out_csv = base_out / "noise_sweep_summary_replot.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV -> {out_csv}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Unnormalised IoU
    ax = axes[0]
    ax.errorbar(summary["noise_count"], summary["mean_iou_orig"], 
                yerr=summary["sem_iou_orig"], fmt='-o', capsize=4, label="Original")
    ax.errorbar(summary["noise_count"], summary["mean_iou_frag"], 
                yerr=summary["sem_iou_frag"], fmt='-o', capsize=4, label="Fragmented")
    
    ax.set_title("Mean IoU vs Noise Density")
    ax.set_xlabel("Noise Count")
    ax.set_ylabel("Mean IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Normalised IoU
    ax = axes[1]
    ax.errorbar(summary["noise_count"], summary["mean_norm_iou_orig"], 
                yerr=summary["sem_norm_iou_orig"], fmt='-o', capsize=4, label="Original")
    ax.errorbar(summary["noise_count"], summary["mean_norm_iou_frag"], 
                yerr=summary["sem_norm_iou_frag"], fmt='-o', capsize=4, label="Fragmented")
    
    ax.set_title("Normalised IoU vs Noise Density\n(0 = Chance)")
    ax.set_xlabel("Noise Count")
    ax.set_ylabel("Normalised IoU")
    ax.axhline(0.0, linestyle="--", color='gray', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = base_out / "noise_sweep_plot_replot.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved plot -> {out_png}")

if __name__ == "__main__":
    main()