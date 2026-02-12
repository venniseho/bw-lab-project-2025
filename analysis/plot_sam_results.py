"""
analysis/plot_sam_results.py
----------------------------
Generates scatterplots (Normalized vs Unnormalized) and histograms for SAM results.

*combination of analysis/iou_scatterplot_normalised.py and analysis/plot_noise_density.py
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sam_iou.csv")
    ap.add_argument("--out_dir", required=True, help="Folder to save plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    required = {"iou_orig", "iou_frag", "chance_iou"}
    if not required.issubset(df.columns):
        print(f"Skipping plot: Missing columns {required - set(df.columns)}")
        return

    # 2. Calculate Normalized IoU (if not present)
    # (IoU - chance) / (1 - chance)
    if "niou_orig" not in df.columns:
        df["niou_orig"] = (df["iou_orig"] - df["chance_iou"]) / (1.0 - df["chance_iou"])
        df["niou_frag"] = (df["iou_frag"] - df["chance_iou"]) / (1.0 - df["chance_iou"])
    
    # Clip strictly for visualization stability
    df["niou_orig_plot"] = df["niou_orig"].clip(-0.2, 1.05)
    df["niou_frag_plot"] = df["niou_frag"].clip(-0.2, 1.05)

    sns.set_theme(style="whitegrid")

    # --- Plot 1: Scatterplots (Your original code, adapted) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Unnormalised
    sns.scatterplot(ax=axes[0], data=df, x="iou_orig", y="iou_frag", alpha=0.6)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("Unnormalised IoU")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_aspect('equal')

    # Right: Normalised
    sns.scatterplot(ax=axes[1], data=df, x="niou_orig_plot", y="niou_frag_plot", alpha=0.6)
    axes[1].plot([-0.2, 1.0], [-0.2, 1.0], "k--", linewidth=1)
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].axvline(0, color="gray", linestyle=":")
    axes[1].set_title("Normalised IoU (0=Chance)")
    axes[1].set_xlim(-0.2, 1.05); axes[1].set_ylim(-0.2, 1.05)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(out_dir / "scatter_iou.png", dpi=150)
    plt.close()

    # --- Plot 2: Histograms (Added bonus) ---
    plt.figure(figsize=(8, 5))
    sns.histplot(df["iou_orig"], color="blue", alpha=0.4, label="Original", kde=True, bins=15)
    sns.histplot(df["iou_frag"], color="orange", alpha=0.4, label="Fragmented", kde=True, bins=15)
    plt.title("Distribution of IoU Scores")
    plt.xlabel("IoU")
    plt.legend()
    plt.savefig(out_dir / "hist_iou.png", dpi=150)
    plt.close()
    
    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    main()