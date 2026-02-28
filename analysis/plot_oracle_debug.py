"""
analysis/plot_oracle_debug.py
-----------------------------
1. Scatter: SAM Confidence vs. Oracle IoU
2. Histogram: Frequency of Oracle's preferred mask index (0, 1, 2)
"""
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sam_iou.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder for plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # -------------------------------------------------------
    # Plot 1: Scatter (Confidence vs Oracle IoU)
    # -------------------------------------------------------
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df, 
        x="score_frag", 
        y="oracle_iou_frag", 
        alpha=0.6, 
        edgecolor=None,
        color="#2ca02c" # Green for Oracle
    )
    
    plt.title("Calibration Check:\nSAM Confidence vs. Best Possible IoU")
    plt.xlabel("SAM Confidence Score (of selected mask)")
    plt.ylabel("Oracle IoU (Best of 3)")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add quadrants lines
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "scatter_conf_vs_oracle.png", dpi=150)
    plt.close()
    print(f"Saved scatter -> {out_dir / 'scatter_conf_vs_oracle.png'}")

    # -------------------------------------------------------
    # Plot 2: Histogram of Oracle Indices (0, 1, 2)
    # -------------------------------------------------------
    if "oracle_idx_frag" in df.columns:
        plt.figure(figsize=(6, 5))
        
        # Count frequencies of 0, 1, 2
        counts = df["oracle_idx_frag"].value_counts().sort_index()
        
        # Ensure 0, 1, 2 are all present in x-axis even if count is 0
        all_indices = [0, 1, 2]
        all_counts = [counts.get(i, 0) for i in all_indices]
        
        bars = plt.bar(all_indices, all_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        
        plt.title("Which Mask is the 'Oracle' Choice?")
        plt.xlabel("Mask Index Output by SAM\n(0=Sub-part, 1=Object, 2=Ambiguous/Large)")
        plt.ylabel("Count")
        plt.xticks([0, 1, 2], ["Index 0", "Index 1", "Index 2"])
        plt.grid(axis='y', alpha=0.3)
        
        # Add text labels on top
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(out_dir / "hist_oracle_idx.png", dpi=150)
        plt.close()
        print(f"Saved histogram -> {out_dir / 'hist_oracle_idx.png'}")
    else:
        print("Skipping Histogram: 'oracle_idx_frag' column not found in CSV. (Did you update sam_runner.py?)")

if __name__ == "__main__":
    main()