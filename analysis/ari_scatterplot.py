"""
analysis/ari_scatterplot.py
---------------------------
Generates a scatterplot for Adjusted Rand Index (ARI).
ARI is already chance-adjusted (0 = random, 1 = perfect), so we don't normalize it further.
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
    
    # Check for ARI columns
    required = {"ari_orig", "ari_frag"}
    if not required.issubset(df.columns):
        print(f"Skipping ARI plot: Missing columns {required - set(df.columns)}")
        return

    # 2. Setup Plotting
    sns.set_theme(style="whitegrid")
    
    # We only need one panel since ARI doesn't have an "unnormalized" version
    plt.figure(figsize=(6, 6))
    
    # Scatterplot
    sns.scatterplot(data=df, x="ari_orig", y="ari_frag", alpha=0.7, edgecolor=None)
    
    # Identity line (y=x)
    plt.plot([-0.2, 1.0], [-0.2, 1.0], "k--", linewidth=1, label="Identity")
    
    # Zero lines (Chance level)
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.axvline(0, color="gray", linestyle=":", linewidth=1)

    plt.xlim(-0.2, 1.05)
    plt.ylim(-0.2, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title("Adjusted Rand Index (ARI)\n(0 = Chance, 1 = Perfect)")
    plt.xlabel("Original ARI")
    plt.ylabel("Fragmented ARI")
    
    out_path = out_dir / "scatter_ari.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved ARI scatter plot -> {out_path}")

if __name__ == "__main__":
    main()