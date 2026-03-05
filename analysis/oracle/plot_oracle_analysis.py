"""
plot_oracle_debug.py
--------------------
VISUALIZATION: Diagnostic plots for SAM's multi-mask output.

Functions:
  - plot_calibration: Confidence vs. Oracle IoU scatter.
  - plot_oracle_index_dist: Histogram of which SAM index (0,1,2) was best.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from core.plotting_utils import setup_style

def plot_calibration(df: pd.DataFrame, out_path: Path):
    """
    Checks if SAM's confidence score correlates with actual quality.
    Ideally, high confidence = high Oracle IoU.
    """
    setup_style()
    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=df, x="score_frag", y="oracle_iou_frag", 
        alpha=0.6, color="#2ca02c", edgecolor=None
    )
    
    # Quadrant lines for calibration visualization
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.title("Calibration Check:\nConfidence Score vs. Best Possible IoU")
    plt.xlabel("SAM Confidence (Selected Mask)")
    plt.ylabel("Oracle IoU (Best of 3 Candidates)")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_oracle_index_dist(df: pd.DataFrame, out_path: Path):
    """
    Visualizes which of the 3 SAM output scales is most accurate.
    Index 0: Sub-part | Index 1: Object | Index 2: Whole/Ambiguous
    """
    if "oracle_idx_frag" not in df.columns:
        return

    setup_style()
    plt.figure(figsize=(7, 5))
    
    counts = df["oracle_idx_frag"].value_counts().sort_index()
    all_indices = [0, 1, 2]
    all_counts = [counts.get(i, 0) for i in all_indices]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(all_indices, all_counts, color=colors, alpha=0.8)
    
    plt.title("Which Mask scale is the 'Oracle' Choice?")
    plt.xlabel("SAM Output Index\n(0=Part, 1=Object, 2=Large)")
    plt.ylabel("Frequency")
    plt.xticks(all_indices, ["Index 0", "Index 1", "Index 2"])
    
    # Label bar heights
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()