"""
plot_noise_sweep.py
-------------------
VISUALIZATION: Plots performance degradation vs. noise density.

Generates:
  1. Performance Curve: nIoU vs. Noise (Orig vs. Frag).
  2. Fragmentation Accuracy: The 'Delta' (gap) between Orig and Frag.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from core.plotting_utils import setup_style

def plot_noise_sensitivity(stats_df: pd.DataFrame, out_dir: Path):
    """Plots normalized performance across noise levels."""
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: nIoU vs Noise Density
    ax = axes[0]
    ax.errorbar(stats_df["noise_level"], stats_df["niou_orig_mean"], 
                yerr=stats_df["niou_orig_sem"], fmt='-o', label="Original")
    ax.errorbar(stats_df["noise_level"], stats_df["niou_frag_mean"], 
                yerr=stats_df["niou_frag_sem"], fmt='-o', label="Fragmented")
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title("SAM Sensitivity to Noise")
    ax.set_xlabel("Noise Density (Segments)")
    ax.set_ylabel("Normalised IoU (0 = Chance)")
    ax.legend()

    # Panel 2: The Fragmentation "Cost" (Delta)
    ax = axes[1]
    ax.errorbar(stats_df["noise_level"], stats_df["delta_niou_mean"], 
                yerr=stats_df["delta_niou_sem"], fmt='-o', color='red')
    
    ax.set_title("Fragmentation Cost Over Noise")
    ax.set_xlabel("Noise Density (Segments)")
    ax.set_ylabel("Δ nIoU (Original - Fragmented)")
    
    plt.tight_layout()
    plt.savefig(out_dir / "noise_sweep_analysis.png", dpi=200)
    plt.close()