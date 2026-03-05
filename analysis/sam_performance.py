"""
sam_performance.py
------------------
Calculates final metrics and generates standardized scatterplots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from core.plotting_utils import plot_identity_scatter, plot_metric_distribution, setup_style

def analyze_iou_and_ari(df: pd.DataFrame, out_dir: Path):
    """Generates the IoU and ARI scatter plots with chance level overlays."""
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Math: Standard Normalization (Global Chance)
    if "niou_orig" not in df.columns and "chance_iou" in df.columns:
        # Standard normalization against whole-image chance
        df["niou_orig"] = (df["iou_orig"] - df["chance_iou"]) / (1.0 - df["chance_iou"])
        df["niou_frag"] = (df["iou_frag"] - df["chance_iou"]) / (1.0 - df["chance_iou"])

    # 2. Math: Box-Prompt Specific Chance (The Selection Gap)
    # We only do this if box_area was recorded in the Stage 2 CSV
    has_box_info = "box_area" in df.columns and (df["box_area"] > 0).any()
    if has_box_info:
        # Box Chance = What IoU would a solid box mask get?
        df["box_chance"] = df["gt_area"] / df["box_area"]
        # nIoU_box = How much better is SAM than just the box itself?
        df["niou_box_frag"] = (df["iou_frag"] - df["box_chance"]) / (1.0 - df["box_chance"])

    # 3. Scatter Plot: Raw vs Normalized
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_identity_scatter(axes[0], df["iou_orig"], df["iou_frag"], 
                          "Raw IoU", "IoU (Original)", "IoU (Fragmented)", limits=(0, 1))
    
    # Add a horizontal line for the average Box Chance if applicable
    if has_box_info:
        avg_box_chance = df["box_chance"].mean()
        axes[0].axhline(y=avg_box_chance, color='red', linestyle='--', 
                        label=f"Box-Prompt Baseline ({avg_box_chance:.2f})")
        axes[0].legend()

    plot_identity_scatter(axes[1], df["niou_orig"].clip(-0.2, 1.05), df["niou_frag"].clip(-0.2, 1.05), 
                          "Global Normalized IoU", "nIoU (Original)", "nIoU (Fragmented)")
    
    plt.tight_layout()
    plt.savefig(out_dir / "iou_comparison_scatter.png", dpi=200)

    # 4. Bounding Box IOU Comparison plot (Only for Box Mode)
    if has_box_info:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plotting SAM performance relative to the information in the box
        plot_identity_scatter(ax, df["box_chance"], df["iou_frag"],
                              "SAM vs. Box Information", "Box Baseline (Chance)", "SAM IoU (Actual)")
        # If dots are above the identity line, SAM is adding value beyond the box
        plt.savefig(out_dir / "box_selection_gap_scatter.png", dpi=200)

    # 5. Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_metric_distribution(ax, df["iou_orig"], df["iou_frag"], 
                             "IoU Distribution Shift", "IoU Score")
    plt.savefig(out_dir / "iou_distribution_hist.png", dpi=200)
    plt.close('all')