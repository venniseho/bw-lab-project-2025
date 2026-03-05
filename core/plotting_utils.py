"""
plotting_utils.py
-----------------
Reusable plotting templates for SAM evaluation metrics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_style():
    """Applies a consistent look and feel to all project plots."""
    sns.set_theme(style="whitegrid", context="talk")

def plot_identity_scatter(ax, x, y, title, xlabel, ylabel, limits=(-0.2, 1.05)):
    """Generic scatter plot with an identity line (y=x)."""
    ax.scatter(x, y, alpha=0.6, edgecolor='white', s=50)
    ax.plot([limits[0], limits[1]], [limits[0], limits[1]], "k--", linewidth=1, label="Identity")
    
    # Optional: Draw chance lines if limits allow
    if limits[0] < 0:
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_metric_distribution(ax, data_orig, data_frag, title, xlabel):
    """Histogram comparison of original vs fragmented performance."""
    sns.histplot(data_orig, color="blue", alpha=0.4, label="Original", kde=True, ax=ax)
    sns.histplot(data_frag, color="orange", alpha=0.4, label="Fragmented", kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()