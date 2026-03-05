"""
noise_sweep_aggregator.py
--------------------------------------------------------------
ANALYSIS: Aggregates results across multiple noise density levels.

This module crawls an experiment directory, identifies "noise_nXXX" 
folders, and calculates the fragmentation cost (Original vs Fragmented) 
as a function of noise density.

Outputs:
  - noise_sweep_summary.csv: Pooled Mean/SEM stats per noise level.
"""

from __future__ import annotations
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Matches folder names like 'n50', 'noise_100', 'noise200'
NOISE_PATTERN = re.compile(r"(?:noise_?|n)(\d+)", re.IGNORECASE)

def parse_noise_level(path: Path) -> int | None:
    """Extracts the noise count integer from a folder name."""
    match = NOISE_PATTERN.search(path.name)
    return int(match.group(1)) if match else None

def aggregate_sweep_results(root_dir: Path) -> pd.DataFrame:
    """
    Finds all metrics CSVs in subfolders and aggregates them 
    by their parsed noise level.
    """
    all_data = []
    
    # Look for sam_iou.csv in any subfolder
    for csv_path in root_dir.glob("**/metrics/sam_iou.csv"):
        noise_level = parse_noise_level(csv_path.parent.parent)
        if noise_level is None: continue
        
        df = pd.read_csv(csv_path)
        df["noise_level"] = noise_level
        
        # Calculate Normalised IoU (0 = chance)
        denom = (1.0 - df["chance_iou"]).replace(0, np.nan)
        df["niou_orig"] = (df["iou_orig"] - df["chance_iou"]) / denom
        df["niou_frag"] = (df["iou_frag"] - df["chance_iou"]) / denom
        df["delta_niou"] = df["niou_orig"] - df["niou_frag"]
        
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    merged = pd.concat(all_data, ignore_index=True)
    
    # Group by noise level and calculate Mean + SEM
    stats = merged.groupby("noise_level").agg({
        "niou_orig": ["mean", lambda x: x.std() / np.sqrt(len(x))],
        "niou_frag": ["mean", lambda x: x.std() / np.sqrt(len(x))],
        "delta_niou": ["mean", lambda x: x.std() / np.sqrt(len(x))]
    })
    
    # Flatten columns: 'niou_orig_mean', 'niou_orig_sem', etc.
    stats.columns = [f"{c[0]}_{'mean' if c[1]=='mean' else 'sem'}" for c in stats.columns]
    return stats.reset_index().sort_values("noise_level")