"""
oracle_verification.py
------------------------
ANALYSIS: Diagnostic logic for Oracle vs. Model selection.

This module provides tools to:
  1. Analyze the 'Selection Gap': Why SAM might choose a lower-IoU
     mask despite a better one being available in its output.
  2. Map SAM's internal indices (0, 1, 2) to hierarchical levels.
"""

import pandas as pd
import numpy as np

def calculate_selection_accuracy(df: pd.DataFrame) -> float:
    """
    Calculates how often SAM's automatically selected mask 
    is actually the best one (the Oracle).
    """
    if "oracle_idx_frag" not in df.columns:
        return 0.0
    # In run_sam_batch, we assume index 1 is often SAM's default object choice
    # Adjust this logic based on how your engine flags its 'chosen' index
    correct_selection = df[df["iou_frag"] >= (df["oracle_iou_frag"] - 1e-5)]
    return len(correct_selection) / len(df) if len(df) > 0 else 0.0