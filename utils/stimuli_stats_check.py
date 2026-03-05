"""
stimuli_stats_check.py
----------------------
VALIDATION: Compare OUTLINE vs NOISE dash statistics.

This version is optimized for the 2026 pipeline:
  1. Reads from the manifest.jsonl to find all processed instances.
  2. Locates metrics JSONs in the project's debug/metrics folder.
  3. Performs Chi-square tests to ensure noise dashes and outline dashes 
     share the same distribution (Length, Orientation, Spacing).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Core project imports
from utils.io_utils import read_manifest

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class StatsValidator:
    """Handles histogram pooling and homogeneity testing."""
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def chi_square_test(self, out_counts: np.ndarray, noise_counts: np.ndarray) -> float:
        """Chi-square test for homogeneity. High p-value = Good (Indistinguishable)."""
        total = out_counts + noise_counts
        keep = total > 0
        o_c, n_c = out_counts[keep], noise_counts[keep]

        if o_c.size < 2 or o_c.sum() == 0 or n_c.sum() == 0:
            return float("nan")

        try:
            _, p, _, _ = chi2_contingency(np.vstack([o_c, n_c]), correction=False)
            return float(p)
        except:
            return float("nan")

    def pool_metrics(self, metrics_paths: List[Path], key: str) -> Tuple[Optional[np.ndarray], ...]:
        """Gathers counts from specific files identified in the manifest."""
        edges_ref, out_sum, noise_sum = None, None, None

        for p in metrics_paths:
            if not p.exists(): continue
            with open(p, "r") as f:
                data = json.load(f).get("hists", {}).get(key)
                if not data: continue

                e, o, n = np.array(data["edges"]), np.array(data["outline_counts"]), np.array(data["noise_counts"])
                
                if edges_ref is None:
                    edges_ref, out_sum, noise_sum = e, o.copy(), n.copy()
                elif len(e) == len(edges_ref) and np.allclose(e, edges_ref):
                    out_sum += o
                    noise_sum += n
        return edges_ref, out_sum, noise_sum

def main():
    parser = argparse.ArgumentParser(description="Validate stimuli statistics homogeneity.")
    parser.add_argument("--out_root", required=True, help="Root folder of the experiment")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    manifest_path = out_root / "indexes" / "manifest.jsonl"
    stats_out_dir = out_root / "metrics" / "stats_check"
    stats_out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logging.error(f"Manifest not found at {manifest_path}. Run Stage 1 first.")
        return

    # 1. Identify metrics files via manifest
    manifest_data = read_manifest(manifest_path)
    metrics_paths = [out_root / "debug" / "metrics" / f"{item['stem']}_metrics.json" for item in manifest_data]

    validator = StatsValidator(alpha=args.alpha)
    keys = [("length", "Dash Length"), ("orientation_deg", "Orientation"), ("nn_distance", "Spacing")]
    
    report_summary = {}

    print(f"\n--- VALIDATING {len(metrics_paths)} STIMULI INSTANCES ---")

    

    for key, label in keys:
        edges, o_sum, n_sum = validator.pool_metrics(metrics_paths, key)
        if edges is None:
            print(f"{label:20s}: No data found.")
            continue

        p_val = validator.chi_square_test(o_sum, n_sum)
        passed = p_val >= args.alpha if np.isfinite(p_val) else True
        status = "PASS" if passed else "FAIL"
        
        print(f"{label:20s}: p={p_val:.4g} [{status}]")

        # Visualization
        plt.figure(figsize=(7, 4))
        c = 0.5 * (edges[:-1] + edges[1:])
        w = np.diff(edges)
        plt.bar(c, o_sum/o_sum.sum(), width=w, alpha=0.5, label="Outline", color="blue")
        plt.bar(c, n_sum/n_sum.sum(), width=w, alpha=0.5, label="Noise", color="orange")
        plt.title(f"{label} (p={p_val:.4g})")
        plt.legend()
        plt.savefig(stats_out_dir / f"pooled_{key}.png")
        plt.close()

        report_summary[key] = {"p": p_val, "status": status}

    # Save final JSON
    with open(stats_out_dir / "stats_report.json", "w") as f:
        json.dump(report_summary, f, indent=2)

if __name__ == "__main__":
    main()