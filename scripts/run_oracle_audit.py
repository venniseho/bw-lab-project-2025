"""
run_oracle_audit.py
-------------------
EXECUTION: Orchestrates the oracle analysis pipeline.

Loads the latest results and generates calibration/index plots 
to determine if SAM is choosing the correct hierarchical level.
"""

import argparse
import pandas as pd
from pathlib import Path
from analysis.oracle.plot_oracle_analysis import plot_calibration, plot_oracle_index_dist
from analysis.oracle.oracle_verification import calculate_selection_accuracy

def main():
    parser = argparse.ArgumentParser(description="Run SAM Oracle Audit")
    parser.add_argument("--csv", default="outputs/metrics/sam_results.csv")
    parser.add_argument("--out_dir", default="outputs/plots/oracle_debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    print(f"\n--- Oracle Audit for {len(df)} instances ---")
    
    # 1. Selection Accuracy
    acc = calculate_selection_accuracy(df)
    print(f"SAM Selection Accuracy: {acc*100:.1f}% (How often it picks the best mask)")

    # 2. Visuals
    plot_calibration(df, out_dir / "calibration_scatter.png")
    plot_oracle_index_dist(df, out_dir / "oracle_index_hist.png")

    print(f"Audit Complete. Results saved to: {out_dir}")

if __name__ == "__main__":
    main()