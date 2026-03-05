"""
run_analysis.py
---------------
Entry point for all metric visualizations.
"""
import argparse
import pandas as pd
from pathlib import Path
from analysis.sam_performance import analyze_iou_and_ari

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/metrics/sam_results.csv")
    parser.add_argument("--out_dir", default="outputs/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    analyze_iou_and_ari(df, Path(args.out_dir))
    
    print(f"Analysis complete. Plots saved to: {args.out_dir}")

if __name__ == "__main__":
    main()