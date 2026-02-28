"""
experiments/test_oracle_debug.py
--------------------------------
Runs a small test (50 images) with 10 random points to diagnose 
the Selection Gap (Oracle vs Model Choice).
"""
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--coco_imgdir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    args = ap.parse_args()

    # Output folder
    out_root = Path("outputs/tests/oracle_debug_n10")
    
    # 1. Generate Stimuli (Limit 50)
    run_cmd([
        "python3", "make_stimuli.py",
        "--coco_ann", args.coco_ann,
        "--coco_imgdir", args.coco_imgdir,
        "--out_root", str(out_root),
        "--limit", "50",
        "--noise_count", "300"
    ])

    # 2. Run SAM (10 Points to force the 'Selection Gap' behavior)
    run_cmd([
        "python3", "run_sam_batch.py",
        "--out_root", str(out_root),
        "--manifest", str(out_root / "indexes/manifest.jsonl"),
        "--sam_ckpt", args.sam_ckpt,
        "--prompt_mode", "random_point",
        "--n_points", "10" 
    ])

    print(f"\nDONE. Check debug panels in:\n{out_root}/debug/sam_overlays/")

if __name__ == "__main__":
    main()