"""
Command to run:
python -m pytest -q tests/test_04_pipeline_no_sam.py
"""

from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path("outputs/tests/test04_pipeline_no_sam")


def test_pipeline_runs_and_creates_outputs():
    if OUT.exists():
        shutil.rmtree(OUT)

    cmd = [
        sys.executable, "coco_pipeline_revised.py",
        "--coco_ann", "COCO/annotations/instances_val2014.json",
        "--coco_imgdir", "COCO/val2014",
        "--out_root", str(OUT),
        "--limit", "1",
    ]
    subprocess.check_call(cmd)

    # required dirs created by fragmenter + pipeline
    for d in ["images", "masks", "fragments", "outlines", "panels", "metrics"]:
        assert (OUT / d).exists(), f"Missing output dir: {d}"

    assert len(list((OUT / "images").glob("*.png"))) >= 1
    # per-instance masks now
    assert len(list((OUT / "masks").glob("*_ann*.png"))) >= 1
    assert len(list((OUT / "fragments").glob("*_fragmented.png"))) >= 1
