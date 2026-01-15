"""
Command to run:
python -m pytest -q tests/test_05_pipeline_with_sam.py
"""

from pathlib import Path
import shutil
import subprocess
import sys
import pytest

OUT = Path("outputs/tests/test05_pipeline_with_sam")
CKPT = Path("checkpoints/sam_vit_h_4b8939.pth")


@pytest.mark.skipif(not CKPT.exists(), reason="SAM checkpoint missing")
def test_pipeline_runs_with_sam_and_writes_iou_csv():
    if OUT.exists():
        shutil.rmtree(OUT)

    cmd = [
        sys.executable, "coco_pipeline_revised.py",
        "--coco_ann", "COCO/annotations/instances_val2014.json",
        "--coco_imgdir", "COCO/val2014",
        "--out_root", str(OUT),
        "--limit", "1",
        "--sam_ckpt", str(CKPT),
        "--sam_model_type", "vit_h",
    ]
    subprocess.check_call(cmd)

    assert (OUT / "sam_overlays").exists()
    assert (OUT / "metrics" / "sam_iou.csv").exists()

    # at least one overlay should be produced if at least one instance is kept
    assert len(list((OUT / "sam_overlays").glob("*_sam_orig.png"))) >= 1
    assert len(list((OUT / "sam_overlays").glob("*_sam_frag.png"))) >= 1
