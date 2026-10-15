"""Rebuild a saved SAM3 contact sheet on CPU, without model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from bwlab.models.proposals import Proposal
from bwlab.visualization import all_proposal_overlay, contact_sheet
from bwlab.infrastructure import load_rgb


def relabel_scene(scene_dir: Path, output: Path) -> None:
    """Require original RGB, candidate masks and the original metadata ledger."""
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    rgb = np.array(load_rgb(scene_dir / "original.png"))
    rows = {row["proposal_id"]: row for row in report["proposals"]}
    retained = []
    for identifier in report["retained_proposal_ids"]:
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
            raise ValueError("invalid retained proposal ID")
        row = rows[identifier]
        path = scene_dir / f"retained/proposal_{identifier:05d}.png"
        if not path.is_file():
            path = scene_dir / f"candidates/proposal_{identifier:05d}.png"
        with Image.open(path) as source:
            mask = np.array(source)
        if mask.shape != rgb.shape[:2] or not np.all((mask == 0) | (mask == 255)):
            raise ValueError("archived mask must be aligned binary HW")
        if int(np.count_nonzero(mask)) != row["area_pixels"]:
            raise ValueError("archived mask area disagrees with metadata")
        retained.append(Proposal(np.packbits(mask != 0), mask.shape, row))
    sheet = contact_sheet(rgb, all_proposal_overlay(rgb, retained), retained,
                          prompts=report["proposals"], title=report.get("__key__") or report.get("condition"))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an older sheet even if the path appears late.
    with output.open("xb") as target:
        sheet.save(target, format="PNG")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New PNG path")
    args = parser.parse_args(argv)
    relabel_scene(args.scene_dir, args.output)
    print(f"Saved labeled sheet: {args.output}")


if __name__ == "__main__":
    main()
