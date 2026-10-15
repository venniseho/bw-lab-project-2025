"""CPU preparation from explicit local image/mask records; no SAM/GPU execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from data_selection.dataset import ImageMaskDataset
from stimuli_generation.fragmentation import FragmentationConfig
from stimuli_generation.transforms import ContourFragmentation, FenceOcclusion


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON list of __key__, image_path, mask_path; paths relative to manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--transform", choices=("fence", "fragmentation"), required=True)
    parser.add_argument("--widths", nargs=4, type=float, default=[0, 4, 12, 24])
    parser.add_argument("--spacing", type=float, default=40)
    parser.add_argument("--angle", type=float, default=90)
    parser.add_argument("--phase", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fragmentation-config", type=Path, help="JSON FragmentationConfig fields")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    for record in records:
        for field in ("image_path", "mask_path"):
            record[field] = str(args.manifest.resolve().parent / record[field])
    config = FragmentationConfig(**json.loads(args.fragmentation_config.read_text())) if args.fragmentation_config else FragmentationConfig()
    if args.transform == "fence":
        if args.widths[0] != 0 or not all(a < b for a, b in zip(args.widths, args.widths[1:])):
            parser.error("fence widths must start at zero and strictly increase")
        conditions = [(name, FenceOcclusion(width, args.spacing, args.angle, args.seed, args.phase))
                      for name, width in zip(("clean", "light", "medium", "heavy"), args.widths)]
    else:
        conditions = [("fragmentation", ContourFragmentation(config, args.seed))]
    dataset = ImageMaskDataset(records)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "inputs").mkdir()
    (args.output_dir / "targets").mkdir()
    report = {"status": "preparing", "transform": args.transform, "records": []}
    report_path = args.output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for index, sample in enumerate(dataset):
        for name, transform in conditions:
            output = transform(sample)
            stem = f"{index:06d}_{name}"
            image_path = f"inputs/{stem}.png"
            target_path = f"targets/{stem}.npz"
            Image.fromarray(output["image"]).save(args.output_dir / image_path)
            targets = {key: output[key] for key in ("segmentation", "visible_segmentation", "occlusion_mask") if key in output}
            if "fragmentation" in output:
                targets.update({key: output["fragmentation"][key] for key in ("outline_segments", "noise_segments")})
            np.savez_compressed(args.output_dir / target_path, **targets)
            report["records"].append({"__key__": sample["__key__"], "condition": name,
                                      "image": image_path, "targets": target_path,
                                      "transforms": output["transforms"],
                                      "metadata": output.get("metadata", {})})
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared {len(report['records'])} inputs in {args.output_dir}; SAM3 has not been run.")


if __name__ == "__main__":
    main()
