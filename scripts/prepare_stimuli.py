"""CPU preparation from explicit local image/mask records; no SAM/GPU execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from bwlab.stimuli.fragmentation import FragmentationConfig


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
    config = FragmentationConfig(**json.loads(args.fragmentation_config.read_text(encoding="utf-8"))) if args.fragmentation_config else FragmentationConfig()
    from experiments.sam3_fragmentation.experiment import Config, prepare
    experiment = Config(manifest=args.manifest, condition=args.transform,
                        seed=args.seed, fence_widths=tuple(args.widths),
                        fence_spacing=args.spacing, fence_angle=args.angle,
                        fence_phase=args.phase, fragmentation=config)
    report = prepare(experiment, args.output_dir)
    print(f"Prepared {len(report['records'])} inputs in {args.output_dir}; SAM3 has not been run.")


if __name__ == "__main__":
    main()
