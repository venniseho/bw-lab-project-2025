"""Prepare one-target-per-image COCO splits for point-prompt experiments."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models.sam3 import VERIFIED_SAM3_API_REVISION
from core.data_loading.download import ensure_annotations
from core.data_loading.splits import select_splits, validate_masks
from core.data_loading.download import BASE, download
from core.data_loading.point_preparation import select_official_split


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/scratch/vho/sam3-whole-scene/checkpoints/sam3.pt")
    )
    parser.add_argument("--train-images", type=int, default=8)
    parser.add_argument("--heldout-images", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--official-splits",
        action="store_true",
        help="Use train2017 for training and val2017 for validation; image count 0 means all eligible images",
    )
    parser.add_argument(
        "--transform", choices=("identity", "fragmentation", "fence"), default="identity"
    )
    parser.add_argument("--fence-width", type=float, default=12)
    parser.add_argument("--fence-spacing", type=float, default=40)
    parser.add_argument("--validation-interval-steps", type=int)
    parser.add_argument(
        "--validation-image-count",
        type=int,
        help="Seeded subset for interval checks; annotations retain the full validation selection",
    )
    args = parser.parse_args()
    return (parser, args)


def _prepare_splits(args, root):
    annotations = ensure_annotations(root, args.download)
    print(f"Reading training annotations: {annotations}", flush=True)
    if args.official_splits:
        validation = ensure_annotations(root, args.download, split="val2017")
        splits = {
            "train": select_official_split(
                json.loads(annotations.read_text()), args.train_images, args.seed
            ),
            "test": select_official_split(
                json.loads(validation.read_text()), args.heldout_images, args.seed
            ),
        }
    else:
        splits = select_splits(
            json.loads(annotations.read_text()), args.train_images, args.heldout_images, args.seed
        )
        splits = {
            name: select_official_split(split, 0, args.seed) for name, split in splits.items()
        }
    for name, split in splits.items():
        label = "validation" if name == "test" else name
        source_split = "val2017" if args.official_splits and name == "test" else "train2017"
        validate_masks(split)
        for image in split["images"]:
            filename = image["file_name"]
            if Path(filename).name != filename or not filename.endswith(".jpg"):
                raise ValueError(f"Unexpected COCO filename: {filename}")
            path = root / source_split / filename
            if args.download:
                print(f"{label}: caching {filename}", flush=True)
                download(f"{BASE}/{source_split}/{filename}", path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; add --download")
            image["file_name"] = str(path)
    return splits


def _config(args, native, output, revision, splits):
    return {
        "native_root": str(native),
        "native_revision": revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "epochs": args.epochs,
        "workers": args.workers,
        "learning_rate": 1e-5,
        "train_annotations": str(output / "train.json"),
        "heldout_annotations": str(output / "validation.json"),
        "annotation_sha256": {
            name: hashlib.sha256(
                (output / f"{'validation' if name == 'test' else name}.json").read_bytes()
            ).hexdigest()
            for name in splits
        },
        "source_split": "COCO train2017 training; val2017 validation"
        if args.official_splits
        else "Disjoint subsets of COCO train2017",
        "instance_policy": "one seeded random noncrowd object per image",
        "one_object_per_image": True,
        "resolution": 1008,
        "validation_interval_steps": args.validation_interval_steps,
        "validation_image_count": args.validation_image_count,
        "transform": args.transform,
        "transform_parameters": {
            "bar_width": args.fence_width,
            "bar_spacing": args.fence_spacing,
            "bar_angle": 90.0,
        }
        if args.transform == "fence"
        else {},
        "positive_point": "uniform random foreground pixel, fixed per instance by seed",
        "loss": "20*focal(alpha=.25,gamma=2)+dice+quality_mse; minimum-mask candidate",
        "splits": {
            name: {
                "source_ids": [image["id"] for image in split["images"]],
                "instances": len(split["annotations"]),
                "selection_counts": split["selection_counts"],
            }
            for name, split in splits.items()
        },
    }


def main():
    parser, args = _parse_args()
    native, root, output = (
        args.native_root.resolve(),
        args.coco_root.resolve(),
        args.output_dir.resolve(),
    )
    revision = subprocess.check_output(
        ["git", "-C", str(native), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != VERIFIED_SAM3_API_REVISION:
        raise ValueError(f"Expected SAM3 {VERIFIED_SAM3_API_REVISION}; found {revision}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if output.exists():
        raise FileExistsError("Use a fresh preparation directory")
    if args.epochs < 1 or args.workers < 0:
        raise ValueError("Positive epochs and nonnegative workers required")
    if args.validation_interval_steps is not None and args.validation_interval_steps < 1:
        parser.error("Validation interval must be positive")
    if args.validation_image_count is not None and args.validation_image_count < 1:
        parser.error("Validation subset must be positive")
    splits = _prepare_splits(args, root)
    output.mkdir(parents=True)
    for name, split in splits.items():
        filename = "validation" if name == "test" else name
        (output / f"{filename}.json").write_text(json.dumps(split) + "\n")
    config = _config(args, native, output, revision, splits)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["split", "image_id", "annotation_id", "image_path", "target_area", "seed"],
        )
        writer.writeheader()
        for name, split in splits.items():
            images = {image["id"]: image for image in split["images"]}
            writer.writerows(
                {
                    "split": "validation" if name == "test" else name,
                    "image_id": ann["image_id"],
                    "annotation_id": ann["id"],
                    "image_path": images[ann["image_id"]]["file_name"],
                    "target_area": ann["area"],
                    "seed": args.seed,
                }
                for ann in split["annotations"]
            )
    print(f"Config: {output / 'config.json'}", flush=True)
    for name, split in splits.items():
        label = "validation" if name == "test" else name
        print(
            f"{label}: {len(split['images'])} images, {len(split['annotations'])} masks", flush=True
        )
    print("Preparation only; no model forward or training launched.", flush=True)


if __name__ == "__main__":
    main()
