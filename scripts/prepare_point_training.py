"""Cache a tiny, disjoint COCO train2017 subset for point-prompt fine-tuning."""

import argparse
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    return (parser, args)


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
    annotations = ensure_annotations(root, args.download)
    print(f"Reading training annotations: {annotations}", flush=True)
    splits = select_splits(
        json.loads(annotations.read_text()), args.train_images, args.heldout_images, args.seed
    )
    for name, split in splits.items():
        validate_masks(split)
        for image in split["images"]:
            filename = image["file_name"]
            if Path(filename).name != filename or not filename.endswith(".jpg"):
                raise ValueError(f"Unexpected COCO filename: {filename}")
            path = root / "train2017" / filename
            if args.download:
                print(f"{name}: caching {filename}", flush=True)
                download(f"{BASE}/train2017/{filename}", path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; add --download")
            image["file_name"] = str(path)
    output.mkdir(parents=True)
    for name, split in splits.items():
        (output / f"{name}.json").write_text(json.dumps(split) + "\n")
    config = {
        "native_root": str(native),
        "native_revision": revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "epochs": args.epochs,
        "workers": args.workers,
        "learning_rate": 1e-5,
        "train_annotations": str(output / "train.json"),
        "heldout_annotations": str(output / "test.json"),
        "annotation_sha256": {
            name: hashlib.sha256((output / f"{name}.json").read_bytes()).hexdigest()
            for name in splits
        },
        "source_split": "COCO train2017 only; val2017 untouched",
        "instance_policy": "all noncrowd instances in the selected sources",
        "transform": "identity",
        "positive_point": "foreground pixel nearest centroid",
        "loss": "20*focal(alpha=.25,gamma=2)+dice+quality_mse; minimum-mask candidate",
        "splits": {
            name: {
                "source_ids": [image["id"] for image in split["images"]],
                "instances": len(split["annotations"]),
            }
            for name, split in splits.items()
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)
    print("Preparation only; no model forward or training launched.", flush=True)


if __name__ == "__main__":
    main()
