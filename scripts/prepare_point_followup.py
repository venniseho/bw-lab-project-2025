"""Prepare half-data comparisons from existing COCO manifests without downloading images."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data_loading.point_followup import filter_targets, training_subset


CONDITIONS = {
    "fragmentation": ("fragmentation", {}),
    "fence-low": ("fence", {"bar_width": 4, "bar_spacing": 40, "bar_angle": 90.0}),
    "fence-medium": ("fence", {"bar_width": 12, "bar_spacing": 40, "bar_angle": 90.0}),
    "fence-high": ("fence", {"bar_width": 24, "bar_spacing": 40, "bar_angle": 90.0}),
}


def prepare(base_path, output, fraction=0.5, interval=5000, fence_loss_region="full"):
    if interval < 1 or fence_loss_region not in ("full", "visible"):
        raise ValueError("Positive interval and full/visible loss region required")
    if output.exists():
        raise FileExistsError("Choose a fresh preparation directory")
    base = json.loads(base_path.read_text())
    data = {}
    for split, key in (("train", "train_annotations"), ("test", "heldout_annotations")):
        path = Path(base[key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != base["annotation_sha256"][split]:
            raise ValueError(f"Prepared {split} annotations changed")
        data[split] = json.loads(path.read_text())
    train_ids = training_subset(data["train"], fraction, base["seed"])
    val_ids = sorted(image["id"] for image in data["test"]["images"])
    output.mkdir(parents=True)
    (output / "requested_training_ids.json").write_text(json.dumps(train_ids) + "\n")
    counts = []
    for label, (transform, parameters) in CONDITIONS.items():
        root = output / label
        root.mkdir()
        config = deepcopy(base)
        config.update(
            protocol_version=2,
            epochs=1,
            transform=transform,
            transform_parameters=parameters,
            loss_region=fence_loss_region if transform == "fence" else "full",
            drop_fully_hidden=True,
            validation_interval_steps=interval,
            validation_image_count=None,
            checkpoint_interval_steps=1000,
            download_ahead=False,
            requested_training_fraction=fraction,
        )
        config["checkpoint_metric"] = (
            "mean_visible_iou" if config["loss_region"] == "visible" else "mean_full_iou"
        )
        config.pop("cached_training_images_at_preparation", None)
        with (root / "excluded_targets.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["split", "image_id", "annotation_id", "reason"]
            )
            writer.writeheader()
            for split, ids in (("train", train_ids), ("test", val_ids)):
                selected, order = filter_targets(
                    data[split], ids, transform, parameters, base["seed"]
                )
                name = "train" if split == "train" else "validation"
                path = root / f"{name}.json"
                path.write_text(json.dumps(selected) + "\n")
                key = "train_annotations" if split == "train" else "heldout_annotations"
                config[key] = str(path.resolve())
                config["annotation_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
                config["splits"][split] = {
                    "source_ids": order,
                    "instances": len(order),
                    "selection_counts": selected["selection_counts"],
                }
                if split == "train":
                    config["training_order"] = order
                writer.writerows({"split": name, **row} for row in selected["selection_audit"])
                record = {"condition": label, "split": name, **selected["selection_counts"]}
                counts.append(record)
                print(json.dumps(record), flush=True)
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (output / "sample_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    print(f"Prepared comparisons: {output.resolve()}", flush=True)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--validation-interval-steps", type=int, default=5000)
    parser.add_argument("--fence-loss-region", choices=("full", "visible"), default="full")
    args = parser.parse_args()
    prepare(
        args.base_config,
        args.output_dir,
        args.train_fraction,
        args.validation_interval_steps,
        args.fence_loss_region,
    )
