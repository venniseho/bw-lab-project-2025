"""Prepare matched LR/accumulation pilots and balanced mixtures from cached targets."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data_loading.point_followup import filter_targets
from core.stimuli.point_mixture import PointMixture


FENCES = {
    "fence-low": {"bar_width": 4, "bar_spacing": 40, "bar_angle": 90.0},
    "fence-medium": {"bar_width": 12, "bar_spacing": 40, "bar_angle": 90.0},
    "fence-high": {"bar_width": 24, "bar_spacing": 40, "bar_angle": 90.0},
}


def subset(data, ids):
    wanted = set(ids)
    images = [item for item in data["images"] if item["id"] in wanted]
    annotations = [item for item in data["annotations"] if item["image_id"] in wanted]
    if (
        len(images) != len(ids)
        or len(annotations) != len(ids)
        or len({item["image_id"] for item in annotations}) != len(ids)
    ):
        raise ValueError("Expected one saved target per requested source image")
    return {"images": images, "annotations": annotations, "categories": data["categories"]}


def manifest(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")
    return str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()


def attach(config, split, path, digest, ids):
    key = "train_annotations" if split == "train" else "heldout_annotations"
    config[key] = path
    config["annotation_sha256"][split] = digest
    config["splits"][split] = {"instances": len(ids), "source_ids": ids}
    if split == "train":
        config["training_order"] = ids


def prepare(base_path, output, pilot_images=5000):
    base = json.loads(base_path.read_text())
    if (
        base["transform"] != "fence"
        or base.get("transform_parameters") != FENCES["fence-medium"]
        or base.get("loss_region", "full") != "full"
    ):
        raise ValueError("Use the saved fixed-rate full-loss medium-fence control")
    if (
        base.get("lr_schedule", {}).get("name", "constant") != "constant"
        or base.get("validation_image_count") is not None
    ):
        raise ValueError("Requires constant LR and the full prepared validation cohort")
    data = {}
    for split, key in (("train", "train_annotations"), ("test", "heldout_annotations")):
        path = Path(base[key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != base["annotation_sha256"][split]:
            raise ValueError(f"Saved {split} annotations changed")
        data[split] = json.loads(path.read_text())
    if not 16 < pilot_images <= len(base["training_order"]):
        raise ValueError("Pilot image count must exceed 16 and fit the training cohort")
    output.mkdir(parents=True, exist_ok=False)
    pilot_ids = base["training_order"][:pilot_images]
    pilot_manifest = manifest(output / "data/pilot-train.json", subset(data["train"], pilot_ids))
    base.update(
        epochs=1,
        learning_rate=1e-5,
        lr_schedule={"name": "constant"},
        loss_region="full",
        checkpoint_metric="mean_full_iou",
        validation_interval_steps=5000,
        checkpoint_interval_steps=1000,
        download_ahead=False,
        accumulation_steps=1,
    )
    base.pop("train_mixture", None)
    jobs = []

    def save(name, config, evaluate):
        path = output / name / "config.json"
        path.parent.mkdir()
        (path).write_text(json.dumps(config, indent=2) + "\n")
        count = config["splits"]["train"]["instances"]
        record = {
            "name": name,
            "config": str(path.resolve()),
            "images": count,
            "optimizer_updates": math.ceil(count / config["accumulation_steps"]),
            "evaluate_all": evaluate,
        }
        jobs.append(record)
        print(json.dumps(record), flush=True)

    for accumulation in (1, 16):
        for label, rate in (("1e-5", 1e-5), ("3e-5", 3e-5), ("1e-4", 1e-4)):
            config = deepcopy(base)
            config.update(
                learning_rate=rate,
                accumulation_steps=accumulation,
                comparison_group="lr-accumulation-pilot",
            )
            attach(config, "train", *pilot_manifest, pilot_ids)
            config["parent_training_fraction"] = config.pop("requested_training_fraction", None)
            save(f"pilot-lr-{label}-acc{accumulation}", config, False)

    specifications = {
        "matched-clean": ["identity"],
        "matched-fragmentation": ["fragmentation"],
        "mix-clean-fragmentation": ["identity", "fragmentation"],
        "mix-clean-fence": ["identity", "fence"],
        "mix-clean-fragmentation-fence": ["identity", "fragmentation", "fence"],
    }
    for name, conditions in specifications.items():
        config = deepcopy(base)
        config.update(
            transform="fragmentation" if conditions == ["fragmentation"] else "identity",
            transform_parameters={},
        )
        if len(conditions) > 1:
            config["train_mixture"] = [
                {
                    "name": condition,
                    "parameters": FENCES["fence-medium"] if condition == "fence" else {},
                }
                for condition in conditions
            ]
            config["expected_training_condition_counts"] = PointMixture(
                config["train_mixture"], config["training_order"], config["seed"]
            ).counts
        config["comparison_group"] = "matched-medium-visible-cohort"
        save(name, config, True)

    # All mixtures share the existing medium-fence cohort; evaluate each condition
    # on its own explicit saved eligibility list. Do not fall back behind bars.
    validation_paths = []
    val_ids = sorted(image["id"] for image in data["test"]["images"])
    for label, transform, parameters in (
        ("clean", "identity", {}),
        ("fragmentation", "fragmentation", {}),
        *((label, "fence", parameters) for label, parameters in FENCES.items()),
    ):
        config = deepcopy(base)
        config.update(transform=transform, transform_parameters=parameters)
        if transform == "fence":
            selected, ids = filter_targets(
                data["test"], val_ids, transform, parameters, base["seed"]
            )
            path, digest = manifest(output / f"validation/{label}/targets.json", selected)
            attach(config, "test", path, digest, ids)
        path = output / "validation" / label / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
        validation_paths.append(str(path.resolve()))

    smoke = deepcopy(base)
    smoke_ids = base["training_order"][:32]
    smoke_manifest = manifest(output / "data/smoke-train.json", subset(data["train"], smoke_ids))
    attach(smoke, "train", *smoke_manifest, smoke_ids)
    heldout_ids = val_ids[:8]
    heldout_manifest = manifest(output / "data/smoke-val.json", subset(data["test"], heldout_ids))
    attach(smoke, "test", *heldout_manifest, heldout_ids)
    smoke.update(
        transform="identity",
        transform_parameters={},
        accumulation_steps=16,
        train_mixture=[
            {"name": "identity"},
            {"name": "fragmentation"},
            {"name": "fence", "parameters": FENCES["fence-medium"]},
        ],
    )
    smoke_path = output / "preflight/config.json"
    smoke_path.parent.mkdir()
    smoke_path.write_text(json.dumps(smoke, indent=2) + "\n")
    plan = {
        "jobs": jobs,
        "validation_configs": validation_paths,
        "smoke_config": str(smoke_path.resolve()),
        "existing_medium_control": str(base_path.resolve()),
        "step_unit": "image_presentations",
    }
    (output / "jobs.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-images", type=int, default=5000)
    args = parser.parse_args()
    prepare(args.base_config, args.output_dir, args.pilot_images)
