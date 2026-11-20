"""Reuse the half-data fragmentation targets/order for an untransformed control."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(base, output):
    config = json.loads(base.read_text())
    if (
        config["transform"] != "fragmentation"
        or config.get("loss_region", "full") != "full"
        or config.get("lr_schedule", {}).get("name", "constant") != "constant"
        or config.get("validation_image_count") is not None
        or config.get("requested_training_fraction") != 0.5
    ):
        raise ValueError("Use the fixed-rate, full-validation, half-data fragmentation config")
    for split, key in (("train", "train_annotations"), ("test", "heldout_annotations")):
        if (
            hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
            != config["annotation_sha256"][split]
        ):
            raise ValueError(f"Prepared {split} annotations changed")
    config.update(
        transform="identity",
        transform_parameters={},
        loss_region="full",
        checkpoint_metric="mean_full_iou",
        comparison_control=str(base.resolve()),
        condition_label="clean",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared clean config: {output / 'config.json'}")
    print(
        f"Train images/updates: {config['splits']['train']['instances']}; validation targets: {config['splits']['test']['instances']}"
    )
    print("Reuses saved target manifests and training order; no downloads or training launched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base_config, args.output_dir)
