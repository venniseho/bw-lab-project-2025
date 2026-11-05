"""Assemble the clean single-point SAM3 fine-tuning experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from datasets.point_training import dataset, loader
from core.infrastructure import require_slurm_step
from core.models.point_sam3 import PointSegmenter, native_parity
from core.models.sam3 import load_scene_model, VERIFIED_SAM3_API_REVISION
from core.tasks.point_segmentation import forward_sample
from core.trainer import PointTrainer
from core.training_output import TrainingOutput


def validate_config(config, max_steps):
    require_slurm_step(os.environ)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an allocated CUDA GPU")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config["epochs"] < 1 or config["workers"] < 0:
        raise ValueError("Positive epochs and nonnegative workers required")
    if config["transform"] != "identity":
        raise ValueError("Only the clean-image smoke test is configured")
    revision = subprocess.check_output(
        ["git", "-C", config["native_root"], "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["native_revision"] or revision != VERIFIED_SAM3_API_REVISION:
        raise RuntimeError(f"Unexpected native revision: {revision}")
    import sam3

    if Path(sam3.__file__).resolve().parent.parent != Path(config["native_root"]).resolve():
        raise RuntimeError(f"Python imported a different SAM3 checkout: {sam3.__file__}")
    for split, key in (("train", "train_annotations"), ("test", "heldout_annotations")):
        digest = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
        if digest != config["annotation_sha256"][split]:
            raise RuntimeError(f"Prepared {split} annotations changed")


def build_data(config):
    train = dataset(config["train_annotations"])
    heldout = dataset(config["heldout_annotations"])
    if not len(train) or not len(heldout):
        raise ValueError("Both instance splits must be nonempty")
    if {r["image_id"] for r in train.records} & {r["image_id"] for r in heldout.records}:
        raise ValueError("Training and held-out source images overlap")
    return train, heldout


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_output(path, metadata, wandb_mode):
    tracking = None
    if wandb_mode != "disabled":
        import wandb

        tracking = wandb.init(
            project="bwlab-point-finetuning",
            name=path.name,
            dir=str(path),
            config=metadata,
            mode=wandb_mode,
        )
    output = TrainingOutput(path, tracking)
    output.save_json("metadata", metadata)
    return output


def run(config, scope, output, wandb_mode="offline", max_steps=None):
    """Construct dependencies once; the trainer owns the training loop."""
    validate_config(config, max_steps)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    (path / "previews").mkdir()
    seed_everything(config["seed"])
    train, heldout = build_data(config)
    model, processor, provenance = load_scene_model(config["checkpoint"], None, torch)
    segmenter = PointSegmenter(model, processor, scope)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
    metadata = {
        "config": config,
        "scope": scope,
        "trainable_parameters": segmenter.parameter_counts,
        "provenance": provenance,
        "gpu": torch.cuda.get_device_name(),
        "precision": "bf16 forward, float32 loss/parameters",
        "max_steps": max_steps,
        "native_parity": native_parity(model, processor, heldout[0]),
    }
    output = build_output(path, metadata, wandb_mode)
    print(
        json.dumps(
            {
                "scope": scope,
                "parameters": segmenter.parameter_counts,
                "train_instances": len(train),
                "heldout_instances": len(heldout),
                "parity": metadata["native_parity"],
            }
        ),
        flush=True,
    )
    trainer = PointTrainer(segmenter, optimizer, forward_sample, output)
    try:
        return trainer.fit(
            loader(train, config["workers"]), loader(heldout, config["workers"]), config, max_steps
        )
    finally:
        output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("full", "frozen_encoder"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("offline", "online", "disabled"), default="offline"
    )
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.scope,
        args.output,
        args.wandb_mode,
        args.max_steps,
    )


if __name__ == "__main__":
    main()
