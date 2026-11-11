"""Assemble clean or transformed single-point SAM3 fine-tuning experiments."""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from core.data_loading.point_training import dataset, loader
from core.infrastructure import require_slurm_step
from core.models.point_sam3 import PointSegmenter, native_parity
from core.models.sam3 import load_scene_model, VERIFIED_SAM3_API_REVISION
from core.tasks.point_segmentation import forward_sample
from core.trainer import PointTrainer
from core.training_output import TrainingOutput
from core.stimuli.point_transforms import PointTransform
from core.data_loading.download_ahead import DownloadReadyDataset
from core.training_resume import shutdown_signals


def validate_config(config, max_steps):
    require_slurm_step(os.environ)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an allocated CUDA GPU")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config["epochs"] < 1 or config["workers"] < 0:
        raise ValueError("Positive epochs and nonnegative workers required")
    if config.get("checkpoint_interval_steps", 1000) < 1:
        raise ValueError("Checkpoint interval must be positive")
    if config["transform"] not in ("identity", "fence", "fragmentation"):
        raise ValueError("Unknown point-prompt transform")
    if config.get("resolution", 1008) != 1008:
        raise ValueError("These experiments require native 1008x1008 preprocessing")
    if (
        config.get("validation_interval_steps") is not None
        and config["validation_interval_steps"] < 1
    ):
        raise ValueError("Validation interval must be positive")
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
    transform = PointTransform(
        config["transform"], config["seed"], config.get("transform_parameters", {})
    )
    options = {"transform": transform, "one_per_image": True, "seed": config["seed"]}
    train = dataset(config["train_annotations"], **options)
    heldout = dataset(
        config["heldout_annotations"], image_count=config.get("validation_image_count"), **options
    )
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
            project=os.environ.get("WANDB_PROJECT", "bwlab-point-finetuning"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=os.environ.get("WANDB_RUN_GROUP", path.parent.name),
            job_type=metadata["scope"],
            name=path.name,
            dir=str(path),
            config=metadata,
            mode=wandb_mode,
        )
        tracking.define_metric("train/step")
        tracking.define_metric("train/*", step_metric="train/step")
        tracking.define_metric("heldout/epoch")
        tracking.define_metric("heldout/step")
        tracking.define_metric("heldout/*", step_metric="heldout/step")
        tracking.define_metric("heldout_full/step")
        tracking.define_metric("heldout_full/*", step_metric="heldout_full/step")
        for prefix in ("examples", "examples_full"):
            tracking.define_metric(f"{prefix}/step")
            tracking.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
        for phase in ("train", "validation", "validation_full"):
            tracking.define_metric(f"timing/{phase}/step")
            tracking.define_metric(f"timing/{phase}/*", step_metric=f"timing/{phase}/step")
        if wandb_mode == "online":
            print(f"W&B dashboard: {tracking.url}", flush=True)
    output = TrainingOutput(path, tracking, image_factory=wandb.Image if tracking else None)
    output.save_json("metadata", metadata)
    return output


def run(config, scope, output, wandb_mode="online", max_steps=None, resume=None):
    """Construct dependencies once; the trainer owns the training loop."""
    validate_config(config, max_steps)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    (path / "previews").mkdir()
    seed_everything(config["seed"])
    train, heldout = build_data(config)
    if config.get("download_ahead"):
        train = DownloadReadyDataset(train)
    if resume and "training_order" not in config:
        raise ValueError(
            "Resume needs an explicit saved training order; prepare a new download-ahead run"
        )
    order = None
    if "training_order" in config:
        by_image = {record["image_id"]: index for index, record in enumerate(train.records)}
        if len(config["training_order"]) != len(train) or set(config["training_order"]) != set(
            by_image
        ):
            raise ValueError("Saved training order does not match the prepared dataset")
        order = [by_image[image_id] for image_id in config["training_order"]]
    model, processor, provenance = load_scene_model(config["checkpoint"], None, torch)
    segmenter = PointSegmenter(model, processor, scope)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
    metadata = {
        "config": config,
        "scope": scope,
        "project_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "trainable_parameters": segmenter.parameter_counts,
        "provenance": provenance,
        "gpu": torch.cuda.get_device_name(),
        "precision": "bf16 forward, float32 loss/parameters",
        "max_steps": max_steps,
        "resumed_from": str(resume) if resume else None,
        "point_policy": "uniform_foreground_fixed_per_instance",
        "dataset_counts": {
            name: {
                "unique_images": len({record["image_id"] for record in data.records}),
                "target_masks": len(data),
            }
            for name, data in (("train", train), ("validation", heldout))
        },
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
    trainer = PointTrainer(
        segmenter, optimizer, partial(forward_sample, seed=config["seed"]), output
    )
    exit_code = 1
    try:
        if resume:
            trainer.resume(resume, config)
        final_validation = None
        if config.get("validation_image_count") and config.get("full_validation_at_end", True):
            _, final_validation = build_data({**config, "validation_image_count": None})
        with shutdown_signals(trainer):
            result = trainer.fit(
                loader(train, config["workers"], shuffle=True, seed=config["seed"]),
                loader(heldout, config["workers"]),
                config,
                max_steps,
                final_validation_data=loader(final_validation, config["workers"])
                if final_validation is not None
                else None,
                train_loader_factory=(
                    lambda epoch, cursor: loader(
                        train,
                        config["workers"],
                        order=order,
                        start=cursor,
                        seed=config["seed"] + epoch,
                    )
                )
                if order is not None
                else None,
            )
        exit_code = 0
        return result
    finally:
        output.close(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("full", "frozen_encoder"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.scope,
        args.output,
        args.wandb_mode,
        args.max_steps,
        args.resume,
    )


if __name__ == "__main__":
    main()
