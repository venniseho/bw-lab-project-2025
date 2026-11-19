"""Evaluate a completed point checkpoint on other prepared validation conditions."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data_loading.point_training import loader
from core.infrastructure import file_fingerprint
from core.models.point_sam3 import PointSegmenter
from core.models.sam3 import load_scene_model
from core.tasks.point_segmentation import forward_sample
from core.trainer import PointTrainer
from core.training_output import TrainingOutput
from experiments.sam3_point_finetuning.experiment import (
    build_data,
    build_output,
    seed_everything,
    validate_config,
)


def completed_source(path):
    """Refuse partial runs and ambiguous checkpoint selection."""
    metadata = json.loads((path / "metadata.json").read_text())
    complete = json.loads((path / "complete.json").read_text())
    checkpoint = path / "checkpoint.pt"
    if not checkpoint.is_file() or complete["checkpoint"]["step"] != complete["steps"]:
        raise ValueError("Source must have a final checkpoint from a completed run")
    return metadata, complete, checkpoint


def restore_final(model, path, metadata, complete):
    """Load our full state strictly; never treat it as Meta's prefixed checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("format") != "bwlab_point_finetune_v1"
        or saved.get("step") != complete["steps"]
        or saved.get("scope") != metadata["scope"]
        or saved.get("config") != metadata["config"]
    ):
        raise ValueError("Final checkpoint does not match completed source metadata")
    model.load_state_dict(saved["model"], strict=True)


def target_configs(paths, source):
    """Use each target's saved cohort, not the training condition's exclusions."""
    targets = {}
    for path in paths:
        name = path.parent.name
        if name in targets:
            raise ValueError(f"Duplicate target name: {name}")
        config = json.loads(path.read_text())
        validate_config(config, None)
        for key in ("seed", "native_root", "native_revision", "checkpoint"):
            if config[key] != source[key]:
                raise ValueError(f"Source and target differ in {key}")
        if config.get("validation_image_count") is not None:
            raise ValueError("Transfer evaluation requires the full prepared validation cohort")
        targets[name] = config
    if not targets:
        raise ValueError("At least one target condition is required")
    return targets


def log_evaluation(output, name, summary, phase):
    if output.tracking:
        output.tracking.log(
            {
                "cross/step": summary["step"],
                **{
                    f"cross/{name}/{key}": value
                    for key, value in summary.items()
                    if isinstance(value, (int, float))
                },
            }
        )
        output.tracking.summary.update(
            {f"{phase}/{name}/{key}": value for key, value in summary.items()}
        )


@torch.no_grad()
def run(source_run, target_paths, destination, wandb_mode):
    started = time.perf_counter()
    metadata, complete, checkpoint = completed_source(source_run)
    source = metadata["config"]
    validate_config(source, None)
    targets = target_configs(target_paths, source)
    destination.mkdir(parents=True, exist_ok=False)
    seed_everything(source["seed"])
    model, processor, provenance = load_scene_model(source["checkpoint"], None, torch)
    segmenter = PointSegmenter(model, processor, "mask_decoder_only")
    model.requires_grad_(False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    output = build_output(
        destination,
        {
            "scope": "cross-condition-evaluation",
            "source_run": str(source_run.resolve()),
            "source_config": source,
            "source_scope": metadata["scope"],
            "source_steps": complete["steps"],
            "checkpoint_selection": "final (not validation-selected best)",
            "checkpoint": file_fingerprint(checkpoint),
            "targets": targets,
            "provenance": provenance,
            "gpu": torch.cuda.get_device_name(),
            "project_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "point_policy": "uniform_foreground_fixed_per_instance",
            "training_updates": 0,
        },
        wandb_mode,
    )
    if output.tracking:
        output.tracking.define_metric("cross/step")
        output.tracking.define_metric("cross/*", step_metric="cross/step")
    baselines, summaries = {}, {}
    exit_code = 1
    try:
        # Finish every baseline before loading fine-tuned weights once.
        for phase in ("pretrained", "finetuned"):
            if phase == "finetuned":
                restore_final(model, checkpoint, metadata, complete)
            for name, config in targets.items():
                path = destination / name
                if phase == "pretrained":
                    (path / "previews").mkdir(parents=True)
                _, validation = build_data(config)
                if len(validation) != config["splits"]["test"]["instances"]:
                    raise ValueError(f"Validation count changed: {name}")
                # No optimizer is constructed. Reuse the existing held-out evaluator.
                trainer = PointTrainer(
                    segmenter,
                    None,
                    partial(forward_sample, seed=config["seed"], loss_region="full"),
                    TrainingOutput(path),
                )
                trainer.steps = 0 if phase == "pretrained" else complete["steps"]
                print(f"{phase}: {source_run.name} -> {name}, n={len(validation)}", flush=True)
                examples = trainer.evaluate(
                    loader(validation, config["workers"]),
                    0 if phase == "pretrained" else complete["checkpoint"]["epoch"],
                    baseline=baselines.get(name),
                    validation_set="full",
                )
                if phase == "pretrained":
                    baselines[name] = examples
                summaries.setdefault(name, {})[phase] = trainer.last_summary
                log_evaluation(output, name, trainer.last_summary, phase)
        if output.tracking:
            for name in targets:
                for index, image in enumerate(
                    sorted((destination / name / "previews").glob("*.png"))
                ):
                    output.tracking.log(
                        {f"cross_examples/{name}/{index}": output.image_factory(str(image))}
                    )
        output.save_json(
            "complete",
            {
                "steps": complete["steps"],
                "training_updates": 0,
                "wall_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "results": summaries,
            },
        )
        exit_code = 0
    finally:
        output.close(exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    args = parser.parse_args()
    run(args.source_run, args.target_config, args.output, args.wandb_mode)


if __name__ == "__main__":
    main()
