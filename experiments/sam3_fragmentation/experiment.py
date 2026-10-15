"""Wire a configured dataset, stimulus arm, native model and proposal-export task.

Like the pinned mlvbench experiment, Config/run live outside the reusable library.
GPU execution is explicit, requires an existing SLURM step and submits no jobs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
import time
import platform
from pathlib import Path
from typing import Literal

from bwlab.datasets import DataModule, ManifestDataModule
from bwlab.datasets.conditions import ConditionDataset
from bwlab.infrastructure import file_fingerprint, require_slurm_step
from bwlab.models import Sam3GridAdapter
from bwlab.models.sam3 import load_scene_model
from bwlab.runner import Runner
from bwlab.stimuli import ContourFragmentation, FenceOcclusion, FragmentationConfig
from bwlab.stimuli.export import prepare_stimuli
from bwlab.tasks.scene_proposals import SceneProposalTask


@dataclass(frozen=True)
class Config:
    manifest: Path
    condition: Literal["fence", "fragmentation"]
    subset: Literal["val", "test"] = "test"
    seed: int | None = 42
    fence_widths: tuple[float, ...] = (0, 4, 12, 24)
    fence_spacing: float = 40
    fence_angle: float = 90
    fence_phase: float | None = None
    fragmentation: FragmentationConfig = field(default_factory=FragmentationConfig)
    checkpoint: Path | None = None
    bpe_path: Path | None = None
    points_per_side: int = 3
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    min_score: float | None = None
    dedup_iou: float | None = None
    proposal_limit: int = 100
    save_candidate_limit: int | None = None


def build_conditions(config: Config) -> dict:
    """Construct each experimental condition independently."""
    if config.subset not in ("val", "test"):
        raise ValueError("choose val or test explicitly")
    if config.condition == "fence":
        widths = config.fence_widths
        if (len(widths) != 4 or widths[0] != 0
                or not all(a < b for a, b in zip(widths, widths[1:]))
                or widths[-1] > config.fence_spacing):
            raise ValueError("four fence widths must start at zero, increase, and not exceed spacing")
        if config.fence_phase is not None and config.seed is not None:
            raise ValueError("set seed to null when using an explicit fence phase")
        conditions = {name: FenceOcclusion(width, config.fence_spacing, config.fence_angle,
                                            config.seed, config.fence_phase)
                      for name, width in zip(("clean", "light", "medium", "heavy"), widths)}
    elif config.condition == "fragmentation":
        conditions = {"fragmentation": ContourFragmentation(config.fragmentation, config.seed)}
    else:
        raise ValueError("condition must be fence or fragmentation")
    return conditions


def build_data_module(config: Config) -> DataModule:
    """Wire explicit source data to independently applied stimulus transforms."""
    conditions = build_conditions(config)
    sources = ManifestDataModule({config.subset: config.manifest}, seed=config.seed or 0)
    original = sources.val_dataset() if config.subset == "val" else sources.test_dataset()
    return DataModule({config.subset: ConditionDataset(original, conditions)}, seed=config.seed or 0)


def _save_config(config: Config, output_path: Path) -> None:
    metadata = {"config": asdict(config), "input_manifest": file_fingerprint(config.manifest),
                "experiment_source": file_fingerprint(Path(__file__)),
                "architecture_reference": "c204a8185fb662b3b1edacfb7a98f7b36fbd9a70"}
    (output_path / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def prepare(config: Config, output_path: Path) -> dict:
    """CPU path: export stimuli and targets; never load a model."""
    data_module = build_data_module(config)
    dataset = data_module.val_dataset() if config.subset == "val" else data_module.test_dataset()
    report = prepare_stimuli(dataset, output_path)
    _save_config(config, output_path)
    return report


def run(config: Config, output_path: Path) -> dict:
    """Existing automatic grid protocol on configured stimuli, inside SLURM only.

    This archives proposals and targets; it does not extend quantitative metrics
    or claim that an oracle-selected proposal is an autonomous result.
    """
    require_slurm_step(os.environ)
    run_start = time.perf_counter()
    if output_path.exists():
        raise FileExistsError(output_path)
    if config.checkpoint is None or not config.checkpoint.is_file():
        raise ValueError("run needs an existing native SAM3 checkpoint")
    if config.points_per_side < 1:
        raise ValueError("points_per_side must be positive")
    setup_start = time.perf_counter()
    data_module = build_data_module(config)
    dataset_setup_seconds = time.perf_counter()-setup_start
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for native SAM3; use prepare for CPU export")
    torch.manual_seed(0)
    load_start = time.perf_counter()
    model, processor, model_info = load_scene_model(config.checkpoint, config.bpe_path, torch)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter()-load_start
    task = SceneProposalTask(min_score=config.min_score, dedup_iou=config.dedup_iou,
                             proposal_limit=config.proposal_limit,
                             save_candidate_limit=config.save_candidate_limit,
                             model_metadata=model_info)
    runner = Runner(data_module, task, batch_size=config.batch_size,
                    num_workers=config.num_workers, prefetch_factor=config.prefetch_factor)
    output_path.mkdir(parents=True, exist_ok=False)
    _save_config(config, output_path)
    pipeline_start = time.perf_counter()
    with torch.inference_mode():
        report = runner.evaluate(Sam3GridAdapter(model, processor, config.points_per_side,
                                                 synchronize=torch.cuda.synchronize),
                                  output_path / "predictions", subset=config.subset)
    timing = {"dataset_setup_seconds": dataset_setup_seconds,
              "model_load_seconds": model_load_seconds,
              "pipeline_seconds": time.perf_counter()-pipeline_start,
              "run_total_seconds": time.perf_counter()-run_start,
              "input_count": len(report["records"]), "cuda_synchronized": True,
              "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "python_version": platform.python_version(),
              "timing_scope": "wall seconds; run excludes process imports and scheduler queue; pipeline includes loading/transforms/inference/export"}
    (output_path / "timing.json").write_text(json.dumps(timing, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(timing))
    return report


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for name in ("manifest", "checkpoint", "bpe_path"):
        if raw.get(name) is not None:
            raw[name] = (path.resolve().parent / raw[name]).resolve()
    if "fragmentation" in raw:
        raw["fragmentation"] = FragmentationConfig(**raw["fragmentation"])
    if "fence_widths" in raw:
        raw["fence_widths"] = tuple(raw["fence_widths"])
    return Config(**raw)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = prepare(config, args.output) if args.command == "prepare" else run(config, args.output)
    print(f"{args.command}: {report['status']} -> {args.output}")


if __name__ == "__main__":
    main()
