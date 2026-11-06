"""Profile native-size point validation before choosing evaluation intervals."""

import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data_loading.point_training import loader
from core.models.point_sam3 import PointSegmenter, native_parity
from core.models.sam3 import load_scene_model
from core.point_benchmark import validation_benchmark
from experiments.sam3_point_finetuning.experiment import (
    build_data,
    build_output,
    seed_everything,
    validate_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-image", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.points_per_image) < 1:
        parser.error("Positive repeats and point counts required")
    config = json.loads(args.config.read_text())
    validate_config(config, None)
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(config["seed"])
    _, validation = build_data(config)
    model, processor, provenance = load_scene_model(config["checkpoint"], None, torch)
    segmenter = PointSegmenter(model, processor, "frozen_encoder")
    samples = loader(validation, config["workers"])
    segmenter.train(False)
    with torch.no_grad():
        sample = next(iter(samples))
        parity = native_parity(model, processor, sample)
        segmenter.predict(sample["image"], [0.0, 0.0])
        torch.cuda.synchronize()
    output = build_output(
        args.output,
        {
            "scope": "validation-benchmark",
            "config": config,
            "provenance": provenance,
            "gpu": torch.cuda.get_device_name(),
            "warmup_images": 1,
            "native_parity": parity,
            "repeats": args.repeats,
        },
        args.wandb_mode,
    )
    exit_code = 1
    try:
        reports, deltas = [], []
        for count in args.points_per_image:
            for repeat in range(args.repeats):
                order = (False, True) if repeat % 2 == 0 else (True, False)
                comparisons = []
                for shared in order:
                    report, values = validation_benchmark(
                        segmenter,
                        samples,
                        output,
                        seed=config["seed"],
                        points_per_image=count,
                        shared=shared,
                        repeat=repeat,
                    )
                    reports.append({**report, "repeat": repeat})
                    comparisons.append(values)
                delta = max(
                    abs(comparisons[0][key] - comparisons[1][key]) for key in comparisons[0]
                )
                deltas.append(
                    {"points_per_image": count, "repeat": repeat, "max_selected_iou_delta": delta}
                )
                print(
                    f"points={count}, repeat={repeat}: max selected-IoU delta={delta:.8f}",
                    flush=True,
                )
        output.save_json(
            "benchmark-results",
            {
                "reports": reports,
                "comparisons": deltas,
                "timing_scope": "Includes loader wait, transforms, model, loss/metrics and CSV; excludes initialization and warmup. No weight updates or checkpoints.",
            },
        )
        exit_code = 0
    finally:
        output.close(exit_code)


if __name__ == "__main__":
    main()
