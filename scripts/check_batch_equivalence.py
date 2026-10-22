"""Compare native single-image SAM3 with optimized paths and isolate numerical drift."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwlab.infrastructure import require_slurm_step
from bwlab.models import Sam3GridAdapter
from bwlab.models.batched_sam3 import BatchedSam3GridAdapter, _tensor_image
from bwlab.models.sam3 import load_scene_model
from bwlab.runtime import configure_model, RuntimeConfig
from experiments.sam3_fragmentation.experiment import build_data_module, load_config
from scripts._equivalence_metrics import comparison_rows, summarize


FIELDS = ("stage", "__key__", "proposal_id", "different_pixels", "different_fraction",
          "reference_area", "actual_area", "mask_iou", "score_difference",
          "reference_target_iou", "actual_target_iou", "target_iou_difference", "strict_mismatch")


def preprocessing_diagnostics(processor, samples):
    rows = []
    for sample in samples:
        rgb = sample["image"]
        native = processor.transform(v2.functional.to_image(Image.fromarray(rgb)).to(processor.device))
        tensor = _tensor_image(processor, rgb)
        rows.append({"__key__": sample["__key__"], "exact": torch.equal(native, tensor),
                     "max_absolute_difference": float((native-tensor).abs().max().item())})
    return rows


def prediction_stages(model, processor, reference, images, points_per_side, runtime, diagnose):
    if diagnose:
        yield "reference_repeat", [reference.predict(image)[0] for image in images]
        cases = (("tensor_single", 1, True), ("image_batch_only", 1, False),
                 ("prompt_batch_only", runtime.prompt_batch_size, True),
                 ("combined_without_benchmark", runtime.prompt_batch_size, False))
        for name, prompt_count, single in cases:
            adapter = BatchedSam3GridAdapter(model, processor, points_per_side, prompt_count,
                                            False, torch.cuda.synchronize)
            predictions = ([adapter.predict_batch([image])[0] for image in images] if single else
                           adapter.predict_batch(images))
            yield name, [prediction[0] for prediction in predictions]
    configure_model(model, runtime)
    adapter = BatchedSam3GridAdapter(model, processor, points_per_side,
                                    runtime.prompt_batch_size, runtime.pin_memory, torch.cuda.synchronize)
    yield "configured_runtime", [prediction[0] for prediction in adapter.predict_batch(images)]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--score-atol", type=float, default=1e-5)
    parser.add_argument("--strict", action="store_true",
                        help="Opt in to failing on any mask pixel or score-tolerance difference")
    parser.add_argument("--diagnose", action="store_true", help="Isolate each optimization without stopping at the first mismatch")
    args = parser.parse_args()
    import os
    require_slurm_step(os.environ)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU allocated. Request --gpus=1 with srun or use scripts/vulcan_throughput.sbatch")
    if args.count < 1 or not np.isfinite(args.score_atol) or args.score_atol < 0:
        parser.error("positive count and finite nonnegative score-atol required")
    config = load_config(args.config)
    module = build_data_module(config)
    dataset = module.val_dataset() if config.subset == "val" else module.test_dataset()
    samples = [dataset[i] for i in range(min(args.count, len(dataset)))]
    if not samples:
        raise ValueError("no inputs to compare")
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError("use a fresh diagnostic output path")
    model, processor, model_info = load_scene_model(config.checkpoint, config.bpe_path, torch)
    runtime = config.runtime or RuntimeConfig(cudnn_benchmark=False)
    previous_benchmark = torch.backends.cudnn.benchmark
    if args.diagnose:
        torch.backends.cudnn.benchmark = False
    try:
        reference = Sam3GridAdapter(model, processor, config.points_per_side, torch.cuda.synchronize)
        expected = [reference.predict(sample["image"])[0] for sample in samples]
        evidence = {"torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                    "runtime": asdict(runtime), "model": model_info,
                    "input_count": len(samples), "score_atol": args.score_atol,
                    "comparison_policy": "strict" if args.strict else "report_only",
                    "baseline_cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "preprocessing": preprocessing_diagnostics(processor, samples) if args.diagnose else [],
                    "stages": {}}
        if evidence["preprocessing"]:
            print(f"preprocessing: {sum(row['exact'] for row in evidence['preprocessing'])}/{len(samples)} exact; "
                  f"max absolute difference={max(row['max_absolute_difference'] for row in evidence['preprocessing'])}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            stages = prediction_stages(model, processor, reference, [s["image"] for s in samples],
                                       config.points_per_side, runtime, args.diagnose)
            for stage, actual in stages:
                if len(actual) != len(samples):
                    raise RuntimeError("image count changed")
                rows = [row for sample, old, new in zip(samples, expected, actual)
                        for row in comparison_rows(sample["__key__"], old, new, stage,
                                                   args.score_atol, sample["segmentation"])]
                writer.writerows(rows)
                stream.flush()
                summary = summarize(rows, expected, actual)
                evidence["stages"][stage] = summary
                args.output.with_suffix(".summary.json").write_text(
                    json.dumps(evidence, indent=2, allow_nan=False)+"\n", encoding="utf-8")
                print(f"{stage}: {summary['strict_mismatch_count']}/{summary['candidate_count']} strict differences; "
                      f"min mask IoU={summary['min_mask_iou']:.6f}; "
                      f"max score delta={summary['max_score_difference']:.6f}; "
                      f"max target IoU delta={summary['max_target_iou_difference']}; "
                      f"selected candidate changes={summary['selected_candidate_changes']}", flush=True)
        mismatches = sum(stage["strict_mismatch_count"] for stage in evidence["stages"].values())
        if mismatches and args.strict:
            raise RuntimeError(f"{mismatches} strict differences across stages; inspect {args.output} "
                               f"and {args.output.with_suffix('.summary.json')} before a sweep")
        if mismatches:
            print(f"Comparison complete: {mismatches} strict differences recorded for review; "
                  "numerical differences do not fail this report-only check. "
                  "This does not establish equivalent accuracy.")
        else:
            print("All tested masks identical; scores within configured tolerance")
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark


if __name__ == "__main__":
    main()
