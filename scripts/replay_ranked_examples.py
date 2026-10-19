"""Select measured-IoU best/worst inputs and rerun their original image batches."""
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwlab.infrastructure import require_slurm_step, file_fingerprint
from bwlab.samples import binary_mask
from bwlab.models import Sam3GridAdapter
from bwlab.models.sam3 import load_scene_model
from bwlab.models.batched_sam3 import BatchedSam3GridAdapter
from bwlab.runner import _predict_batch
from bwlab.runtime import configure_model
from bwlab.tasks.scene_proposals import SceneProposalTask
from experiments.sam3_fragmentation.experiment import build_data_module, load_config


def ranked_rows(path, condition, count, metric="full_iou"):
    if count < 1 or metric not in ("full_iou", "visible_iou"):
        raise ValueError("positive count and measured full/visible IoU required")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["condition"] == condition
                and row["status"] == "complete" and row[metric] and math.isfinite(float(row[metric]))]
    ordered = sorted(rows, key=lambda r: (float(r[metric]), int(r["input_index"])))
    return {"worst": ordered[:count], "best": list(reversed(ordered[-count:]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--metric", choices=("full_iou", "visible_iou"), default="full_iou")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    selected = ranked_rows(args.run_dir / "predictions/image_results.csv", args.condition, args.count, args.metric)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(selected, indent=2)+"\n", encoding="utf-8")
    if args.list_only:
        return
    # Saved config.json contains a provenance wrapper; load_config expects config.
    provenance = json.loads((args.run_dir / "config.json").read_text())
    from experiments.sam3_fragmentation import experiment
    if file_fingerprint(Path(experiment.__file__))["sha256"] != provenance["experiment_source"]["sha256"]:
        raise ValueError("experiment implementation changed since original run")
    saved = provenance["config"]
    if file_fingerprint(Path(saved["manifest"]))["sha256"] != provenance["input_manifest"]["sha256"]:
        raise ValueError("source manifest changed since original run")
    config_path = args.output / "replay_config.json"
    config_path.write_text(json.dumps(saved, indent=2)+"\n", encoding="utf-8")
    config = load_config(config_path)
    import os
    if config.runtime is None or not config.runtime.allow_local:
        require_slurm_step(os.environ)
    import torch
    torch.manual_seed(0)
    model, processor, info = load_scene_model(config.checkpoint, config.bpe_path, torch)
    original_model = json.loads((args.run_dir / "timing.json").read_text())["model"]
    for key in ("checkpoint", "builder_source", "predictor_source", "processor_source", "image_model_source"):
        if info[key]["sha256"] != original_model[key]["sha256"]:
            raise ValueError(f"native model provenance changed: {key}")
    if config.runtime is not None:
        configure_model(model, config.runtime)
    adapter = (BatchedSam3GridAdapter(model, processor, config.points_per_side,
                                    config.runtime.prompt_batch_size, config.runtime.pin_memory,
                                    torch.cuda.synchronize)
               if config.runtime is not None and config.runtime.batched else
               Sam3GridAdapter(model, processor, config.points_per_side, torch.cuda.synchronize))
    module = build_data_module(config)
    dataset = module.val_dataset() if config.subset == "val" else module.test_dataset()
    wanted = {int(row["input_index"]): row for group in selected.values() for row in group}
    groups = sorted({index//config.batch_size for index in wanted})
    evaluator = SceneProposalTask(model_metadata=info).get_evaluator(args.output / "examples")
    with torch.inference_mode():
        for group in groups:
            start = group*config.batch_size
            batch = [dataset[index] for index in range(start, min(start+config.batch_size, len(dataset)))]
            predictions = _predict_batch(adapter, batch)
            for offset, (sample, prediction) in enumerate(zip(batch, predictions)):
                index = start+offset
                if index in wanted:
                    if sample["__key__"] != wanted[index]["__key__"]:
                        raise ValueError("dataset changed since original run; refusing a mismatched replay")
                    for field, pixels in (("input_sha256", sample["image"]),
                                          ("target_sha256", binary_mask(sample["segmentation"], "target"))):
                        if hashlib.sha256(pixels.tobytes()).hexdigest() != wanted[index][field]:
                            raise ValueError(f"sample pixels changed since original run: {field}")
                    evaluator.update(sample, prediction)
    evaluator.finalize()


if __name__ == "__main__":
    main()
