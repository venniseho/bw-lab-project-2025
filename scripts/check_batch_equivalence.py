"""Compare optimized batches against the existing single-image SAM3 protocol."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwlab.infrastructure import require_slurm_step
from bwlab.models import Sam3GridAdapter
from bwlab.models.batched_sam3 import BatchedSam3GridAdapter
from bwlab.models.sam3 import load_scene_model
from bwlab.runtime import configure_model
from experiments.sam3_fragmentation.experiment import build_data_module, load_config


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--score-atol", type=float, default=1e-5)
    args = parser.parse_args()
    import os
    require_slurm_step(os.environ)
    config = load_config(args.config)
    module = build_data_module(config)
    dataset = module.val_dataset() if config.subset == "val" else module.test_dataset()
    if args.count < 1:
        parser.error("count must be positive")
    samples = [dataset[i] for i in range(min(args.count, len(dataset)))]
    model, processor, _ = load_scene_model(config.checkpoint, config.bpe_path, torch)
    reference = Sam3GridAdapter(model, processor, config.points_per_side, torch.cuda.synchronize)
    expected = [reference.predict(sample["image"])[0] for sample in samples]
    if config.runtime is not None:
        configure_model(model, config.runtime)
    adapter = BatchedSam3GridAdapter(model, processor, config.points_per_side,
                                    config.runtime.prompt_batch_size if config.runtime else 9,
                                    False, torch.cuda.synchronize)
    actual = adapter.predict_batch([sample["image"] for sample in samples])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mismatches = 0
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        fields = ["__key__", "proposal_id", "different_pixels", "score_difference"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, proposals, (batched, _) in zip(samples, expected, actual):
            if len(proposals) != len(batched):
                raise RuntimeError("candidate count changed")
            for old, new in zip(proposals, batched):
                for field in ("proposal_id", "grid_point_index", "candidate_index", "point_xy"):
                    if old.metadata[field] != new.metadata[field]:
                        raise RuntimeError(f"prompt/candidate correspondence changed: {field}")
                different = int(np.count_nonzero(old.mask() != new.mask()))
                delta = abs(old.metadata["predicted_iou_score"]-new.metadata["predicted_iou_score"])
                mismatches += int(different > 0 or delta > args.score_atol)
                writer.writerow({"__key__": sample["__key__"], "proposal_id": old.metadata["proposal_id"],
                                 "different_pixels": different, "score_difference": delta})
    if mismatches:
        raise RuntimeError(f"{mismatches} candidate comparisons differed; inspect {args.output} before a sweep")
    print(f"Passed {sum(map(len, expected))} comparisons; masks identical, scores within {args.score_atol}")


if __name__ == "__main__":
    main()
