"""Point-mask training orchestration with injected model, task and outputs."""

import time
from typing import Protocol

import numpy as np
import torch


class TrainableSegmenter(Protocol):
    model: torch.nn.Module
    scope: str

    def predict(self, image, point): ...
    def train(self, training=True): ...
    def gradient_probes(self): ...
    def check_update(self, probes): ...


def checkpoint(model, optimizer, output, epoch, config, scope):
    path = output / "checkpoint.pt"
    temporary = output / "checkpoint.part"
    torch.save(
        {
            "format": "bwlab_point_finetune_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "scope": scope,
        },
        temporary,
    )
    temporary.replace(path)
    # Read back metadata, rather than treating an existing filename as success.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "bwlab_point_finetune_v1" or saved["epoch"] != epoch:
        raise RuntimeError("Saved checkpoint is invalid")
    return {"path": str(path), "bytes": path.stat().st_size, "epoch": epoch}


class PointTrainer:
    """Run optimization and held-out evaluation; construct no datasets or models."""

    def __init__(self, segmenter: TrainableSegmenter, optimizer, task, output):
        self.segmenter, self.optimizer = segmenter, optimizer
        self.task, self.output = task, output
        self.parameters = [p for p in segmenter.model.parameters() if p.requires_grad]
        self.steps, self.audit = 0, {}

    def train_step(self, sample, epoch):
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics, _, _, _ = self.task(self.segmenter, sample)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        probes = self.segmenter.gradient_probes() if self.steps == 0 else None
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        if probes is not None:
            self.audit = self.segmenter.check_update(probes)
            self.output.save_json("audit", self.audit)
        self.steps += 1
        self.output.record(
            "train",
            {
                "phase": "train",
                "epoch": epoch,
                "step": self.steps,
                "key": sample["__key__"],
                "gradient_norm": float(norm),
                **metrics,
            },
        )

    def train_epoch(self, samples, epoch, max_steps=None):
        self.segmenter.train(True)
        for sample in samples:
            self.train_step(sample, epoch)
            if max_steps is not None and self.steps >= max_steps:
                break

    @torch.no_grad()
    def evaluate(self, samples, epoch, baseline=None):
        self.segmenter.train(False)
        rows, examples = [], {}
        for index, sample in enumerate(samples):
            _, metrics, logits, scores, point = self.task(self.segmenter, sample)
            rows.append(metrics)
            self.output.record(
                "heldout", {"phase": "heldout", "epoch": epoch, "key": sample["__key__"], **metrics}
            )
            if index < 4:
                prediction = {
                    "mask": (logits[scores.argmax()] > 0).cpu().numpy(),
                    "iou": metrics["selected_iou"],
                }
                examples[sample["__key__"]] = prediction
                if baseline is not None:
                    self.output.preview(
                        sample, point, baseline[sample["__key__"]], prediction, epoch, index
                    )
        summary = {
            "phase": "heldout_summary",
            "epoch": epoch,
            "instances": len(rows),
            "mean_iou": float(np.mean([row["selected_iou"] for row in rows])),
            "mean_loss": float(np.mean([row["loss"] for row in rows])),
        }
        self.output.record("summary", summary)
        return examples

    def fit(self, train_data, heldout_data, config, max_steps=None):
        start = time.perf_counter()
        baseline = self.evaluate(heldout_data, 0)
        for epoch in range(1, config["epochs"] + 1):
            self.train_epoch(train_data, epoch, max_steps)
            self.evaluate(heldout_data, epoch, baseline)
            saved = checkpoint(
                self.segmenter.model,
                self.optimizer,
                self.output.path,
                epoch,
                config,
                self.segmenter.scope,
            )
            if max_steps is not None and self.steps >= max_steps:
                break
        summary = {
            "steps": self.steps,
            "checkpoint": saved,
            "gradient_and_step_audit": self.audit,
            "wall_seconds": time.perf_counter() - start,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        self.output.save_json("complete", summary)
        print(summary, flush=True)
        return summary
