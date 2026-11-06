"""Point-mask training orchestration with injected model, task and outputs."""

import time
import shutil
from typing import Protocol

import numpy as np
import torch

from core.training_timing import PhaseTiming
from core.point_records import sample_record, mask_rle


class TrainableSegmenter(Protocol):
    model: torch.nn.Module
    scope: str

    def predict(self, image, point): ...
    def encode(self, image): ...
    def predict_encoded(self, maps, point, original_shape): ...
    def train(self, training=True): ...
    def gradient_probes(self): ...
    def check_update(self, probes): ...


def checkpoint(model, optimizer, output, epoch, config, scope, step=0):
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
            "step": step,
        },
        temporary,
    )
    temporary.replace(path)
    # Read back metadata, rather than treating an existing filename as success.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "bwlab_point_finetune_v1" or saved["epoch"] != epoch:
        raise RuntimeError("Saved checkpoint is invalid")
    return {"path": str(path), "bytes": path.stat().st_size, "epoch": epoch, "step": step}


class PointTrainer:
    """Run optimization and held-out evaluation; construct no datasets or models."""

    def __init__(self, segmenter: TrainableSegmenter, optimizer, task, output):
        self.segmenter, self.optimizer = segmenter, optimizer
        self.task, self.output = task, output
        self.parameters = [p for p in segmenter.model.parameters() if p.requires_grad]
        self.steps, self.audit = 0, {}
        self.best_iou = -float("inf")
        self.last_summary = None
        self.last_validation_step = None
        self.latest_checkpoint = None

    def train_step(self, sample, epoch):
        start = time.perf_counter()
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
                **sample_record(sample),
                "gradient_norm": float(norm),
                "compute_step_seconds": time.perf_counter() - start,
                **metrics,
            },
        )

    def train_epoch(
        self, samples, epoch, max_steps=None, after_step=None, validation_interval=None
    ):
        self.segmenter.train(True)
        timing = PhaseTiming(self.segmenter.model, "train", epoch)
        for sample in samples:
            self.train_step(sample, epoch)
            timing.observe(sample)
            if after_step is not None and self.steps % validation_interval == 0:
                timing.exclude(after_step)
            if max_steps is not None and self.steps >= max_steps:
                break
        report = timing.finish()
        report["step"] = self.steps
        self.output.record("timing", report)

    @torch.no_grad()
    def evaluate(self, samples, epoch, baseline=None, validation_set="interval"):
        self.segmenter.train(False)
        timing = PhaseTiming(self.segmenter.model, "validation", epoch)
        rows, examples = [], {}
        cached_image, cached_features, encoder_forwards = None, None, 0
        for index, sample in enumerate(samples):
            if cached_image is None or not np.array_equal(cached_image, sample["image"]):
                cached_features = self.segmenter.encode(sample["image"])
                cached_image = sample["image"]
                encoder_forwards += 1
            _, metrics, logits, scores, point = self.task(
                self.segmenter, sample, encoded=cached_features
            )
            rows.append(metrics)
            timing.observe(sample)
            self.output.record(
                "heldout",
                {
                    "phase": "heldout",
                    "epoch": epoch,
                    "step": self.steps,
                    "validation_set": validation_set,
                    **sample_record(sample),
                    **metrics,
                    "selected_mask_rle": mask_rle((logits[scores.argmax()] > 0).cpu().numpy()),
                },
            )
            if index < 4:
                prediction = {
                    "mask": (logits[scores.argmax()] > 0).cpu().numpy(),
                    "iou": metrics["selected_iou"],
                }
                examples[sample["__key__"]] = prediction
                if baseline is not None and sample["__key__"] in baseline:
                    self.output.preview(
                        sample,
                        point,
                        baseline[sample["__key__"]],
                        prediction,
                        epoch,
                        index,
                        step=self.steps,
                        validation_set=validation_set,
                    )
        report = timing.finish()
        report.update(
            step=self.steps, encoder_forwards=encoder_forwards, validation_set=validation_set
        )
        self.output.record("timing", report)
        summary = {
            "phase": "heldout_summary",
            "epoch": epoch,
            "step": self.steps,
            "validation_set": validation_set,
            "instances": len(rows),
            "mean_iou": float(np.mean([row["selected_iou"] for row in rows])),
            "mean_loss": float(np.mean([row["loss"] for row in rows])),
        }
        self.output.record("summary", summary)
        self.last_summary = summary
        self.last_validation_step = self.steps
        return examples

    def save_checkpoint(self, config, epoch):
        start = time.perf_counter()
        saved = checkpoint(
            self.segmenter.model,
            self.optimizer,
            self.output.path,
            epoch,
            config,
            self.segmenter.scope,
            self.steps,
        )
        if (
            self.last_summary["validation_set"] == "interval"
            and self.last_summary["mean_iou"] > self.best_iou
        ):
            self.best_iou = self.last_summary["mean_iou"]
            destination = self.output.path / "checkpoint-best.part"
            shutil.copyfile(saved["path"], destination)
            destination.replace(self.output.path / "checkpoint-best.pt")
        self.output.record(
            "checkpoints",
            {
                **saved,
                "wall_seconds": time.perf_counter() - start,
                "validation_mean_iou": self.last_summary["mean_iou"],
            },
        )
        self.latest_checkpoint = saved
        return saved

    def fit(self, train_data, heldout_data, config, max_steps=None, final_validation_data=None):
        start = time.perf_counter()
        baseline = self.evaluate(heldout_data, 0)
        self.save_checkpoint(config, 0)
        full_baseline = None
        if final_validation_data is not None:
            full_baseline = self.evaluate(final_validation_data, 0, validation_set="full")
        interval = config.get("validation_interval_steps")
        for epoch in range(1, config["epochs"] + 1):

            def periodic_validation():
                if interval and self.steps % interval == 0:
                    self.evaluate(heldout_data, epoch, baseline)
                    self.save_checkpoint(config, epoch)
                    self.segmenter.train(True)

            self.train_epoch(
                train_data, epoch, max_steps, periodic_validation if interval else None, interval
            )
            if self.last_validation_step != self.steps:
                self.evaluate(heldout_data, epoch, baseline)
            saved = self.latest_checkpoint
            if saved is None or saved["step"] != self.steps:
                saved = self.save_checkpoint(config, epoch)
            if max_steps is not None and self.steps >= max_steps:
                break
        if final_validation_data is not None:
            self.evaluate(
                final_validation_data, epoch, baseline=full_baseline, validation_set="full"
            )
        summary = {
            "steps": self.steps,
            "checkpoint": saved,
            "best_checkpoint": str(self.output.path / "checkpoint-best.pt"),
            "best_interval_validation_iou": self.best_iou,
            "final_validation_summary": self.last_summary,
            "gradient_and_step_audit": self.audit,
            "wall_seconds": time.perf_counter() - start,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        self.output.save_json("complete", summary)
        print(summary, flush=True)
        return summary
