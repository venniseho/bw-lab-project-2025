"""Reusable inference loop; datasets, models and task implementations are injected."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import time
import torch

from bwlab.datasets._base import DataModule
from bwlab.models._base import Model
from bwlab.tasks._base import Evaluator, Task


def run_pipeline(samples: Iterable[dict], model, evaluator: Evaluator) -> object:
    """Compatibility loop for an iterable of individual dictionary samples."""
    for sample in samples:
        prediction = model(sample["image"])
        evaluator.update(sample, prediction)
    return evaluator.finalize()


@dataclass
class Runner:
    """The inference analogue of mlvbench's trainer/evaluate responsibility."""

    data_module: DataModule
    task: Task
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    warmup_batches: int = 0

    @torch.inference_mode()
    def evaluate(self, model: Model, output_path: Path, *, subset: str = "test") -> object:
        options = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                       prefetch_factor=self.prefetch_factor)
        if subset == "test":
            loader = self.data_module.test_dataloader(**options)
        elif subset == "val":
            loader = self.data_module.val_dataloader(**options)
        else:
            raise ValueError("evaluation requires an explicit val or test split")
        evaluator = self.task.get_evaluator(output_path)
        self.warmup_seconds = 0.0
        self.evaluation_seconds = 0.0
        warmup_done = 0
        evaluation_start = time.perf_counter()
        try:
            for batch_number, (batch, wait_seconds) in enumerate(_timed_batches(loader)):
                if warmup_done < self.warmup_batches:
                    start = time.perf_counter()
                    for _ in range(self.warmup_batches-warmup_done):
                        _predict_batch(model, batch)
                        warmup_done += 1
                    self.warmup_seconds = time.perf_counter()-start
                    evaluation_start += self.warmup_seconds
                start = time.perf_counter()
                predictions = _predict_batch(model, batch)
                inference_seconds = time.perf_counter()-start
                if len(predictions) != len(batch):
                    raise ValueError("prediction count differs from image batch")
                for index, sample in enumerate(batch):
                    sample["batch_index"] = batch_number
                    sample.setdefault("timing", {})["batch_wait_seconds"] = wait_seconds if index == 0 else 0.0
                    sample["timing"]["inference_wall_seconds"] = inference_seconds/len(batch)
                    evaluator.update(sample, predictions[index])
                del predictions
        except BaseException:
            if callable(getattr(evaluator, "close", None)):
                evaluator.close()
            raise
        self.evaluation_seconds = time.perf_counter()-evaluation_start
        return evaluator.finalize()


def _predict_batch(model, batch):
    if callable(getattr(model, "predict_batch", None)):
        return model.predict_batch([sample["image"] for sample in batch])
    return [model.predict(sample["image"]) for sample in batch]


def _timed_batches(loader):
    """Include iterator startup; charge each batch wait to its first sample only."""
    start = time.perf_counter()
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter()-start
        start = time.perf_counter()
