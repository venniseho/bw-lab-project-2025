"""Reusable inference loop; datasets, models and task implementations are injected."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
        for batch in loader:
            for sample in batch:
                prediction = model.predict(sample["image"])
                evaluator.update(sample, prediction)
        return evaluator.finalize()
