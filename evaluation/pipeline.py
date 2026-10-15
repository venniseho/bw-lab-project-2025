"""Minimal orchestration contracts; quantitative protocols remain separate."""
from collections.abc import Iterable
from typing import Protocol

import numpy as np


class ModelAdapter(Protocol):
    def __call__(self, image: np.ndarray) -> object: ...


class Evaluator(Protocol):
    def update(self, sample: dict, prediction: object) -> None: ...
    def finalize(self) -> object: ...


def run_pipeline(samples: Iterable[dict], model: ModelAdapter, evaluator: Evaluator) -> object:
    """Dataset -> transforms (dataset-owned) -> model -> evaluator -> results.

    The evaluator may read targets; the model sees only RGB. This does not
    define new metrics, oracle selection, or a paired prompting protocol.
    """
    for sample in samples:
        prediction = model(sample["image"])
        evaluator.update(sample, prediction)
    return evaluator.finalize()
