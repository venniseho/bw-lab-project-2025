"""Tasks own result handling; evaluators accumulate and finalize a run."""
from pathlib import Path
from typing import Protocol


class Evaluator(Protocol):
    def update(self, sample: dict, prediction: object) -> None: ...
    def finalize(self) -> object: ...


class Task(Protocol):
    def get_evaluator(self, output_path: Path) -> Evaluator: ...
