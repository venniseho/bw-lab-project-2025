"""Explicit dataset splits and train/evaluation data loading."""
from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import DataLoader, Dataset

from ._loading import build_loader


class DataModule:
    """Own already constructed splits, without downloads or implicit splitting.

    Like mlvbench's DataModule, the experiment asks for train/val/test loaders;
    the runner does not know where files or annotations came from.
    """

    def __init__(self, datasets: Mapping[str, Dataset], *, seed: int = 0):
        if not datasets or not set(datasets).issubset({"train", "val", "test"}):
            raise ValueError("provide at least one explicit train/val/test dataset")
        self._datasets = dict(datasets)
        self.seed = seed

    def subsets(self) -> list[str]:
        return list(self._datasets)

    def _dataset(self, subset: str) -> Dataset:
        if subset not in self._datasets:
            raise ValueError(f"split {subset!r} was not supplied")
        return self._datasets[subset]

    def train_dataset(self) -> Dataset:
        return self._dataset("train")

    def val_dataset(self) -> Dataset:
        return self._dataset("val")

    def test_dataset(self, subset: str = "test") -> Dataset:
        if subset != "test":
            raise ValueError("test_dataset expects the test split")
        return self._dataset(subset)

    def train_dataloader(self, **kwargs) -> DataLoader:
        return build_loader(self.train_dataset(), split="train", seed=self.seed, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        return build_loader(self.val_dataset(), split="val", seed=self.seed, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return build_loader(self.test_dataset(), split="test", seed=self.seed, **kwargs)
