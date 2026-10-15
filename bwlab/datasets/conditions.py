"""Apply independently configured conditions to each original dataset sample."""
from collections.abc import Callable, Mapping
from copy import deepcopy

from torch.utils.data import Dataset


class ConditionDataset(Dataset):
    """Every condition starts from the original, not the previous corruption."""

    def __init__(self, dataset: Dataset, conditions: Mapping[str, Callable]):
        if not conditions or any(not name for name in conditions):
            raise ValueError("provide named stimulus conditions")
        self.dataset = dataset
        self.conditions = list(conditions.items())

    def __len__(self) -> int:
        return len(self.dataset) * len(self.conditions)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index, condition_index = divmod(index, len(self.conditions))
        name, transform = self.conditions[condition_index]
        source = self.dataset[source_index]
        sample = transform(deepcopy(source))
        sample["source_key"] = source["__key__"]
        sample["condition"] = name
        sample["__key__"] = f"{source['__key__']}/{name}"
        return sample
