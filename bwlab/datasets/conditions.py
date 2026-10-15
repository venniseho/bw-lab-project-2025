"""Apply independently configured conditions to each original dataset sample."""
from collections.abc import Callable, Mapping
from copy import deepcopy
import time

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
        start = time.perf_counter()
        source = self.dataset[source_index]
        load_seconds = time.perf_counter()-start
        start = time.perf_counter()
        transform_input = deepcopy(source)
        copy_seconds = time.perf_counter()-start
        start = time.perf_counter()
        sample = transform(transform_input)
        sample["timing"] = {"source_load_seconds": load_seconds,
                            "input_copy_seconds": copy_seconds,
                            "transform_seconds": time.perf_counter()-start}
        sample["source_key"] = source["__key__"]
        sample["condition"] = name
        sample["__key__"] = f"{source['__key__']}/{name}"
        return sample
