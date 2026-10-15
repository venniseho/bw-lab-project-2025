"""DataLoader construction for variable native image sizes."""
from torch.utils.data import DataLoader, Dataset


def collate_samples(samples: list[dict]) -> list[dict]:
    """Keep variable native resolutions and optional fields; no tensor stacking."""
    return samples


def build_loader(dataset: Dataset, *, split: str, batch_size: int = 1,
                 num_workers: int = 0, prefetch_factor: int = 2,
                 seed: int = 0) -> DataLoader:
    """Shuffle training only. Callers explicitly provide disjoint split datasets."""
    import torch
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    options = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=num_workers, collate_fn=collate_samples,
                      generator=torch.Generator().manual_seed(seed), **options)
