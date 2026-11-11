"""Wait for an external downloader without doing network work on GPU workers."""

from pathlib import Path
import random
import time

from torch.utils.data import Dataset


def cached_first_order(images, seed):
    """Snapshot availability once; save this order rather than rebuilding on resume."""
    ready, pending = [], []
    for image in sorted(images, key=lambda item: item["id"]):
        (ready if Path(image["file_name"]).is_file() else pending).append(image["id"])
    random.Random(seed).shuffle(ready)
    # Match the existing downloader's ascending-ID traversal for missing files.
    return ready + pending, len(ready)


class DownloadReadyDataset(Dataset):
    """Read only published JPEGs; fail after a bounded network outage."""

    def __init__(self, samples, timeout=120):
        self.samples = samples
        self.records = samples.records
        self.timeout = timeout

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = Path(self.records[index]["image_path"])
        start = time.monotonic()
        announced = False
        while not path.is_file():
            if not announced:
                print(f"Waiting for cached image: {path}", flush=True)
                announced = True
            if time.monotonic() - start >= self.timeout:
                raise TimeoutError(f"Downloader has not published {path} after {self.timeout}s")
            time.sleep(min(1, self.timeout))
        sample = self.samples[index]
        sample["metadata"]["cache_wait_seconds"] = time.monotonic() - start
        return sample
