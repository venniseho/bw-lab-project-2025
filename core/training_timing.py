"""End-to-end phase throughput, distinguishing instances from source images."""

import time

import torch


class PhaseTiming:
    """Synchronize only at phase boundaries, not at every training step."""

    def __init__(self, model, phase, epoch):
        self.device = next(model.parameters()).device
        self.phase, self.epoch = phase, epoch
        self.instances, self.sources = 0, set()
        self._synchronize()
        self.start = time.perf_counter()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def observe(self, sample):
        self.instances += 1
        self.sources.add(sample.get("metadata", {}).get("image_id", sample["__key__"]))

    def finish(self):
        self._synchronize()
        seconds = time.perf_counter() - self.start
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "instances": self.instances,
            "image_presentations": self.instances,
            "unique_source_images": len(self.sources),
            "wall_seconds": seconds,
            "instances_per_second": self.instances / seconds,
            "images_per_second": self.instances / seconds,
            "unique_source_images_per_second": len(self.sources) / seconds,
        }
