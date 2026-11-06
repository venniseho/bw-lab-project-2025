"""End-to-end phase throughput, distinguishing instances from source images."""

import time

import torch


class PhaseTiming:
    """Synchronize only at phase boundaries, not at every training step."""

    def __init__(self, model, phase, epoch):
        self.device = next(model.parameters()).device
        self.phase, self.epoch = phase, epoch
        self.instances, self.sources = 0, set()
        self.excluded_seconds = 0.0
        self._synchronize()
        self.start = time.perf_counter()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def observe(self, sample):
        self.instances += 1
        self.sources.add(sample.get("metadata", {}).get("image_id", sample["__key__"]))

    def exclude(self, operation):
        """Pause phase accounting for interval validation and checkpoint I/O."""
        self._synchronize()
        start = time.perf_counter()
        operation()
        self._synchronize()
        self.excluded_seconds += time.perf_counter() - start

    def finish(self):
        self._synchronize()
        seconds = time.perf_counter() - self.start - self.excluded_seconds
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "step": 0,
            "encoder_forwards": self.instances,
            "validation_set": None,
            "instances": self.instances,
            "image_presentations": self.instances,
            "unique_source_images": len(self.sources),
            "wall_seconds": seconds,
            "instances_per_second": self.instances / seconds,
            "images_per_second": self.instances / seconds,
            "unique_source_images_per_second": len(self.sources) / seconds,
        }
