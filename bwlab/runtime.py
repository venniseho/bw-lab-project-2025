"""Explicit inference options and resource evidence for throughput experiments."""
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import subprocess

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    batched: bool = True
    prompt_batch_size: int = 9
    pin_memory: bool = True
    cudnn_benchmark: bool = True
    compile_encoder: bool = False
    compile_mode: str = "default"
    warmup_batches: int = 1
    preview_batches: int = 1
    monitor_gpu: bool = True
    allow_local: bool = False

    def __post_init__(self):
        if self.prompt_batch_size < 1 or min(self.warmup_batches, self.preview_batches) < 0:
            raise ValueError("invalid runtime batch counts")


def configure_model(model, config):
    model.eval()
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if config.compile_encoder:
        # Compile the active image encoder, not text grounding or the whole
        # Python predictor (which includes NumPy and variable-size mask export).
        model.backbone.forward_image = torch.compile(
            model.backbone.forward_image, mode=config.compile_mode)
    parameters = list(model.parameters())
    dtypes = {}
    for parameter in parameters:
        key = str(parameter.dtype)
        dtypes[key] = dtypes.get(key, 0)+parameter.numel()
    return {"runtime": asdict(config), "parameter_count": sum(p.numel() for p in parameters),
            "parameters_by_dtype": dtypes,
            "parameter_bytes": sum(p.numel()*p.element_size() for p in parameters),
            "buffer_bytes": sum(b.numel()*b.element_size() for b in model.buffers()),
            "precision": "native checkpoint dtypes; no additional autocast or TF32",
            "compile_scope": "backbone.forward_image" if config.compile_encoder else None}


def gpu_memory():
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_total_bytes": properties.total_memory,
            "cuda_device_index": device, "device_uuid": str(getattr(properties, "uuid", "unavailable")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


class GPUMonitor:
    """Node GPU utilization samples; identify the assigned device via run metadata."""
    def __init__(self, path: Path, enabled=True):
        self.path, self.enabled = path, enabled
        self.process = self.stream = None
        self.status = "disabled"

    def __enter__(self):
        if self.enabled:
            self.stream = self.path.open("w", encoding="utf-8")
            try:
                self.process = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used",
                    "--format=csv", "--loop-ms=1000"], stdout=self.stream,
                    stderr=subprocess.DEVNULL)
                self.status = "started"
            except OSError as error:
                self.status = str(error)
        return self

    def __exit__(self, *args):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            else:
                self.status = f"exited ({self.process.returncode})"
        if self.stream is not None:
            self.stream.close()
