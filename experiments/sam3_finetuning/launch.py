"""Run an external config through Meta's native single-node launcher in Slurm."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from bwlab.infrastructure import file_fingerprint, require_slurm_step
from bwlab.runtime import GPUMonitor
from .config import NATIVE_REVISION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Resolve config and load/collate masks on CPU; no model")
    args = parser.parse_args(argv)
    import sam3
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import register_omegaconf_resolvers
    native = Path(sam3.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
    if revision != NATIVE_REVISION:
        raise RuntimeError(f"Wrong native SAM3 revision: {revision}")
    register_omegaconf_resolvers()
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    if config.bwlab.native_revision != revision:
        raise RuntimeError("Config and native revision differ")
    if args.preflight:
        from hydra.utils import instantiate
        for split in ("train", "val"):
            dataset = instantiate(config.trainer.data[split].dataset)
            collate = instantiate(config.trainer.data[split].collate_fn)
            batch = next(iter(collate([dataset[0]]).values()))
            target = batch.find_targets[0]
            if target.segments is None or not target.segments.any():
                raise RuntimeError(f"{split}: no mask targets in native collated batch")
            print(f"PREFLIGHT {split}: {len(dataset)} sources; "
                  f"images={tuple(batch.img_batch.shape)}; masks={tuple(target.segments.shape)}", flush=True)
        return
    require_slurm_step(os.environ)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm compute GPU is required")
    from sam3.train.train import single_node_runner
    import socket
    output = Path(config.launcher.experiment_log_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")
    provenance = {"native_revision": revision, "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0), "input_config": file_fingerprint(args.config),
                  "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "requested_cpu_cores": os.environ.get("SLURM_CPUS_PER_TASK"),
                  "loader_workers": config.trainer.data.train.num_workers,
                  "bwlab_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "checkpoint": file_fingerprint(Path(config.trainer.model.checkpoint_path)),
                  "train_annotation": file_fingerprint(Path(config.trainer.data.train.dataset.ann_file)),
                  "heldout_annotation": file_fingerprint(Path(config.trainer.data.val.dataset.ann_file)),
                  "scope": config.bwlab.scope, "task": config.bwlab.task}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    # Obtain an available local rendezvous port for the native world-size-one job.
    with socket.socket() as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        port = rendezvous.getsockname()[1]
    started = time.perf_counter()
    try:
        with GPUMonitor(output / "gpu_utilization.csv", True):
            single_node_runner(config, port)
    finally:
        (output / "process_timing.json").write_text(json.dumps({
            "launcher_wall_seconds": time.perf_counter()-started,
            "scope": "model build, checkpoint load, training, held-out loss and checkpoint writing"}, indent=2)+"\n")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
