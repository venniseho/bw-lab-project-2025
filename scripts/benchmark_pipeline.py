"""Sweep GPU batch sizes in fresh processes; run inside an srun allocation."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwlab.infrastructure import require_slurm_step
from bwlab.runtime import RuntimeConfig
from experiments.sam3_fragmentation.experiment import build_data_module, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-inputs", type=int, default=128)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    parser.add_argument("--preview-batches", type=int, default=0)
    args = parser.parse_args()
    import os
    require_slurm_step(os.environ)
    if min(args.batch_sizes) < 1 or args.workers < 0 or args.max_inputs < max(args.batch_sizes):
        parser.error("positive batch sizes and max-inputs >= largest batch size required")
    config = load_config(args.config)
    module = build_data_module(config)
    available = len(module.val_dataset() if config.subset == "val" else module.test_dataset())
    if min(args.max_inputs, available) < max(args.batch_sizes):
        parser.error(f"dataset has only {available} inputs; cannot benchmark the requested full batches")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = RuntimeConfig(compile_encoder=args.compile,
                            cudnn_benchmark=not args.no_cudnn_benchmark,
                            preview_batches=args.preview_batches)
    with (args.output / "batch_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["batch_size", "status", "inputs_per_second", "evaluation_seconds",
                  "peak_allocated_bytes", "peak_reserved_bytes", "input_count"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for size in args.batch_sizes:
            current = replace(config, batch_size=size, num_workers=args.workers,
                              max_inputs=args.max_inputs, runtime=runtime)
            config_path = args.output / f"batch-{size}.json"
            config_path.write_text(json.dumps(asdict(current), default=str, indent=2)+"\n", encoding="utf-8")
            output = args.output / f"batch-{size}"
            with (args.output / f"batch-{size}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, "-m", "experiments.sam3_fragmentation.experiment",
                                          "run", "--config", str(config_path.resolve()),
                                          "--output", str(output.resolve())], stdout=log, stderr=subprocess.STDOUT,
                                         cwd=Path(__file__).resolve().parents[1])
            if process.returncode:
                message = (args.output / f"batch-{size}.log").read_text(encoding="utf-8")
                oom = any(text in message.lower() for text in
                          ("cuda out of memory", "cuda error: out of memory"))
                writer.writerow({"batch_size": size, "status": "out_of_memory" if oom else "failed"})
                stream.flush()
                if not oom:
                    raise RuntimeError(f"batch {size} failed; inspect {args.output / f'batch-{size}.log'}")
                # Larger image batches are unlikely to fit; preserve the failed log.
                break
            timing = json.loads((output / "timing.json").read_text())
            writer.writerow({"batch_size": size, "status": "complete",
                             **{key: timing[key] for key in fields if key in timing}})
            stream.flush()
            print(f"batch={size}: {timing['inputs_per_second']:.3f} inputs/s", flush=True)


if __name__ == "__main__":
    main()
