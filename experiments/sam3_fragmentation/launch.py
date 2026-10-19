"""Submit configurations via Submitit; experimental logic stays in run()."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from experiments.sam3_fragmentation.experiment import load_config, run
from bwlab.runtime import RuntimeConfig


def execute_config(config_path, output_path, local=False):
    """Called by Submitit on its compute node, or directly for local debugging."""
    config = load_config(Path(config_path))
    runtime = config.runtime or RuntimeConfig()
    # Submitit runs Python on its allocated node without an srun step. Permit
    # this only inside the Submitit process; direct local use is explicit.
    if not local:
        import os
        import submitit
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Submitit worker requires a SLURM allocation")
        submitit.JobEnvironment()  # Validate this is a real Submitit job context.
    config = replace(config, runtime=replace(runtime, allow_local=True))
    return run(config, Path(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", action="store_true", help="Run sequentially on your own GPU or allocated compute node")
    parser.add_argument("--account", default="aip-sven")
    parser.add_argument("--partition", default="gpubase_bygpu_b1")
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument("--after-job", nargs="+", default=[], help="Optional SLURM afterok dependencies")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.local:
        for index, path in enumerate(args.configs):
            execute_config(path.resolve(), (args.output / f"run-{index:03d}").resolve(), local=True)
        return
    import submitit
    executor = submitit.AutoExecutor(folder=str((args.output / "slurm").resolve()))
    executor.update_parameters(timeout_min=args.minutes, gpus_per_node=1, cpus_per_task=args.cpus,
                               mem_gb=args.memory_gb, slurm_account=args.account,
                               slurm_partition=args.partition, nodes=1, tasks_per_node=1)
    if args.after_job:
        if any(not identifier.replace("_", "").isdigit() for identifier in args.after_job):
            parser.error("invalid SLURM dependency ID")
        executor.update_parameters(slurm_additional_parameters={"dependency": "afterok:"+":".join(args.after_job)})
    jobs = []
    with executor.batch():
        for index, path in enumerate(args.configs):
            output = (args.output / f"run-{index:03d}").resolve()
            job = executor.submit(execute_config, path.resolve(), output)
            jobs.append({"job": job, "config": str(path.resolve()), "output": str(output)})
    (args.output / "jobs.json").write_text(json.dumps([
        {"job_id": row["job"].job_id, "config": row["config"], "output": row["output"]} for row in jobs
    ], indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
