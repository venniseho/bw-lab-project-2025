"""Queue prepared jobs and dependent evaluations, recording every submission ID."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def submit(plan_path, repo, output, replay_source, replay_config):
    plan = json.loads(plan_path.read_text())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    ledger_path = output / "submitted_jobs.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {
            "plan_sha256": digest,
            "repo": str(repo),
            "replay_source": str(replay_source),
            "replay_config": str(replay_config),
            "jobs": {},
        }
    )
    for key, value in (
        ("plan_sha256", digest),
        ("repo", str(repo)),
        ("replay_source", str(replay_source)),
        ("replay_config", str(replay_config)),
    ):
        if ledger[key] != value:
            raise ValueError(f"Existing submission differs in {key}; do not reuse this directory")
    if not (replay_source / "complete.json").is_file() or not replay_config.is_file():
        raise FileNotFoundError("Original fragmentation replay source/config missing")

    def queue(name, script, args, hours, dependency=None):
        if name in ledger["jobs"]:
            return ledger["jobs"][name]
        command = [
            "sbatch",
            "--parsable",
            f"--time={hours:02d}:00:00",
            f"--job-name=bwlab-{name}",
            f"--output={output / (name + '-%j.out')}",
        ]
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        command.extend([str(repo / "scripts" / script), str(repo), *map(str, args)])
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job_id}")
        ledger["jobs"][name] = job_id
        temporary = ledger_path.with_suffix(".part")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(ledger_path)
        print(f"{name}: {job_id}", flush=True)
        return job_id

    smoke = queue(
        "preflight-mixture-accum",
        "vulcan_point_training.sbatch",
        [plan["smoke_config"], "mask_decoder_only", output / "preflight", "17"],
        1,
    )
    replay = queue(
        "preflight-source-replay",
        "vulcan_point_transfer.sbatch",
        [replay_source, output / "source-replay", replay_config, "--require-source-replay"],
        2,
    )
    queue(
        "eval-existing-medium-control",
        "vulcan_point_transfer.sbatch",
        [
            replay_source.parent / "training-fence-medium",
            output / "evaluation-existing-medium-control",
            *plan["validation_configs"],
        ],
        2,
        f"{smoke}:{replay}",
    )
    for job in plan["jobs"]:
        name = job["name"]
        run_path = output / f"training-{name}"
        training = queue(
            name,
            "vulcan_point_training.sbatch",
            [job["config"], "mask_decoder_only", run_path],
            5 if job["evaluate_all"] else 2,
            f"{smoke}:{replay}",
        )
        if job["evaluate_all"]:
            queue(
                f"eval-{name}",
                "vulcan_point_transfer.sbatch",
                [run_path, output / f"evaluation-{name}", *plan["validation_configs"]],
                2,
                training,
            )
    print(f"Submitted/recorded {len(ledger['jobs'])} jobs. Ledger: {ledger_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path, required=True)
    parser.add_argument("--replay-config", type=Path, required=True)
    args = parser.parse_args()
    submit(
        args.plan.resolve(),
        args.repo.resolve(),
        args.output.resolve(),
        args.replay_source.resolve(),
        args.replay_config.resolve(),
    )
