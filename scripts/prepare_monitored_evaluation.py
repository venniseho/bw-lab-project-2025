"""Reuse an existing source manifest for a fixed, monitored evaluation of both arms."""

import argparse
import json
from pathlib import Path


def prepare(input_dir, output_dir, count=512, workers=8):
    if count < 1 or workers < 0:
        raise ValueError("Positive source count and nonnegative workers required")
    configurations, manifests = {}, {}
    for arm, batch in (("fence", 1), ("fragmentation", 2)):
        path = input_dir / f"{arm}.json"
        config = json.loads(path.read_text())
        for field in ("manifest", "checkpoint", "bpe_path"):
            if config.get(field) is not None:
                config[field] = str((path.parent / config[field]).resolve())
        records = json.loads(Path(config["manifest"]).read_text())
        selected = records[:count]
        ids = [record.get("image_id", record["image_path"]) for record in selected]
        if len(selected) < count or len(set(ids)) != count:
            raise ValueError(
                f"{arm}: need {count} distinct source records; use the one-instance throughput manifest"
            )
        manifests[arm] = selected
        config.update(
            condition=arm,
            batch_size=batch,
            num_workers=workers,
            max_inputs=count * (4 if arm == "fence" else 1),
        )
        config["runtime"] = {
            **(config.get("runtime") or {}),
            "batched": True,
            "preview_batches": 0,
            "warmup_batches": 1,
            "compile_encoder": False,
            "monitor_gpu": True,
        }
        configurations[arm] = config
    if manifests["fence"] != manifests["fragmentation"]:
        raise ValueError("Both arms must use the same source/target records")
    output_dir.mkdir(parents=True, exist_ok=False)
    for arm, config in configurations.items():
        (output_dir / f"{arm}.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Same {count} sources: {4 * count} fence inputs, {count} fragmentation inputs.")
    print(f"Workers={workers}; fence batch=1; fragmentation batch=2; previews off.")
    print("Configuration only; no inference, downloads or fine-tuning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.input_dir.resolve(), args.output_dir.resolve(), args.count, args.workers)


if __name__ == "__main__":
    main()
