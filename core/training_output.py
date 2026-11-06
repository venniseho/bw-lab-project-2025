"""Persist training logs and figures; tracking is supplied by the experiment."""

import json
import csv
from core.visualization import training_contact_sheet


class TrainingOutput:
    def __init__(self, path, tracking=None, image_factory=None):
        self.path, self.tracking = path, tracking
        self.image_factory = image_factory

    def record(self, channel, record):
        with (self.path / f"{channel}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if channel in (
            "train",
            "heldout",
            "timing",
            "summary",
            "checkpoints",
            "benchmark",
            "benchmark_samples",
        ):
            csv_path = self.path / f"{channel}.csv"
            fresh = not csv_path.exists()
            with csv_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                if fresh:
                    writer.writeheader()
                writer.writerow(record)
        if channel in ("train", "summary", "timing", "benchmark"):
            print(json.dumps(record), flush=True)
        if self.tracking and channel == "benchmark":
            self.tracking.log(
                {
                    f"benchmark/{record['mode']}/{key}": value
                    for key, value in record.items()
                    if isinstance(value, (int, float))
                }
            )
        if self.tracking and channel == "timing":
            phase = "validation_full" if record.get("validation_set") == "full" else record["phase"]
            self.tracking.log(
                {
                    f"timing/{phase}/{key}": value
                    for key, value in record.items()
                    if isinstance(value, (int, float))
                }
            )
        if self.tracking and channel == "train":
            metrics = {
                f"train/{key}": value
                for key, value in record.items()
                if isinstance(value, (int, float))
            }
            self.tracking.log(metrics)
        if self.tracking and channel == "summary":
            prefix = "heldout_full" if record.get("validation_set") == "full" else "heldout"
            self.tracking.log(
                {
                    f"{prefix}/mean_iou": record["mean_iou"],
                    f"{prefix}/mean_loss": record["mean_loss"],
                    f"{prefix}/instances": record["instances"],
                    f"{prefix}/epoch": record["epoch"],
                    f"{prefix}/step": record.get("step", 0),
                }
            )

    def save_json(self, name, record):
        (self.path / f"{name}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        if self.tracking and name == "complete":
            self.tracking.summary.update(
                {
                    f"run/{key}": record[key]
                    for key in ("steps", "wall_seconds", "peak_gpu_memory_bytes")
                }
            )

    def preview(
        self, sample, point, before, after, epoch, index, step=None, validation_set="interval"
    ):
        suffix = "" if step is None else f"-step-{step}"
        if validation_set == "full":
            suffix += "-full"
        path = self.path / "previews" / f"epoch-{epoch}{suffix}-{index}.png"
        training_contact_sheet(path, sample, point, before, after)
        if self.tracking and self.image_factory:
            prefix = "examples_full" if validation_set == "full" else "examples"
            caption = (
                f"{sample['__key__']}: target-point prompt; "
                f"IoU before={before['iou']:.3f}, after={after['iou']:.3f}"
            )
            self.tracking.log(
                {
                    f"{prefix}/epoch": epoch,
                    f"{prefix}/step": step or 0,
                    f"{prefix}/instance_{index}": self.image_factory(str(path), caption=caption),
                }
            )

    def close(self, exit_code=0):
        if self.tracking:
            self.tracking.finish(exit_code=exit_code)
