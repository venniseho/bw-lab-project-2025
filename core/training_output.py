"""Persist training logs and figures; tracking is supplied by the experiment."""

import json
from core.visualization import training_contact_sheet


class TrainingOutput:
    def __init__(self, path, tracking=None):
        self.path, self.tracking = path, tracking

    def record(self, channel, record):
        with (self.path / f"{channel}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if channel in ("train", "summary"):
            print(json.dumps(record), flush=True)
        if self.tracking and channel == "train":
            metrics = {
                f"train/{key}": value
                for key, value in record.items()
                if isinstance(value, (int, float))
            }
            self.tracking.log(metrics)
        if self.tracking and channel == "summary":
            self.tracking.log({"heldout/mean_iou": record["mean_iou"], "epoch": record["epoch"]})

    def save_json(self, name, record):
        (self.path / f"{name}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )

    def preview(self, sample, point, before, after, epoch, index):
        training_contact_sheet(
            self.path / "previews" / f"epoch-{epoch}-{index}.png", sample, point, before, after
        )

    def close(self):
        if self.tracking:
            self.tracking.finish()
