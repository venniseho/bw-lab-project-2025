"""Persist training logs and figures; tracking is supplied by the experiment."""

import json
from core.visualization import training_contact_sheet


class TrainingOutput:
    def __init__(self, path, tracking=None, image_factory=None):
        self.path, self.tracking = path, tracking
        self.image_factory = image_factory

    def record(self, channel, record):
        with (self.path / f"{channel}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if channel in ("train", "summary", "timing"):
            print(json.dumps(record), flush=True)
        if self.tracking and channel == "timing":
            self.tracking.log(
                {
                    f"timing/{record['phase']}/{key}": value
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
            self.tracking.log(
                {
                    "heldout/mean_iou": record["mean_iou"],
                    "heldout/mean_loss": record["mean_loss"],
                    "heldout/instances": record["instances"],
                    "heldout/epoch": record["epoch"],
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

    def preview(self, sample, point, before, after, epoch, index):
        path = self.path / "previews" / f"epoch-{epoch}-{index}.png"
        training_contact_sheet(path, sample, point, before, after)
        if self.tracking and self.image_factory:
            caption = (
                f"{sample['__key__']}: target-point prompt; "
                f"IoU before={before['iou']:.3f}, after={after['iou']:.3f}"
            )
            self.tracking.log(
                {
                    "examples/epoch": epoch,
                    f"examples/instance_{index}": self.image_factory(str(path), caption=caption),
                }
            )

    def close(self, exit_code=0):
        if self.tracking:
            self.tracking.finish(exit_code=exit_code)
