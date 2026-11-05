"""Optional W&B monitoring around evaluation, separate from scientific logic."""

import csv
import json
import math
from collections import defaultdict
from statistics import mean


def accuracy_rows(path):
    """Keep full/visible denominators separate; undefined IoU is never zero."""
    groups = defaultdict(lambda: {"full_iou": [], "visible_iou": []})
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "complete":
                continue
            values = groups[row["condition"]]
            for metric in values:
                if row.get(metric) and math.isfinite(float(row[metric])):
                    values[metric].append(float(row[metric]))
    return [
        [
            condition,
            len(values["full_iou"]),
            mean(values["full_iou"]) if values["full_iou"] else None,
            len(values["visible_iou"]),
            mean(values["visible_iou"]) if values["visible_iou"] else None,
        ]
        for condition, values in sorted(groups.items())
    ]


class EvaluationTracking:
    """Collect system metrics for the whole run and publish final local summaries."""

    def __init__(self, mode, output, config):
        if mode not in ("disabled", "offline", "online"):
            raise ValueError(f"Unknown W&B mode: {mode}")
        self.mode, self.output, self.config = mode, output, config
        self.run = None

    def __enter__(self):
        if self.mode == "disabled":
            return self
        import os
        import wandb

        self.output.parent.mkdir(parents=True, exist_ok=True)
        group = self.output.parent.name
        if group == "runs":
            group = self.output.parent.parent.name
        self.run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "SAM-finetuning"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=os.environ.get("WANDB_RUN_GROUP", group),
            job_type=f"evaluation-{self.config['condition']}",
            name=f"evaluation-{self.config['condition']}",
            dir=str(self.output.parent),
            config=json.loads(json.dumps(self.config, default=str)),
            mode=self.mode,
        )
        print(f"W&B evaluation dashboard: {self.run.url}", flush=True)
        return self

    def log_results(self):
        if self.run is None:
            return
        import wandb

        timing = json.loads((self.output / "timing.json").read_text())
        self.run.summary.update(
            {
                f"timing/{key}": value
                for key, value in timing.items()
                if isinstance(value, (int, float))
            }
        )
        path = self.output / "predictions/image_results.csv"
        if path.is_file():
            table = wandb.Table(
                columns=[
                    "condition",
                    "full_count",
                    "mean_full_iou",
                    "visible_count",
                    "mean_visible_iou",
                ],
                data=accuracy_rows(path),
            )
            self.run.log({"evaluation/accuracy_by_condition": table})

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            self.run.finish(exit_code=int(exc_type is not None))
