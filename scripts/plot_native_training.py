"""Plot verified native training updates, including partially completed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run, output):
    with (run / "optimizer_steps.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("No verified optimizer updates yet")
    fields = (("Losses/train_all_loss_mask", "Mask focal loss"),
              ("Losses/train_all_loss_dice", "Mask Dice loss"),
              ("Losses/train_all_loss", "Total weighted training loss"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, (field, label) in zip(axes, fields):
        axis.plot([row["optimizer_step"] for row in rows], [row[field] for row in rows], marker="o", markersize=3)
        axis.set_xlabel("Verified optimizer update")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    fig.suptitle("Native SAM3 training smoke check")
    fig.supxlabel("Tiny COCO train2017 subset; not a benchmark result. Randomized native training transforms.\n"
                  "Mask components are unweighted; total includes weighted box/classification and mask losses.", fontsize=9)
    output.mkdir(parents=True, exist_ok=False)
    for extension in ("png", "svg"):
        path = output / f"training_loss.{extension}"
        fig.savefig(path, dpi=180)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run_dir, args.output_dir)
