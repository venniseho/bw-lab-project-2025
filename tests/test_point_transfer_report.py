"""Ensure transfer plots use paired targets, not CSV row order or unmatched prompts."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_point_transfer import FENCES, SOURCES, paired_records, report


def record(key, step, value):
    return {
        "key": key,
        "step": str(step),
        "image_id": key,
        "annotation_id": key,
        "point_x": "2.0",
        "point_y": "3.0",
        "point_seed": "42",
        "transform_json": "[]",
        "target_pixels": "10",
        "visible_pixels": "8",
        "full_iou": str(value),
        "visible_iou": str(value),
    }


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TransferReportTests(unittest.TestCase):
    def test_pairs_by_identity_and_rejects_changed_prompts_or_duplicate_records(self):
        rows = [
            record("a", 0, 0.1),
            record("b", 0, 0.3),
            record("b", 10, 0.8),
            record("a", 10, 0.4),
        ]
        pairs = paired_records(rows, 10)
        self.assertEqual([pair["key"] for pair in pairs], ["a", "b"])
        self.assertAlmostEqual(pairs[0]["change_full_iou"], 0.3)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_records(rows + [rows[0]], 10)
        rows[-1] = {**rows[-1], "point_x": "9.0"}
        with self.assertRaisesRegex(ValueError, "point_x"):
            paired_records(rows, 10)

    def test_renders_nine_comparisons_from_seven_completed_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cross = root / "cross"
            for source in SOURCES:
                path = cross / f"{source}-transfer"
                path.mkdir(parents=True)
                targets = FENCES if source == "fragmentation" else ("fragmentation",)
                results = {}
                for target in targets:
                    folder = path / target
                    folder.mkdir()
                    write_rows(folder / "heldout.csv", [record("a", 0, 0.1), record("a", 10, 0.4)])
                    write_rows(
                        folder / "timing.csv",
                        [
                            {"step": 0, "images_per_second": 10},
                            {"step": 10, "images_per_second": 11},
                        ],
                    )
                    results[target] = {
                        phase: {"instances": 1, "mean_full_iou": value, "mean_visible_iou": value}
                        for phase, value in (("pretrained", 0.1), ("finetuned", 0.4))
                    }
                (path / "complete.json").write_text(
                    json.dumps({"steps": 10, "training_updates": 0, "results": results})
                )
                (path / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scope": "cross-condition-evaluation",
                            "targets": {target: {"seed": 42} for target in targets},
                        }
                    )
                )
            output = root / "report"
            report(cross, output)
            with (output / "run_summary.csv").open() as stream:
                summaries = list(csv.DictReader(stream))
            self.assertEqual(len(summaries), 9)
            self.assertTrue(
                all(abs(float(row["change_full_iou"]) - 0.3) < 1e-6 for row in summaries)
            )
            self.assertEqual(len(list(output.glob("*.png"))), 4)
            self.assertEqual(len(list(output.glob("*.svg"))), 4)
            self.assertIn("pretrained", (output / "index.html").read_text(encoding="utf-8"))
            (cross / "fence-low-transfer" / "complete.json").unlink()
            with self.assertRaises(FileNotFoundError):
                report(cross, root / "incomplete")
            self.assertFalse((root / "incomplete").exists())


if __name__ == "__main__":
    unittest.main()
