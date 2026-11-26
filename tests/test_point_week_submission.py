"""Submission dependencies must gate training and avoid duplicate jobs on retry."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_point_transfer import verify_replay
from scripts.submit_point_week import submit


class WeekSubmissionTests(unittest.TestCase):
    def test_gates_training_and_evaluation_and_reuses_recorded_job_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay = root / "source"
            replay.mkdir()
            (replay / "complete.json").write_text("{}")
            config = root / "config.json"
            config.write_text("{}")
            plan = root / "jobs.json"
            plan.write_text(
                json.dumps(
                    {
                        "smoke_config": str(config),
                        "validation_configs": [str(config)],
                        "jobs": [
                            {"name": "mixed", "config": str(config), "evaluate_all": True},
                            {"name": "pilot", "config": str(config), "evaluate_all": False},
                        ],
                    }
                )
            )
            with patch(
                "scripts.submit_point_week.subprocess.check_output",
                side_effect=[f"{i}\n" for i in range(100, 106)],
            ) as sbatch:
                submit(plan, root, root, replay, config)
                commands = [call.args[0] for call in sbatch.call_args_list]
                self.assertIn("--dependency=afterok:100:101", commands[2])
                self.assertIn("--dependency=afterok:100:101", commands[3])
                self.assertIn("--dependency=afterok:103", commands[4])
                self.assertIn("--dependency=afterok:100:101", commands[5])
                self.assertIn("--require-source-replay", commands[1])
                submit(plan, root, root, replay, config)
                self.assertEqual(sbatch.call_count, 6)
            self.assertEqual(len(json.loads((root / "submitted_jobs.json").read_text())["jobs"]), 6)
            plan.write_text(plan.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "plan_sha256"):
                submit(plan, root, root, replay, config)

    def test_replay_checks_condition_counts_and_aggregate_accuracy(self):
        source = {"transform": "fragmentation"}
        original = {"final_validation_summary": {"instances": 2, "mean_iou": 0.28}}
        result = {"fragmentation": {"finetuned": {"instances": 2, "mean_full_iou": 0.28}}}
        verify_replay(source, original, {"fragmentation": source}, result)
        result["fragmentation"]["finetuned"]["mean_full_iou"] = 0.01
        with self.assertRaisesRegex(RuntimeError, "replay IoU"):
            verify_replay(source, original, {"fragmentation": source}, result)
        with self.assertRaisesRegex(ValueError, "transform"):
            verify_replay(source, original, {"clean": {"transform": "identity"}}, result)


if __name__ == "__main__":
    unittest.main()
