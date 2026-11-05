"""Evaluation monitoring preserves missing values and reports real completion."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.evaluation_tracking import EvaluationTracking
from scripts.prepare_monitored_evaluation import prepare


class EvaluationTrackingTests(unittest.TestCase):
    def test_preparation_reuses_the_same_sources_and_does_not_mutate_old_configs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [
                {"__key__": str(i), "image_id": i, "image_path": f"{i}.jpg"} for i in range(3)
            ]
            (root / "records.json").write_text(json.dumps(records))
            for arm in ("fence", "fragmentation"):
                (root / f"{arm}.json").write_text(
                    json.dumps({"manifest": "records.json", "seed": 42, "points_per_side": 3})
                )
            prepare(root, root / "new", count=2)
            fence = json.loads((root / "new/fence.json").read_text())
            frag = json.loads((root / "new/fragmentation.json").read_text())
            self.assertEqual(fence["manifest"], frag["manifest"])
            self.assertEqual((fence["max_inputs"], frag["max_inputs"]), (8, 2))
            self.assertEqual((fence["batch_size"], frag["batch_size"]), (1, 2))
            self.assertEqual(fence["seed"], 42)
            self.assertEqual(fence["points_per_side"], 3)
            self.assertEqual(
                json.loads((root / "fence.json").read_text())["manifest"], "records.json"
            )

    def test_online_context_logs_timing_and_condition_table_with_correct_denominators(self):
        run = Mock(url="https://example.invalid/run")
        sdk = SimpleNamespace(init=Mock(return_value=run), Table=Mock(return_value="table"))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"wandb": sdk}):
            output = Path(folder) / "fence"
            (output / "predictions").mkdir(parents=True)
            (output / "timing.json").write_text(
                json.dumps({"input_count": 3, "pipeline_seconds": 1.5})
            )
            (output / "predictions/image_results.csv").write_text(
                "condition,status,full_iou,visible_iou\nheavy,complete,0.2,\n"
                "heavy,complete,0.4,0.6\nheavy,failed,1.0,1.0\n"
            )
            with EvaluationTracking("online", output, {"condition": "fence"}) as tracking:
                tracking.log_results()
            self.assertEqual(sdk.init.call_args.kwargs["job_type"], "evaluation-fence")
            row = sdk.Table.call_args.kwargs["data"][0]
            self.assertEqual(row[0:2], ["heavy", 2])
            self.assertAlmostEqual(row[2], 0.3)
            self.assertEqual(row[3:], [1, 0.6])
            run.summary.update.assert_called_once_with(
                {"timing/input_count": 3, "timing/pipeline_seconds": 1.5}
            )
            run.finish.assert_called_once_with(exit_code=0)

    def test_failure_is_marked_failed_and_disabled_needs_no_sdk(self):
        run = Mock()
        sdk = SimpleNamespace(init=Mock(return_value=run))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"wandb": sdk}):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                with EvaluationTracking("online", Path(folder) / "run", {"condition": "fence"}):
                    raise RuntimeError("model failure")
            run.finish.assert_called_once_with(exit_code=1)
        with patch.dict(sys.modules, {"wandb": None}):
            with EvaluationTracking("disabled", Path("unused"), {}) as tracking:
                tracking.log_results()
