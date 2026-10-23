import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from bwlab.models.proposals import Proposal
from bwlab.runtime import RuntimeConfig
from scripts._equivalence_metrics import comparison_rows, summarize
from scripts.check_batch_equivalence import prediction_stages
from scripts import check_batch_equivalence


def proposal(mask, identifier=0, score=0.9):
    return Proposal(np.packbits(mask), mask.shape,
                    {"proposal_id": identifier, "grid_point_index": identifier//3,
                     "candidate_index": identifier%3, "point_xy": [2.0, 3.0],
                     "predicted_iou_score": score})


class EquivalenceMetricsTests(unittest.TestCase):
    def test_identical_empty_masks_have_pair_iou_one(self):
        empty = np.zeros((10, 10), bool)
        rows = list(comparison_rows("a", [proposal(empty)], [proposal(empty)],
                                    "test", 1e-5, np.ones_like(empty)))
        self.assertEqual(rows[0]["mask_iou"], 1)
        self.assertFalse(rows[0]["strict_mismatch"])
        self.assertEqual(rows[0]["reference_target_iou"], 0)

    def test_small_pixel_fraction_can_be_large_for_small_masks(self):
        a = np.zeros((10, 10), bool)
        a[0, 0] = True
        b = np.zeros_like(a)
        rows = list(comparison_rows("a", [proposal(a)], [proposal(b)], "test", 1e-5, a))
        self.assertEqual(rows[0]["different_pixels"], 1)
        self.assertEqual(rows[0]["different_fraction"], 0.01)
        self.assertEqual(rows[0]["mask_iou"], 0)
        self.assertEqual(rows[0]["target_iou_difference"], 1)
        self.assertTrue(rows[0]["strict_mismatch"])

    def test_score_only_drift_and_changed_choice_are_visible(self):
        mask = np.ones((3, 4), bool)
        old = [proposal(mask, 0, 0.91), proposal(mask, 1, 0.90)]
        new = [proposal(mask, 0, 0.89), proposal(mask, 1, 0.90)]
        rows = list(comparison_rows("a", old, new, "test", 1e-5, mask))
        report = summarize(rows, [old], [new])
        self.assertEqual(report["strict_mismatch_count"], 1)
        self.assertEqual(report["min_mask_iou"], 1)
        self.assertEqual(report["selected_candidate_changes"], 1)
        self.assertAlmostEqual(report["max_score_difference"], 0.02)

    def test_prompt_mapping_changes_remain_hard_errors(self):
        mask = np.ones((3, 4), bool)
        a, b = proposal(mask), proposal(mask)
        b.metadata["point_xy"] = [9, 9]
        with self.assertRaisesRegex(RuntimeError, "correspondence"):
            list(comparison_rows("a", [a], [b], "test", 1e-5, mask))

    def test_diagnostic_matrix_isolates_image_and_prompt_batches(self):
        calls = []
        class Adapter:
            def __init__(self, model, processor, sides, prompts, pin, sync):
                self.prompts = prompts
            def predict_batch(self, images):
                calls.append((len(images), self.prompts))
                return [(["candidate"], {}) for _ in images]
        reference = SimpleNamespace(predict=lambda image: (["candidate"], {}))
        with patch("scripts.check_batch_equivalence.BatchedSam3GridAdapter", Adapter), \
                patch("scripts.check_batch_equivalence.configure_model") as configure:
            stages = list(prediction_stages(None, None, reference, [None, None], 3,
                                           RuntimeConfig(prompt_batch_size=9), True))
        self.assertEqual([name for name, _ in stages], ["reference_repeat", "tensor_single", "image_batch_only",
                                                       "prompt_batch_only", "combined_without_benchmark", "configured_runtime"])
        self.assertEqual(calls, [(1, 1), (1, 1), (2, 1), (1, 9), (1, 9), (2, 9), (2, 9)])
        self.assertEqual(configure.call_count, 1)

    def test_main_reports_numerical_differences_without_failing_by_default(self):
        target = np.ones((3, 4), bool)
        altered = target.copy()
        altered[0, 0] = False
        class Adapter:
            def __init__(self, model, processor, sides, prompts, pin, sync):
                self.mask = altered if prompts > 1 else target
            def predict_batch(self, images):
                return [([proposal(self.mask)], {}) for _ in images]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"diagnostic.csv"
            config = SimpleNamespace(subset="test", checkpoint=None, bpe_path=None,
                                     runtime=RuntimeConfig(), points_per_side=3)
            sample = {"__key__": "a", "image": np.zeros((3, 4, 3), np.uint8), "segmentation": target}
            module = SimpleNamespace(test_dataset=lambda: [sample])
            model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
            reference = SimpleNamespace(predict=lambda image: ([proposal(target)], {}))
            with patch.dict("os.environ", {"SLURM_JOB_ID": "test", "SLURM_STEP_ID": "0"}), \
                    patch("sys.argv", ["check", "--config", "unused", "--output", str(output), "--diagnose"]), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.get_device_name", return_value="mock GPU"), \
                    patch.object(check_batch_equivalence, "load_config", return_value=config), \
                    patch.object(check_batch_equivalence, "build_data_module", return_value=module), \
                    patch.object(check_batch_equivalence, "load_scene_model", return_value=(model, None, {})), \
                    patch.object(check_batch_equivalence, "Sam3GridAdapter", return_value=reference), \
                    patch.object(check_batch_equivalence, "BatchedSam3GridAdapter", Adapter), \
                    patch.object(check_batch_equivalence, "preprocessing_diagnostics", return_value=[{"exact": True, "max_absolute_difference": 0}]), \
                    patch.object(check_batch_equivalence, "configure_model"):
                check_batch_equivalence.main()
            evidence = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(evidence["comparison_policy"], "report_only")
            self.assertEqual(len(evidence["stages"]), 6)
            self.assertEqual(evidence["stages"]["tensor_single"]["strict_mismatch_count"], 0)
            self.assertEqual(evidence["stages"]["configured_runtime"]["strict_mismatch_count"], 1)
