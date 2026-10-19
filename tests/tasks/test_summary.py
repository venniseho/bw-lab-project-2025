import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bwlab.models.proposals import Proposal
from bwlab.tasks.summary import SummaryEvaluator
from scripts.replay_ranked_examples import ranked_rows


class SummaryTests(unittest.TestCase):
    def test_empty_model_choice_is_scored_and_oracle_stays_separate_no_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"run"
            evaluator = SummaryEvaluator(root, preview_batches=0)
            target = np.ones((4, 5), bool)
            proposals = [Proposal(np.packbits(mask), mask.shape, {
                "proposal_id": i, "predicted_iou_score": score, "area_pixels": int(mask.sum())})
                for i, (mask, score) in enumerate([(np.zeros_like(target), 0.9), (target, 0.5)])]
            sample = {"__key__": "source/clean", "image": np.zeros((4, 5, 3), np.uint8),
                      "segmentation": target, "condition": "clean", "batch_index": 0}
            evaluator.update(sample, (proposals, {}))
            evaluator.finalize()
            with (root/"image_results.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["full_iou"]), 0)
            self.assertEqual(float(row["oracle_full_iou"]), 1)
            self.assertEqual(row["chosen_proposal_id"], "0")
            self.assertFalse((root/"previews").exists())
            self.assertEqual(ranked_rows(root/"image_results.csv", "clean", 10)["worst"][0]["__key__"], "source/clean")

    def test_best_worst_filter_by_condition_and_measured_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"results.csv"
            path.write_text("input_index,__key__,condition,status,full_iou,predicted_iou_score\n"
                            "0,a,clean,complete,0.1,0.99\n1,b,clean,complete,0.9,0.1\n"
                            "2,c,heavy,complete,1,1\n3,d,clean,empty_target,,1\n")
            groups = ranked_rows(path, "clean", 1)
            self.assertEqual(groups["best"][0]["__key__"], "b")
            self.assertEqual(groups["worst"][0]["__key__"], "a")
