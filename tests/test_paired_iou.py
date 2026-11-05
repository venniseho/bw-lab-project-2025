from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.plot_paired_iou import pair_scores, read_scores


class PairedIoUTests(unittest.TestCase):
    def test_pairs_use_source_ids_instead_of_row_order_and_keep_zero(self):
        sources, clean, transformed = pair_scores({"a": 0, "b": 0.9, "c": 0.1}, {"b": 0.2, "a": 0})
        self.assertEqual(sources, ["a", "b"])
        np.testing.assert_array_equal(clean, [0, 0.9])
        np.testing.assert_array_equal(transformed, [0, 0.2])

    def test_duplicate_source_condition_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "source_key,condition,status,full_iou\na,clean,complete,0\na,clean,complete,0.5\n"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_scores(path)

    def test_nonfinite_and_incomplete_scores_are_not_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "source_key,condition,status,full_iou\na,clean,complete,0\nb,clean,empty_target,\nc,clean,complete,nan\n"
            )
            self.assertEqual(read_scores(path), {"clean": {"a": 0}})
