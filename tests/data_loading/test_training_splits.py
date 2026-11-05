"""Training subsets must remain reproducible and disjoint by source image."""

from copy import deepcopy
import unittest
from core.data_loading.splits import select_splits


class TrainingSplitTests(unittest.TestCase):
    def test_disjoint_reproducible_splits_keep_every_noncrowd_instance(self):
        data = {
            "images": [{"id": i} for i in range(6)],
            "categories": [],
            "annotations": [
                {"id": i * 3 + j, "image_id": i, "iscrowd": int(j == 2)}
                for i in range(6)
                for j in range(3)
            ],
        }
        before = deepcopy(data)
        splits = select_splits(data, 3, 2, 42)
        self.assertEqual(splits, select_splits(data, 3, 2, 42))
        self.assertEqual(data, before)
        ids = [{im["id"] for im in splits[name]["images"]} for name in ("train", "test")]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual([len(splits[name]["annotations"]) for name in ("train", "test")], [6, 4])
        for split in splits.values():
            self.assertTrue(all(a["iscrowd"] == 0 for a in split["annotations"]))
