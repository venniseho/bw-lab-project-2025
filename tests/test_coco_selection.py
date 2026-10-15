"""Synthetic CPU tests; actual COCO API test uses an in-memory dataset only."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import unittest

import numpy as np

from data_selection.coco import (
    load_coco_image_instances, prepare_selected_probe, summarize_selection,
    validate_instance,
)
from evaluation.preparation import OcclusionCondition
from stimuli_generation.fence import generate_fence


def annotation(identifier=1, crowd=0):
    return {"id": identifier, "image_id": 10, "iscrowd": crowd,
            "category_id": 3, "area": 99999, "segmentation": [[0, 0, 1, 0, 1, 1]]}


class CocoSelectionTests(unittest.TestCase):
    def test_one_pixel_object_and_tiny_area_fraction_are_accepted(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[70, 80] = 1
        record = validate_instance(annotation(), mask, mask.shape)
        self.assertTrue(record.accepted)
        self.assertEqual(record.metadata["object_area_pixels"], 1)
        self.assertEqual(record.metadata["object_area_fraction"], 1 / 10000)
        self.assertEqual(record.annotation["area"], 99999)

    def test_concave_object_with_background_centroid_is_accepted(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:6, 1] = mask[1:6, 5] = True
        mask[5, 1:6] = True
        ys, xs = np.nonzero(mask)
        self.assertFalse(mask[int(ys.mean()), int(xs.mean())])
        record = validate_instance(annotation(), mask, mask.shape)
        self.assertTrue(record.accepted)
        probe = prepare_selected_probe(record, [OcclusionCondition(
            "clean", np.zeros_like(mask), {}
        )])
        x, y = probe.prompt_xy
        self.assertTrue(mask[y, x])

    def test_disconnected_regions_and_holes_are_preserved(self):
        disconnected = np.zeros((7, 9), dtype=np.uint8)
        disconnected[1:3, 1:3] = 1
        disconnected[5, 7] = 1
        ring = np.zeros_like(disconnected)
        ring[1:6, 1:8] = 1
        ring[2:5, 2:7] = 0
        for mask in (disconnected, ring):
            record = validate_instance(annotation(), mask, mask.shape)
            self.assertTrue(record.accepted)
            np.testing.assert_array_equal(record.mask, mask)
            probe = prepare_selected_probe(record, [OcclusionCondition(
                "clean", np.zeros_like(mask), {}
            )])
            np.testing.assert_array_equal(probe.targets["clean"].full_object, mask != 0)
            self.assertEqual(record.metadata["object_area_pixels"], np.count_nonzero(mask))

    def test_binary_encodings_and_original_ids_annotation_and_mask_not_modified(self):
        for dtype, scale in ((bool, 1), (np.uint8, 1), (np.uint8, 255)):
            mask = np.array([[0, 1], [1, 0]], dtype=dtype) * scale
            original = mask.copy()
            mask.setflags(write=False)
            ann = annotation(identifier=4321)
            before = json.dumps(ann, sort_keys=True)
            record = validate_instance(ann, mask, mask.shape)
            self.assertTrue(record.accepted)
            self.assertEqual(record.metadata["annotation_id"], 4321)
            self.assertEqual(record.metadata["image_id"], 10)
            self.assertEqual(record.mask.dtype, mask.dtype)
            np.testing.assert_array_equal(record.mask, original)
            record.mask[:] = 0
            record.annotation["segmentation"][0][0] = 999
            np.testing.assert_array_equal(mask, original)
            self.assertEqual(json.dumps(ann, sort_keys=True), before)

    def test_empty_mask_is_retained_with_zero_area_and_reason(self):
        mask = np.zeros((3, 5), dtype=np.uint8)
        record = validate_instance(annotation(), mask, mask.shape)
        self.assertFalse(record.accepted)
        self.assertEqual(record.metadata["exclusion_reason"], "empty_mask")
        self.assertEqual(record.metadata["object_area_pixels"], 0)
        self.assertEqual(record.metadata["object_area_fraction"], 0)
        np.testing.assert_array_equal(record.mask, mask)

    def test_malformed_and_nonbinary_masks_are_explicitly_excluded(self):
        for mask, reason in (
            (None, "malformed_mask"), ([[0, 1]], "malformed_mask"),
            (np.ones((3, 5, 1)), "malformed_mask"),
            (np.full((3, 5), 0.5), "nonbinary_mask"),
            (np.full((3, 5), np.nan), "nonbinary_mask"),
            (np.full((3, 5), -1), "nonbinary_mask"),
            (np.full((3, 5), "foreground"), "nonbinary_mask"),
        ):
            with self.subTest(reason=reason):
                record = validate_instance(annotation(), mask, (3, 5))
                self.assertFalse(record.accepted)
                self.assertEqual(record.metadata["exclusion_reason"], reason)
                self.assertIsNone(record.metadata["object_area_pixels"])

    def test_misalignment_is_not_repaired_by_resizing(self):
        mask = np.ones((2, 3), dtype=np.uint8)
        record = validate_instance(annotation(), mask, (3, 5))
        self.assertFalse(record.accepted)
        self.assertEqual(record.metadata["exclusion_reason"], "mask_shape_mismatch")
        self.assertEqual(record.metadata["object_area_pixels"], 6)
        self.assertIsNone(record.metadata["object_area_fraction"])
        np.testing.assert_array_equal(record.mask, mask)
        self.assertEqual(record.mask.shape, (2, 3))

    def test_crowd_policy_is_explicit_and_opt_in(self):
        mask = np.ones((3, 5), dtype=np.uint8)
        ann = annotation(crowd=1)
        excluded = validate_instance(ann, mask, mask.shape)
        included = validate_instance(ann, mask, mask.shape, crowd_policy="include")
        self.assertFalse(excluded.accepted)
        self.assertEqual(excluded.metadata["exclusion_reason"], "crowd_annotation")
        self.assertEqual(excluded.metadata["iscrowd"], 1)
        self.assertEqual(excluded.metadata["object_area_pixels"], 15)
        np.testing.assert_array_equal(excluded.mask, mask)
        self.assertTrue(included.accepted)
        self.assertEqual(included.metadata["crowd_policy"], "include")
        self.assertEqual(included.annotation, ann)

    def test_unknown_crowd_status_and_bad_ids_are_not_silently_assumed_valid(self):
        mask = np.ones((2, 3), dtype=bool)
        for field, value, reason in (
            ("iscrowd", None, "invalid_iscrowd"), ("iscrowd", 2, "invalid_iscrowd"),
            ("id", None, "invalid_annotation_id"), ("image_id", None, "invalid_image_id"),
        ):
            ann = annotation()
            ann[field] = value
            record = validate_instance(ann, mask, mask.shape)
            self.assertFalse(record.accepted)
            self.assertEqual(record.metadata["exclusion_reason"], reason)
            self.assertEqual(record.annotation[field], value)
        with self.assertRaises(ValueError):
            validate_instance(annotation(), mask, mask.shape, crowd_policy="guess")
        with self.assertRaises(ValueError):
            validate_instance(annotation(), mask, (0, 3))

    def test_statistics_include_empty_source_images_and_all_exclusion_reasons(self):
        masks = [np.ones((3, 5), dtype=bool), np.zeros((3, 5), dtype=bool),
                 np.ones((3, 5), dtype=bool), None]
        records = [validate_instance(annotation(i + 1, crowd=int(i == 2)), mask, (3, 5),
                                     decode_error="ValueError: bad RLE" if i == 3 else None)
                   for i, mask in enumerate(masks)]
        stats = summarize_selection(records, source_image_ids=[10, 11, 12])
        self.assertEqual(stats["source_image_count"], 3)
        self.assertEqual(stats["object_instance_count"], 4)
        self.assertEqual(stats["accepted_count"], 1)
        self.assertEqual(stats["excluded_count"], 3)
        self.assertEqual(stats["crowd_annotation_count"], 1)
        self.assertEqual(stats["exclusion_reasons"], {
            "crowd_annotation": 1, "empty_mask": 1, "mask_decode_error": 1
        })
        self.assertEqual(sum(stats["exclusion_reasons"].values()), stats["excluded_count"])
        self.assertEqual([r["object_area_pixels"] for r in stats["instances"]], [15, 0, 15, None])
        self.assertEqual(json.loads(json.dumps(stats, allow_nan=False)), stats)
        stats["instances"][0]["object_area_pixels"] = 999
        self.assertEqual(records[0].metadata["object_area_pixels"], 15)
        with self.assertRaises(ValueError):
            summarize_selection([records[0], records[0]], source_image_ids=[10])
        with self.assertRaises(ValueError):
            summarize_selection(records, source_image_ids=[11])

    def test_loader_decodes_every_annotation_without_area_or_crowd_prefilter(self):
        class FakeCOCO:
            def getAnnIds(self, **kwargs):
                self.kwargs = kwargs
                return [3, 2, 1]

            def loadAnns(self, identifiers):
                return [annotation(i, crowd=int(i == 2)) for i in identifiers]

            def annToMask(self, ann):
                if ann["id"] == 3:
                    raise ValueError("malformed segmentation")
                return np.ones((3, 5), dtype=np.uint8)

        coco = FakeCOCO()
        records = load_coco_image_instances(coco, 10, (3, 5))
        self.assertEqual(coco.kwargs, {"imgIds": [10]})
        self.assertEqual([r.metadata["annotation_id"] for r in records], [1, 2, 3])
        self.assertEqual([r.metadata["exclusion_reason"] for r in records],
                         [None, "crowd_annotation", "mask_decode_error"])
        self.assertIn("malformed segmentation", records[2].metadata["decode_error"])

    def test_annotation_image_mismatch_retains_source_frame_for_statistics(self):
        ann = annotation()
        ann["image_id"] = 99
        record = validate_instance(ann, np.ones((3, 5), dtype=bool), (3, 5),
                                   expected_image_id=10)
        self.assertEqual(record.metadata["exclusion_reason"], "annotation_image_id_mismatch")
        stats = summarize_selection([record], source_image_ids=[10])
        self.assertEqual(stats["excluded_count"], 1)
        self.assertEqual(stats["instances"][0]["image_id"], 99)
        self.assertEqual(stats["instances"][0]["source_image_id"], 10)

    def test_heavy_fence_unpromptable_object_is_still_accepted_and_retained(self):
        image = np.full((3, 5, 3), 128, dtype=np.uint8)
        mask = np.zeros((3, 5), dtype=np.uint8)
        mask[1, 0] = 1
        record = validate_instance(annotation(123), mask, mask.shape)
        conditions = []
        for name, width in (("clean", 0), ("heavy", 1)):
            params = {"bar_width": width, "bar_spacing": 5, "bar_angle": 90}
            fence = generate_fence(image, **params, seed=None, gt_mask=record.mask)
            conditions.append(OcclusionCondition(name, fence.occlusion_mask, params))
        probe = prepare_selected_probe(record, conditions)
        self.assertTrue(record.accepted)
        self.assertFalse(probe.valid_for_evaluation)
        self.assertIsNone(probe.prompt_xy)
        self.assertEqual(probe.invalid_reason, "no_common_visible_pixel")
        self.assertEqual(len(probe.metadata), 2)
        for row in probe.metadata:
            self.assertTrue(row["accepted"])
            self.assertFalse(row["promptable"])
            self.assertFalse(row["valid_for_evaluation"])
            self.assertEqual(row["annotation_id"], 123)
            self.assertEqual(row["object_id"], 123)
            self.assertEqual(row["invalid_reason"], "no_common_visible_pixel")
        np.testing.assert_array_equal(record.mask, mask)
        stats = summarize_selection([record], source_image_ids=[10])
        self.assertEqual(stats["accepted_count"], 1)
        self.assertEqual(stats["excluded_count"], 0)
        excluded = validate_instance(annotation(2), np.zeros_like(mask), mask.shape)
        with self.assertRaises(ValueError):
            prepare_selected_probe(excluded, conditions)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional pycocotools not installed")
    def test_real_coco_polygon_parts_and_rle_hole_crowd_and_misalignment(self):
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_utils

        hole = np.zeros((7, 9), dtype=np.uint8)
        hole[1:6, 1:8] = 1
        hole[2:5, 2:7] = 0
        anns = [annotation(i) for i in (1, 2, 3, 4)]
        anns[0]["segmentation"] = [[1, 1, 3, 1, 3, 3, 1, 3],
                                    [6, 5, 8, 5, 8, 6, 6, 6]]
        anns[1]["segmentation"] = mask_utils.encode(np.asfortranarray(hole))
        anns[2]["iscrowd"] = 1
        anns[2]["segmentation"] = mask_utils.encode(np.asfortranarray(hole))
        anns[3]["segmentation"] = mask_utils.encode(np.ones((2, 3), dtype=np.uint8, order="F"))
        coco = COCO()
        coco.dataset = {"images": [{"id": 10, "height": 7, "width": 9}],
                        "annotations": anns, "categories": [{"id": 3}]}
        with redirect_stdout(io.StringIO()):
            coco.createIndex()
        records = load_coco_image_instances(coco, 10, (7, 9))
        self.assertEqual([r.accepted for r in records], [True, True, False, False])
        self.assertTrue(records[0].mask[1, 1])
        self.assertTrue(records[0].mask[5, 6])
        self.assertFalse(records[0].mask[4, 4])
        np.testing.assert_array_equal(records[1].mask, hole)
        self.assertEqual(records[2].metadata["exclusion_reason"], "crowd_annotation")
        self.assertEqual(records[3].metadata["exclusion_reason"], "mask_shape_mismatch")
        included = load_coco_image_instances(coco, 10, (7, 9), crowd_policy="include")
        self.assertTrue(included[2].accepted)


if __name__ == "__main__":
    unittest.main()
