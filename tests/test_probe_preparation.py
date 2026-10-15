"""CPU-only tests without datasets, checkpoints, or model imports."""

import json
import unittest

import numpy as np

from evaluation.preparation import (
    OcclusionCondition,
    build_evaluation_targets,
    prepare_probe,
)
from stimuli_generation.fence import generate_fence


class ProbePreparationTests(unittest.TestCase):
    def setUp(self):
        self.object = np.zeros((5, 9), dtype=bool)
        self.object[1:4, 1:8] = True
        self.conditions = []
        for name, columns in (
            ("clean", []), ("light", [4]), ("medium", [3, 4]),
            ("heavy", [3, 4, 5]),
        ):
            mask = np.zeros_like(self.object)
            mask[:, columns] = True
            self.conditions.append(OcclusionCondition(
                name, mask, {"bar_width": len(columns), "bar_spacing": 9,
                             "bar_angle": 90}, seed=42,
            ))

    def prepare(self, **kwargs):
        return prepare_probe(self.object, self.conditions,
                             image_id="image-7", object_id=11, **kwargs)

    def test_full_and_visible_targets_with_binary_encodings(self):
        original = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)
        hidden = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        expected_visible = np.array([[0, 1, 0], [1, 0, 0]], dtype=bool)
        for encoding in (bool, np.uint8):
            for scale in (1, 255) if encoding == np.uint8 else (1,):
                targets = build_evaluation_targets(
                    original.astype(encoding) * scale,
                    hidden.astype(encoding) * scale,
                )
                np.testing.assert_array_equal(targets.full_object, original)
                np.testing.assert_array_equal(targets.visible_object, expected_visible)
                self.assertEqual(targets.full_object.dtype, np.dtype(bool))
                self.assertEqual(targets.visible_object.dtype, np.dtype(bool))

    def test_shared_point_inside_object_and_visible_in_every_condition(self):
        probe = self.prepare()
        # Original centroid is (4, 2); x=2 and x=6 tie among common pixels.
        # Row-major tie-breaking chooses (2, 2), in (x, y) order.
        self.assertEqual(probe.prompt_xy, (2, 2))
        self.assertTrue(probe.valid_for_evaluation)
        self.assertIsNone(probe.invalid_reason)
        x, y = probe.prompt_xy
        self.assertTrue(self.object[y, x])
        for condition, row in zip(self.conditions, probe.metadata):
            self.assertFalse(condition.occlusion_mask[y, x])
            self.assertTrue(probe.targets[condition.name].visible_object[y, x])
            np.testing.assert_array_equal(
                probe.targets[condition.name].full_object, self.object
            )
            self.assertEqual(row["prompt_xy"], [x, y])
            self.assertEqual(row["prompt_label"], 1)

    def test_nonnested_conditions_use_intersection_of_all_visible_pixels(self):
        original = np.ones((1, 3), dtype=bool)
        conditions = [
            OcclusionCondition("a", np.array([[0, 1, 0]], dtype=bool), {}),
            OcclusionCondition("b", np.array([[1, 0, 0]], dtype=bool), {}),
        ]
        probe = prepare_probe(original, conditions, image_id=1)
        self.assertEqual(probe.prompt_xy, (2, 0))
        # Last condition alone would select (1, 0), hidden by condition a.
        reversed_probe = prepare_probe(original, conditions[::-1], image_id=1)
        self.assertEqual(reversed_probe.prompt_xy, probe.prompt_xy)

    def test_disconnected_object_with_centroid_outside_selects_foreground(self):
        original = np.array([[1, 0, 0, 0, 1]], dtype=bool)
        probe = prepare_probe(original, [OcclusionCondition(
            "clean", np.zeros_like(original), {}
        )], image_id=1)
        self.assertEqual(probe.prompt_xy, (0, 0))

    def test_metadata_records_actual_percent_and_provenance(self):
        probe = self.prepare()
        for i, row in enumerate(probe.metadata):
            self.assertEqual(row["image_id"], "image-7")
            self.assertEqual(row["object_id"], 11)
            self.assertEqual(row["seed"], 42)
            self.assertEqual(row["corruption_parameters"],
                             self.conditions[i].corruption_parameters)
            self.assertEqual(row["prompt_condition_names"],
                             [c.name for c in self.conditions])
            self.assertTrue(row["valid_for_evaluation"])
            self.assertAlmostEqual(row["object_occluded_percent"],
                                   [0, 100 / 7, 200 / 7, 300 / 7][i])
        self.assertEqual(json.loads(json.dumps(probe.metadata)), probe.metadata)
        # Metadata ownership is independent of caller-owned parameter data.
        probe.metadata[0]["corruption_parameters"]["bar_width"] = 99
        self.assertEqual(self.conditions[0].corruption_parameters["bar_width"], 0)

    def test_deterministic_results_and_global_random_state_unchanged(self):
        before = np.random.get_state()
        first, second = self.prepare(), self.prepare()
        after = np.random.get_state()
        self.assertEqual(first.prompt_xy, second.prompt_xy)
        self.assertEqual(first.metadata, second.metadata)
        for name in first.targets:
            for a, b in zip(first.targets[name], second.targets[name]):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_empty_object_is_invalid_with_undefined_coverage(self):
        original = np.zeros_like(self.object)
        probe = prepare_probe(original, self.conditions, image_id=1)
        self.assertFalse(probe.valid_for_evaluation)
        self.assertIsNone(probe.prompt_xy)
        self.assertEqual(probe.invalid_reason, "empty_object")
        for row, targets in zip(probe.metadata, probe.targets.values()):
            self.assertIsNone(row["object_occluded_percent"])
            self.assertIsNone(row["object_id"])
            self.assertIsNone(row["prompt_xy"])
            self.assertFalse(row["valid_for_evaluation"])
            self.assertFalse(np.any(targets.full_object))
            self.assertFalse(np.any(targets.visible_object))

    def test_fully_occluded_condition_invalidates_entire_paired_example(self):
        conditions = [self.conditions[0], OcclusionCondition(
            "covered", np.ones_like(self.object), {}, seed=None
        )]
        probe = prepare_probe(self.object, conditions, image_id=1)
        self.assertFalse(probe.valid_for_evaluation)
        self.assertIsNone(probe.prompt_xy)
        self.assertEqual(probe.invalid_reason, "no_common_visible_pixel")
        self.assertEqual(probe.metadata[1]["object_occluded_percent"], 100)
        self.assertIsNone(probe.metadata[1]["seed"])
        self.assertTrue(np.any(probe.targets["clean"].visible_object))
        self.assertFalse(np.any(probe.targets["covered"].visible_object))
        self.assertTrue(all(not r["valid_for_evaluation"] for r in probe.metadata))

    def test_no_common_point_even_when_each_condition_has_visible_pixels(self):
        original = np.ones((1, 2), dtype=bool)
        conditions = [
            OcclusionCondition("a", np.array([[1, 0]], dtype=bool), {}),
            OcclusionCondition("b", np.array([[0, 1]], dtype=bool), {}),
        ]
        probe = prepare_probe(original, conditions, image_id=1)
        self.assertIsNone(probe.prompt_xy)
        self.assertFalse(probe.valid_for_evaluation)
        self.assertEqual(probe.invalid_reason, "no_common_visible_pixel")
        self.assertTrue(all(np.any(t.visible_object) for t in probe.targets.values()))

    def test_inputs_preserved_and_outputs_independent(self):
        self.object.setflags(write=False)
        for c in self.conditions:
            c.occlusion_mask.setflags(write=False)
        original_before = self.object.copy()
        masks_before = [c.occlusion_mask.copy() for c in self.conditions]
        params_before = [dict(c.corruption_parameters) for c in self.conditions]
        probe = self.prepare()
        np.testing.assert_array_equal(self.object, original_before)
        for c, before, params in zip(self.conditions, masks_before, params_before):
            np.testing.assert_array_equal(c.occlusion_mask, before)
            self.assertEqual(c.corruption_parameters, params)
        probe.targets["clean"].full_object[:] = False
        probe.targets["clean"].visible_object[:] = False
        np.testing.assert_array_equal(self.object, original_before)
        np.testing.assert_array_equal(probe.targets["heavy"].full_object, original_before)

    def test_integration_with_existing_fence_generator(self):
        image = np.full((13, 20, 3), 128, dtype=np.uint8)
        original = np.zeros(image.shape[:2], dtype=np.uint8)
        original[2:11, 2:18] = 255
        conditions, results = [], []
        for name, width in zip(("clean", "light", "medium", "heavy"), (0, 1, 3, 5)):
            params = {"bar_width": width, "bar_spacing": 8, "bar_angle": 37}
            fence = generate_fence(image, **params, seed=42, gt_mask=original)
            conditions.append(OcclusionCondition(name, fence.occlusion_mask, params, 42))
            results.append(fence)
        probe = prepare_probe(original, conditions, image_id="synthetic", object_id=1)
        self.assertTrue(probe.valid_for_evaluation)
        x, y = probe.prompt_xy
        for fence, row in zip(results, probe.metadata):
            self.assertFalse(fence.occlusion_mask[y, x])
            self.assertAlmostEqual(row["object_occluded_percent"],
                                   100 * fence.object_occluded_fraction)

    def test_invalid_masks_alignment_and_condition_sets(self):
        for bad in (np.array([[0, 0.5]]), np.array([[np.nan]]),
                    np.array([[-1]]), np.array([[1, 255]]),
                    np.zeros((0, 2)), np.ones((2, 2, 1))):
            with self.assertRaises(ValueError):
                build_evaluation_targets(bad, bad)
        with self.assertRaises(ValueError):
            build_evaluation_targets(self.object, np.zeros((2, 3), dtype=bool))
        with self.assertRaises(ValueError):
            prepare_probe(self.object, [], image_id=1)
        with self.assertRaises(ValueError):
            prepare_probe(self.object, [self.conditions[0]] * 2, image_id=1)
        with self.assertRaises(ValueError):
            prepare_probe(self.object, [OcclusionCondition(
                "bad", np.zeros((2, 3), dtype=bool), {}
            )], image_id=1)


if __name__ == "__main__":
    unittest.main()
