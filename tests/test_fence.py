"""Self-contained CPU tests; no datasets, models, or pytest dependency.

Run: python -B -m unittest discover -s tests -p test_fence.py -v
"""

import unittest

import numpy as np

from stimuli_generation.fence import generate_fence


class FenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((13, 20, 3), (20, 100, 200), dtype=np.uint8)

    def test_shape_dtype_and_inputs_preserved(self):
        gt = np.zeros(self.image.shape[:2], dtype=np.uint8)
        gt[2:9, 3:15] = 255
        image_before, gt_before = self.image.copy(), gt.copy()
        result = generate_fence(self.image, 3, 10, 37, 42, gt)
        self.assertEqual(result.corrupted_rgb.shape, self.image.shape)
        self.assertEqual(result.corrupted_rgb.dtype, self.image.dtype)
        self.assertEqual(result.occlusion_mask.shape, gt.shape)
        self.assertEqual(result.occlusion_mask.dtype, np.dtype(bool))
        np.testing.assert_array_equal(self.image, image_before)
        np.testing.assert_array_equal(gt, gt_before)
        np.testing.assert_array_equal(result.corrupted_rgb[~result.occlusion_mask], self.image[~result.occlusion_mask])
        self.assertTrue(np.all(result.corrupted_rgb[result.occlusion_mask] == 0))
        self.assertFalse(np.shares_memory(result.corrupted_rgb, self.image))

    def test_known_vertical_mask_and_coverage(self):
        result = generate_fence(self.image, 3, 10, 90)
        # Centers at columns 0 and 10; half-open width-3 bars cover residues
        # 9, 0, 1. This includes clipping at both image boundaries.
        expected = np.zeros((13, 20), dtype=bool)
        expected[:, [0, 1, 9, 10, 11, 19]] = True
        np.testing.assert_array_equal(result.occlusion_mask, expected)
        self.assertAlmostEqual(result.image_occluded_fraction, 6 / 20)
        self.assertIsNone(result.object_occluded_fraction)

    def test_horizontal_mask(self):
        result = generate_fence(self.image, 3, 10, 0)
        expected = np.zeros((13, 20), dtype=bool)
        expected[[0, 1, 9, 10, 11], :] = True
        np.testing.assert_array_equal(result.occlusion_mask, expected)

    def test_oblique_mask_against_pixel_center_geometry(self):
        angle, spacing, width, seed = 37, 9, 4, 8
        phase = np.random.default_rng(seed).uniform(0, spacing)
        normal = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
        expected = np.zeros(self.image.shape[:2], dtype=bool)
        # Independent scalar oracle: compare each projected pixel center
        # with nearby bar centers, without the implementation's modulo rule.
        for y in range(expected.shape[0]):
            for x in range(expected.shape[1]):
                position = np.dot([x, y], normal)
                expected[y, x] = any(
                    -width / 2 <= position - (phase + k * spacing) < width / 2
                    for k in range(-5, 6)
                )
        result = generate_fence(self.image, width, spacing, angle, seed)
        np.testing.assert_array_equal(result.occlusion_mask, expected)

    def test_actual_object_coverage_and_binary_encodings(self):
        gt = np.zeros(self.image.shape[:2], dtype=bool)
        gt[2:6, 0:5] = True
        # Eight of the twenty foreground pixels are in columns 0 and 1.
        for mask in (gt, gt.astype(np.uint8), gt.astype(np.uint8) * 255):
            result = generate_fence(self.image, 3, 10, gt_mask=mask)
            self.assertAlmostEqual(result.object_occluded_fraction, 8 / 20)
        self.assertIsNone(generate_fence(self.image, 3, 10, gt_mask=np.zeros_like(gt)).object_occluded_fraction)

    def test_deterministic_generation_and_local_rng(self):
        for seed in (None, 0, 42):
            a = generate_fence(self.image, 3, 10, 37, seed)
            b = generate_fence(self.image, 3, 10, 37, seed)
            np.testing.assert_array_equal(a.corrupted_rgb, b.corrupted_rgb)
            np.testing.assert_array_equal(a.occlusion_mask, b.occlusion_mask)
        np.random.seed(123)
        before = np.random.get_state()
        generate_fence(self.image, 3, 10, 37, 42)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertFalse(np.array_equal(
            generate_fence(self.image, 3, 10, 37, 1).occlusion_mask,
            generate_fence(self.image, 3, 10, 37, 2).occlusion_mask,
        ))

    def test_increasing_width_keeps_centers_and_nested_masks(self):
        for angle in (0, 37, 90, 135):
            for seed in (None, 42):
                results = [generate_fence(self.image, w, 10, angle, seed) for w in (1, 3, 7)]
                for light, heavy in zip(results, results[1:]):
                    self.assertFalse(np.any(light.occlusion_mask & ~heavy.occlusion_mask))
                    self.assertLess(light.image_occluded_fraction, heavy.image_occluded_fraction)

    def test_zero_full_occlusion_and_float_images(self):
        image = self.image.astype(np.float32) / 255
        clear = generate_fence(image, 0, 10)
        covered = generate_fence(image, 10, 10)
        np.testing.assert_array_equal(clear.corrupted_rgb, image)
        self.assertEqual(clear.image_occluded_fraction, 0.0)
        self.assertEqual(covered.image_occluded_fraction, 1.0)
        self.assertEqual(covered.corrupted_rgb.dtype, image.dtype)
        self.assertTrue(np.all(covered.corrupted_rgb == 0))

    def test_angle_periodicity(self):
        a = generate_fence(self.image, 3, 10, 37, 42)
        b = generate_fence(self.image, 3, 10, 217, 42)
        np.testing.assert_array_equal(a.occlusion_mask, b.occlusion_mask)

    def test_invalid_geometry_and_mask_alignment(self):
        for width, spacing, angle in ((-1, 10, 90), (11, 10, 90), (1, 0, 90),
                                      (1, 10, np.nan), (1, np.inf, 90)):
            with self.assertRaises(ValueError):
                generate_fence(self.image, width, spacing, angle)
        with self.assertRaises(ValueError):
            generate_fence(self.image, 3, 10, gt_mask=np.ones((2, 3)))
        with self.assertRaises(ValueError):
            generate_fence(self.image[:, :, 0], 3, 10)
        with self.assertRaises(ValueError):
            generate_fence(self.image, 3, 10, seed=-1)


if __name__ == "__main__":
    unittest.main()
