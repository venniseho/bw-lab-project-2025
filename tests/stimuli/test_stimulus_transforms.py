"""CPU regression tests against the frozen file-based historical entrypoint."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

from legacy.contour_fragmentation import mask_fragmenter_clean as legacy
from bwlab.stimuli.fence import generate_fence
from bwlab.stimuli.fragmentation import FragmentationConfig
from bwlab.stimuli.transforms import ContourFragmentation, FenceOcclusion


class StimulusTransformTests(unittest.TestCase):
    def setUp(self):
        mask = np.zeros((60, 84), dtype=np.uint8)
        mask[12:48, 22:62] = 255
        self.sample = {"__key__": "synthetic/rectangle", "image": np.full((60, 84, 3), 127, np.uint8),
                       "segmentation": mask, "metadata": {"note": ["keep"]}}

    def test_archive_source_is_frozen(self):
        path = Path(legacy.__file__)
        content = path.read_bytes()
        header = f"blob {len(content)}\0".encode()
        self.assertEqual(hashlib.sha1(header + content).hexdigest(),
                         "4623aa609d7007e8a9157fde9cd141379b366520")

    def test_defaults_equal_historical_entrypoint(self):
        signature = inspect.signature(legacy.fragment_one)
        for key, value in asdict(FragmentationConfig()).items():
            self.assertEqual(signature.parameters[key].default, value)

    def test_pixels_match_historical_entrypoint_under_controlled_rng(self):
        # Fix each entropy RNG call only inside the reference comparison. The
        # production adapter uses no monkey patch and retains seed=None behavior.
        real_default_rng = np.random.default_rng
        configs = [FragmentationConfig(noise_count=12),
                   FragmentationConfig(edge_len=7, noise_count=10, outline_mode="random", jitter_deg=10),
                   FragmentationConfig(edge_len=8, noise_mode="grid", noise_per_cell=2, grid=20),
                   FragmentationConfig(inside_noise_count=8, outside_noise_count=9,
                                       inside_noise_len=6, outside_noise_len=9)]
        for config in configs:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cv2.imwrite(str(root / "rgb.png"), self.sample["image"])
                cv2.imwrite(str(root / "mask.png"), self.sample["segmentation"])
                with patch.object(np.random, "default_rng", side_effect=lambda: real_default_rng(17)), \
                        patch.object(legacy, "compute_and_save_metrics"):
                    legacy.fragment_one(str(root / "rgb.png"), str(root / "mask.png"),
                                        str(root / "reference"), **asdict(config))
                    output = ContourFragmentation(config)(self.sample)
                reference = cv2.imread(str(root / "reference/fragments/rgb_fragmented.png"))
                np.testing.assert_array_equal(output["image"], reference)
                self.assertGreater(len(output["fragmentation"]["noise_segments"]), 0)

    def test_seeded_contours_are_deterministic_and_thread_isolated(self):
        transform = ContourFragmentation(FragmentationConfig(noise_count=15), seed=42)
        rng_factory = np.random.default_rng
        first = transform(self.sample)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(transform, [self.sample] * 4))
        for output in outputs:
            np.testing.assert_array_equal(first["image"], output["image"])
        self.assertIs(np.random.default_rng, rng_factory)
        other = ContourFragmentation(FragmentationConfig(noise_count=15), seed=43)(self.sample)
        self.assertFalse(np.array_equal(first["image"], other["image"]))

    def test_contours_preserve_target_and_remove_texture(self):
        output = ContourFragmentation(FragmentationConfig(noise_count=0), seed=7)(self.sample)
        self.assertEqual(set(np.unique(output["image"])), {0, 255})
        np.testing.assert_array_equal(output["segmentation"], self.sample["segmentation"])
        self.assertNotIn("visible_segmentation", output)
        output["metadata"]["note"].append("new")
        self.assertEqual(self.sample["metadata"]["note"], ["keep"])

    def test_fence_masks_alignment_and_no_input_mutation(self):
        before = self.sample["image"].copy()
        output = FenceOcclusion(4, 10, phase=3)(self.sample)
        expected = np.zeros((60, 84), bool)
        expected[:, [x for x in range(84) if 1 <= x % 10 < 5]] = True
        np.testing.assert_array_equal(output["occlusion_mask"], expected)
        np.testing.assert_array_equal(output["visible_segmentation"], (self.sample["segmentation"] > 0) & ~expected)
        np.testing.assert_array_equal(output["segmentation"], self.sample["segmentation"])
        np.testing.assert_array_equal(self.sample["image"], before)
        self.assertFalse(np.shares_memory(output["segmentation"], self.sample["segmentation"]))

    def test_phase_periodic_and_legacy_seed_equivalent(self):
        a = FenceOcclusion(4, 10, seed=42)(self.sample)
        b = generate_fence(self.sample["image"], 4, 10, 90, 42)
        np.testing.assert_array_equal(a["image"], b.corrupted_rgb)
        phase = np.random.default_rng(42).uniform(0, 10)
        c = FenceOcclusion(4, 10, phase=phase + 10)(self.sample)
        np.testing.assert_array_equal(a["image"], c["image"])

    def test_severities_nested_for_seed_and_phase(self):
        for kwargs in ({"seed": 42}, {"phase": 2.5}):
            for angle in (0, 37, 90):
                masks = [FenceOcclusion(w, 20, angle, **kwargs)(self.sample)["occlusion_mask"] for w in (0, 2, 8, 20)]
                for a, b in zip(masks, masks[1:]):
                    self.assertTrue(np.all(~a | b))

    def test_compose_two_fences_unions_visibility(self):
        output = Compose([FenceOcclusion(3, 12, 90), FenceOcclusion(3, 12, 0)])(self.sample)
        np.testing.assert_array_equal(output["visible_segmentation"],
                                      (output["segmentation"] > 0) & ~output["occlusion_mask"])
        self.assertEqual(len(output["transforms"]), 2)
        self.assertEqual(output["__key__"], self.sample["__key__"])

    def test_empty_full_and_single_pixel_fence(self):
        for size in ((1, 1), (3, 8)):
            for foreground in (0, 255):
                sample = {"__key__": "edge", "image": np.ones((*size, 3), np.uint8),
                          "segmentation": np.full(size, foreground, np.uint8)}
                for width in (0, 10):
                    result = FenceOcclusion(width, 10)(sample)
                    self.assertEqual(result["visible_segmentation"].sum(), size[0] * size[1] if foreground and width == 0 else 0)

    def test_invalid_masks_and_conflicting_phase_rejected(self):
        for mask in (np.zeros((2, 3)), np.full((60, 84), 0.5), np.full((60, 84), np.nan)):
            for transform in (FenceOcclusion(4, 10), ContourFragmentation()):
                with self.assertRaises(ValueError):
                    transform({**self.sample, "segmentation": mask})
        for phase in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FenceOcclusion(4, 10, phase=phase)(self.sample)
        with self.assertRaises(ValueError):
            FenceOcclusion(4, 10, seed=42, phase=2)(self.sample)

    def test_empty_contour_fails_explicitly_as_archive(self):
        with self.assertRaisesRegex(RuntimeError, "no contour"):
            ContourFragmentation()(dict(self.sample, segmentation=np.zeros((60, 84), np.uint8)))

    def test_fence_then_contour_rejected(self):
        with self.assertRaisesRegex(ValueError, "before"):
            Compose([FenceOcclusion(3, 12), ContourFragmentation()])(self.sample)


if __name__ == "__main__":
    unittest.main()
