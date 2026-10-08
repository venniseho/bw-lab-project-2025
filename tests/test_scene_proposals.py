import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
import os
import sys

import numpy as np
from scene_segmentation.proposals import (point_grid, collect_proposals, select_proposals,
                                         save_scene, packed_mask_iou)
from scripts.sam3_scene_masks import checked_state, main


class FakePredictor:
    def __init__(self):
        self.points = []
        self.embeddings = 0

    def set_image(self, rgb):
        self.embeddings += 1
        self.shape = rgb.shape[:2]

    def predict(self, **kwargs):
        assert kwargs['multimask_output'] and kwargs['normalize_coords']
        assert not kwargs['return_logits']
        assert kwargs['point_coords'].shape == (1, 2)
        np.testing.assert_array_equal(kwargs['point_labels'], [1])
        self.points.append(kwargs['point_coords'].copy())
        masks = np.zeros((3, *self.shape), dtype=np.float32)
        masks[0, 0, 0] = 1
        masks[1, -1, -1] = 1
        return masks, np.array([0.9, 0.2, 0.1]), None


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.rgb[..., 0] = 200
        self.predictor = FakePredictor()
        self.proposals, self.timing = collect_proposals(self.predictor, self.rgb, 2)

    def test_grid(self):
        grid = point_grid(4, 6, 2)
        np.testing.assert_allclose(grid, [[1.25, .75], [3.75, .75], [1.25, 2.25], [3.75, 2.25]])
        np.testing.assert_array_equal(point_grid(1, 1, 1), [[0, 0]])
        with self.assertRaises(ValueError):
            point_grid(2, 2, 0)

    def test_independent_queries_and_all_candidates(self):
        self.assertEqual(self.predictor.embeddings, 1)
        self.assertEqual(len(self.predictor.points), 4)
        self.assertEqual(len(self.proposals), 12)
        self.assertEqual(self.timing['raw_proposal_count'], 12)
        self.assertEqual(self.proposals[2].metadata['area_pixels'], 0)
        self.assertEqual(self.rgb[0, 0].tolist(), [200, 0, 0])

    def test_dedup_and_limit_ledger(self):
        retained, rows, stats = select_proposals(self.proposals, proposal_limit=1)
        self.assertEqual([p.metadata['proposal_id'] for p in retained], [0])
        self.assertEqual(sum(r['retention_status'] == 'duplicate' for r in rows), 6)
        self.assertEqual(sum(r['retention_status'] == 'empty_mask' for r in rows), 4)
        self.assertEqual(sum(r['retention_status'] == 'proposal_limit' for r in rows), 1)
        self.assertEqual(packed_mask_iou(self.proposals[0], self.proposals[3]), 1)
        self.assertEqual(packed_mask_iou(self.proposals[0], self.proposals[1]), 0)

    def test_no_dedup_and_score_filter(self):
        retained, rows, _ = select_proposals(self.proposals, dedup_iou=None, min_score=.5)
        self.assertEqual(len(retained), 4)
        self.assertEqual(sum(r['retention_status'] == 'score_filter' for r in rows), 4)

    def test_archive_rgb_and_contact_sheet(self):
        from PIL import Image
        retained, rows, stats = select_proposals(self.proposals)
        before = self.rgb.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'scene'
            report = save_scene(self.rgb, self.proposals, retained, rows, root, stats)
            self.assertEqual(len(list((root / 'candidates').glob('*.png'))), 12)
            self.assertTrue((root / 'contact_sheet.png').is_file())
            np.testing.assert_array_equal(np.asarray(Image.open(root / 'original.png')), before)
            mask = np.asarray(Image.open(root / report['proposals'][0]['mask_file']))
            np.testing.assert_array_equal(mask > 0, self.proposals[0].mask())
            self.assertEqual(json.loads((root / 'metadata.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError):
                save_scene(self.rgb, self.proposals, retained, rows, root, stats)
        np.testing.assert_array_equal(self.rgb, before)

    def test_archive_cap_and_empty_visualization(self):
        retained, rows, stats = select_proposals(self.proposals)
        with tempfile.TemporaryDirectory() as directory:
            report = save_scene(self.rgb, self.proposals, retained, rows, Path(directory) / 'cap', stats, 0)
            self.assertEqual(report['raw_masks_saved_count'], 0)
            self.assertEqual(report['retained_masks_saved_outside_archive_limit'], 2)
            retained, rows, stats = select_proposals(self.proposals, proposal_limit=0)
            save_scene(self.rgb, self.proposals, retained, rows, Path(directory) / 'empty', stats)

    def test_malformed_predictions_fail(self):
        predictor = Mock()
        predictor.predict.return_value = (np.zeros((1, 2, 2)), np.array([.1]), None)
        with self.assertRaises(ValueError):
            collect_proposals(predictor, self.rgb, 1)
        predictor.predict.return_value = (np.zeros((1, 4, 6)), np.array([np.nan]), None)
        with self.assertRaises(ValueError):
            collect_proposals(predictor, self.rgb, 1)

    def test_checked_native_checkpoint(self):
        model = Mock()
        model.load_state_dict.return_value = Mock(missing_keys=[], unexpected_keys=['extra'])
        self.assertEqual(checked_state(model, {'model': {'detector.a': 1, 'tracker.b': 2}}), ['extra'])
        model.load_state_dict.assert_called_once_with({'a': 1, 'inst_interactive_predictor.model.b': 2}, strict=False)
        model.load_state_dict.return_value = Mock(missing_keys=['missing'], unexpected_keys=[])
        with self.assertRaises(RuntimeError):
            checked_state(model, {'detector.a': 1})
        with self.assertRaises(ValueError):
            checked_state(model, {'hf_weights': 1})

    def test_cli_seven_images_mocked(self):
        from PIL import Image
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = 'mock GPU'
        torch.cuda.max_memory_allocated.return_value = 100
        torch.cuda.max_memory_reserved.return_value = 200
        torch.__version__ = 'mock'
        torch.inference_mode.side_effect = nullcontext
        predictor = FakePredictor()
        predictor.reset_predictor = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            for index in range(7):
                Image.fromarray(self.rgb).save(images / f'{index}.png')
            checkpoint = root / 'sam3.pt'
            checkpoint.write_bytes(b'mock checkpoint, not real weights')
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'mock', 'SLURM_STEP_ID': '0'}), \
                 patch.dict(sys.modules, {'torch': torch}), \
                 patch('scripts.sam3_scene_masks.load_predictor', return_value=(predictor, {'api': 'mock'})):
                main(['--image-dir', str(images), '--limit-images', '7', '--points-per-side', '1',
                      '--checkpoint', str(checkpoint), '--output-dir', str(root / 'output')])
            manifest = json.loads((root / 'output/run.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['completed_images']), 7)
            self.assertEqual(predictor.embeddings, 7)
            self.assertEqual(len(list((root / 'output').glob('*/contact_sheet.png'))), 7)

    def test_cli_login_node_rejected_before_gpu_import(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'rgb.png'
            Image.fromarray(self.rgb).save(image)
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
                main(['--image', str(image)])


if __name__ == '__main__':
    unittest.main()
