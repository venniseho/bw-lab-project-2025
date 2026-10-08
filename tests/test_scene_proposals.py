import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
from types import SimpleNamespace
import os
import sys

import numpy as np
from scene_segmentation.proposals import (point_grid, collect_proposals, select_proposals,
                                         save_scene, packed_mask_iou)
from scripts.sam3_scene_masks import checked_state, load_scene_model, validate_scene_api, main


class BackboneFreeTracker:
    backbone = None

    def forward_image(self, image):
        # Reproduce the integrated Meta tracker's failure on the old path.
        return self.backbone.forward_image(image)


class FakeIntegratedPredictor:
    def __init__(self):
        self.model = BackboneFreeTracker()
        self._transforms = SimpleNamespace(mask_threshold=0.0, max_hole_area=256.0,
                                           max_sprinkle_area=0.0)
        self.mask_threshold = 0.0
        self.set_image_calls = 0

    def set_image(self, rgb):
        self.set_image_calls += 1
        return self.model.forward_image(rgb)

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None,
                multimask_output=True, return_logits=False, normalize_coords=True):
        raise AssertionError('FakeSceneModel supplies synthetic masks without a standalone predictor')


class FakeProcessor:
    def __init__(self, model):
        self.model = model
        self.embeddings = 0
        self.states = []

    def set_image(self, image):
        from PIL import Image
        assert isinstance(image, Image.Image) and image.mode == 'RGB'
        self.embeddings += 1
        state = {'original_height': image.height, 'original_width': image.width,
                 'backbone_out': {'sam2_backbone_out': object()}}
        self.states.append(state)
        return state


class FakeSceneModel:
    def __init__(self):
        self.points = []
        self.states = []
        self.inst_interactive_predictor = FakeIntegratedPredictor()
        self.rope_buffer = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
        self.to_calls = []

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        if 'dtype' in kwargs:
            raise AssertionError('A blanket dtype cast destroys the complex buffer')
        return self

    def eval(self):
        return self

    def predict_inst(self, inference_state, **kwargs):
        self.states.append(inference_state)
        assert 'sam2_backbone_out' in inference_state['backbone_out']
        assert kwargs['multimask_output'] and kwargs['normalize_coords']
        assert not kwargs['return_logits']
        assert kwargs['point_coords'].shape == (1, 2)
        np.testing.assert_array_equal(kwargs['point_labels'], [1])
        self.points.append(kwargs['point_coords'].copy())
        shape = inference_state['original_height'], inference_state['original_width']
        masks = np.zeros((3, *shape), dtype=np.float32)
        masks[0, 0, 0] = 1
        masks[1, -1, -1] = 1
        return masks, np.array([0.9, 0.2, 0.1]), None


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.rgb[..., 0] = 200
        self.model = FakeSceneModel()
        self.processor = FakeProcessor(self.model)
        self.proposals, self.timing = collect_proposals(self.model, self.processor, self.rgb, 2)

    def test_grid(self):
        grid = point_grid(4, 6, 2)
        np.testing.assert_allclose(grid, [[1.25, .75], [3.75, .75], [1.25, 2.25], [3.75, 2.25]])
        np.testing.assert_array_equal(point_grid(1, 1, 1), [[0, 0]])
        with self.assertRaises(ValueError):
            point_grid(2, 2, 0)

    def test_independent_queries_and_all_candidates(self):
        self.assertEqual(self.processor.embeddings, 1)
        self.assertEqual(len(self.model.points), 4)
        self.assertTrue(all(state is self.processor.states[0] for state in self.model.states))
        self.assertEqual(self.model.inst_interactive_predictor.set_image_calls, 0)
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
            self.assertEqual(len(list((root / 'candidates').glob('proposal_?????.png'))), 12)
            self.assertEqual(len(list((root / 'candidates').glob('*_overlay.png'))), 12)
            self.assertEqual(len(list((root / 'retained').glob('*.png'))), 2)
            self.assertTrue((root / 'raw_all_proposals_overlay.png').is_file())
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
        model = Mock()
        model.predict_inst.return_value = (np.zeros((1, 2, 2)), np.array([.1]), None)
        with self.assertRaises(ValueError):
            collect_proposals(model, self.processor, self.rgb, 1)
        model.predict_inst.return_value = (np.zeros((1, 4, 6)), np.array([np.nan]), None)
        with self.assertRaises(ValueError):
            collect_proposals(model, self.processor, self.rgb, 1)

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
        model = FakeSceneModel()
        processor = FakeProcessor(model)
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
                 patch('scripts.sam3_scene_masks.load_scene_model', return_value=(model, processor, {'api': 'mock'})):
                main(['--image-dir', str(images), '--limit-images', '7',
                      '--checkpoint', str(checkpoint), '--output-dir', str(root / 'output')])
            manifest = json.loads((root / 'output/run.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['completed_images']), 7)
            self.assertEqual(processor.embeddings, 7)
            self.assertEqual(len(model.points), 7 * 9)
            for image_index, state in enumerate(processor.states):
                self.assertTrue(all(query_state is state for query_state in model.states[image_index * 9:(image_index + 1) * 9]))
            first = json.loads((root / 'output/00_0/metadata.json').read_text())
            self.assertEqual(first['raw_proposal_count'], 27)
            self.assertEqual(first['raw_masks_saved_count'], 27)
            self.assertEqual(len(list((root / 'output').glob('*/contact_sheet.png'))), 7)

    def test_cli_login_node_rejected_before_gpu_import(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'rgb.png'
            Image.fromarray(self.rgb).save(image)
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
                main(['--image', str(image)])

    def test_three_by_three_grid_multimask(self):
        model = FakeSceneModel()
        processor = FakeProcessor(model)
        proposals, stats = collect_proposals(model, processor, self.rgb, 3)
        self.assertEqual(stats['grid_point_count'], 9)
        self.assertEqual(len(proposals), 27)
        self.assertEqual(len(model.points), 9)

    def test_backbone_free_tracker_regression(self):
        with self.assertRaisesRegex(AttributeError, "NoneType.*forward_image"):
            self.model.inst_interactive_predictor.set_image(self.rgb)
        # The same model succeeds with processor state, without invoking that path.
        before = self.model.inst_interactive_predictor.set_image_calls
        proposals, _ = collect_proposals(self.model, self.processor, self.rgb, 3)
        self.assertEqual(len(proposals), 27)
        self.assertEqual(self.model.inst_interactive_predictor.set_image_calls, before)

    def test_installed_api_and_cleanup_configuration(self):
        validate_scene_api(self.model, self.processor)
        transforms = self.model.inst_interactive_predictor._transforms
        self.assertEqual(transforms.max_hole_area, 0)
        self.assertEqual(transforms.max_sprinkle_area, 0)
        self.model.predict_inst = None
        with self.assertRaisesRegex(RuntimeError, 'required processor/predict_inst'):
            validate_scene_api(self.model, self.processor)

    def test_bad_installed_signature_fails_explicitly(self):
        self.model.inst_interactive_predictor.predict = lambda point_coords: None
        with self.assertRaisesRegex(RuntimeError, 'incompatible signatures'):
            validate_scene_api(self.model, self.processor)

    def test_missing_or_misaligned_processor_features_fail(self):
        for state in ({}, {'original_height': 4, 'original_width': 7,
                           'backbone_out': {'sam2_backbone_out': object()}},
                      {'original_height': 4, 'original_width': 6, 'backbone_out': {}}):
            processor = Mock()
            processor.set_image.return_value = state
            with self.assertRaisesRegex(RuntimeError, 'interactive image features'):
                collect_proposals(self.model, processor, self.rgb, 3)

    def test_native_loader_preserves_complex_buffer_and_existing_predictor(self):
        model = FakeSceneModel()
        before = model.rope_buffer.copy()
        torch = Mock()
        torch.load.return_value = {'model': {'detector.a': 1, 'tracker.b': 2}}
        model.load_state_dict = Mock(return_value=SimpleNamespace(missing_keys=[], unexpected_keys=[]))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.py'
            source.write_text('# mocked installed source')
            checkpoint = Path(directory) / 'sam3.pt'
            checkpoint.write_bytes(b'mocked checkpoint')
            bpe = Path(directory) / 'bpe.gz'
            bpe.write_bytes(b'mocked tokenizer')
            native = Mock(__file__=str(source))
            builder = Mock(__file__=str(source))
            builder.build_sam3_image_model.return_value = model
            visual = Mock(__file__=str(source))
            processing = Mock(__file__=str(source))
            processing.Sam3Processor.side_effect = lambda loaded, device: FakeProcessor(loaded)
            image_module = Mock(__file__=str(source))
            native.model.model_builder = builder
            native.model.sam1_task_predictor = visual
            native.model.sam3_image_processor = processing
            native.model.sam3_image = image_module
            # Explicit module mapping keeps the test independent of any real SAM3 install.
            modules = {'sam3': native, 'sam3.model': native.model,
                       'sam3.model_builder': builder, 'sam3.model.sam1_task_predictor': visual,
                       'sam3.model.sam3_image_processor': processing, 'sam3.model.sam3_image': image_module}
            native.model_builder = builder
            integrated = model.inst_interactive_predictor
            with patch.dict(sys.modules, modules):
                loaded, processor, info = load_scene_model(checkpoint, bpe, torch)
            self.assertIs(loaded, model)
            self.assertIs(processor.model, model)
            self.assertIs(model.inst_interactive_predictor, integrated)
            self.assertEqual(model.to_calls, [{'device': 'cuda'}])
            np.testing.assert_array_equal(model.rope_buffer, before)
            self.assertTrue(np.iscomplexobj(model.rope_buffer))
            visual.SAM3InteractiveImagePredictor.assert_not_called()
            builder.build_sam3_image_model.assert_called_once_with(
                device='cpu', bpe_path=str(bpe), checkpoint_path=None,
                load_from_HF=False, enable_inst_interactivity=True, compile=False)
            self.assertIn('Sam3Image.predict_inst', info['api'])
            self.assertEqual(info['verified_api_revision'], '0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc')


if __name__ == '__main__':
    unittest.main()
