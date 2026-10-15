"""Shared CPU-only native SAM3 fakes; never used by production inference."""
from types import SimpleNamespace
import numpy as np


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


