"""Model boundary: adapters receive image pixels, never evaluation targets."""
from dataclasses import dataclass

import numpy as np

from scene_segmentation.proposals import collect_proposals


@dataclass
class Sam3GridAdapter:
    """Reuse the verified native processor/predict_inst path on loaded models.

    Caller owns inference_mode, SLURM checks and model loading, as in the
    existing sam3_scene_masks CLI. This class does not submit GPU jobs.
    """

    model: object
    processor: object
    points_per_side: int = 3

    def __call__(self, image: np.ndarray) -> tuple:
        return collect_proposals(self.model, self.processor, image, self.points_per_side)
