"""Image-only model contract, separate from targets and experiment settings."""
from typing import Protocol

import numpy as np


class Model(Protocol):
    def predict(self, image: np.ndarray) -> object:
        """Predict from native RGB pixels; never receive evaluation targets."""
        ...
