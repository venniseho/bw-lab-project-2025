"""Native Meta SAM3 loading and image-only inference adapter."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from bwlab.infrastructure import file_fingerprint
from bwlab.models.proposals import collect_proposals

from dataclasses import dataclass

VERIFIED_SAM3_API_REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"


def checked_state(model, state):
    """Same native checkpoint prefixes as Meta's loader, but reject partial loads."""
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Expected a native SAM3 checkpoint state dictionary")
    mapped = {}
    for key, value in state.items():
        if key.startswith("detector."):
            mapped[key.removeprefix("detector.")] = value
        elif key.startswith("tracker."):
            mapped["inst_interactive_predictor.model." + key.removeprefix("tracker.")] = value
    if not mapped:
        raise ValueError("Not a native Meta SAM3 detector/tracker checkpoint")
    result = model.load_state_dict(mapped, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Incomplete checkpoint: {result.missing_keys}")
    return list(result.unexpected_keys)


def validate_scene_api(model, processor):
    """Check the installed API; refuse incompatible signatures or cleanup defaults."""
    interactive = getattr(model, "inst_interactive_predictor", None)
    transforms = getattr(interactive, "_transforms", None)
    if (not callable(getattr(model, "predict_inst", None))
            or not callable(getattr(processor, "set_image", None))
            or not callable(getattr(interactive, "predict", None))
            or not hasattr(interactive, "mask_threshold")
            or any(not hasattr(transforms, name) for name in
                   ("mask_threshold", "max_hole_area", "max_sprinkle_area"))):
        raise RuntimeError("Installed native SAM3 lacks the required processor/predict_inst API")
    arguments = dict(point_coords=np.zeros((1, 2), dtype=np.float32),
                     point_labels=np.ones(1, dtype=np.int32),
                     multimask_output=True, return_logits=False, normalize_coords=True)
    try:
        inspect.signature(processor.set_image).bind(object())
        inspect.signature(model.predict_inst).bind({}, **arguments)
        inspect.signature(interactive.predict).bind(**arguments)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Installed native SAM3 visual API has incompatible signatures") from error
    # Configure the EXISTING integrated predictor. Never call its set_image:
    # its tracker intentionally has no backbone; predict_inst supplies features.
    interactive.mask_threshold = 0.0
    transforms.mask_threshold = 0.0
    transforms.max_hole_area = 0.0
    transforms.max_sprinkle_area = 0.0


def load_scene_model(checkpoint, bpe_path, torch):
    import sam3
    import sam3.model_builder as builder
    import sam3.model.sam1_task_predictor as visual
    import sam3.model.sam3_image_processor as processing
    import sam3.model.sam3_image as image_model
    bpe = bpe_path or Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
    if not Path(bpe).is_file():
        raise FileNotFoundError(f"Local tokenizer asset missing: {bpe}")
    model = builder.build_sam3_image_model(
        device="cpu", bpe_path=str(bpe), checkpoint_path=None,
        load_from_HF=False, enable_inst_interactivity=True, compile=False,
    )
    unexpected = checked_state(model, torch.load(checkpoint, map_location="cpu", weights_only=True))
    # Preserve native parameter/buffer dtypes. A blanket float32 cast discards
    # the imaginary component of registered complex RoPE frequency buffers.
    model = model.to(device="cuda").eval()
    # Meta's model_builder enables TF32 during import. Override after importing it.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = processing.Sam3Processor(model, device="cuda")
    validate_scene_api(model, processor)
    return model, processor, {
        "checkpoint": file_fingerprint(Path(checkpoint)),
        "builder_source": file_fingerprint(Path(builder.__file__)),
        "predictor_source": file_fingerprint(Path(visual.__file__)),
        "processor_source": file_fingerprint(Path(processing.__file__)),
        "image_model_source": file_fingerprint(Path(image_model.__file__)),
        "tokenizer": file_fingerprint(Path(bpe)),
        "unexpected_checkpoint_keys": unexpected,
        "api": "Meta Sam3Processor.set_image / Sam3Image.predict_inst",
        "verified_api_revision": VERIFIED_SAM3_API_REVISION,
        "model_family": "SAM3 (not SAM3.1)", "text_queries": False,
        "mask_threshold": 0.0, "max_hole_area": 0.0, "max_sprinkle_area": 0.0,
    }


@dataclass
class Sam3GridAdapter:
    """Reuse the verified native processor/predict_inst path on loaded models.

    Caller owns inference_mode, SLURM checks and model loading, as in the
    existing sam3_scene_masks CLI. This class does not submit GPU jobs.
    """

    model: object
    processor: object
    points_per_side: int = 3

    def predict(self, image: np.ndarray) -> tuple:
        return collect_proposals(self.model, self.processor, image, self.points_per_side)

    def __call__(self, image: np.ndarray) -> tuple:
        return self.predict(image)
