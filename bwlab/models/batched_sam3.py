"""Tensor image batches and independent prompt batches for pinned native SAM3."""
from dataclasses import dataclass
import time

import numpy as np
import torch

from bwlab.models.proposals import Proposal, point_grid


def _tensor_image(processor, rgb, pin_memory=False):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected uint8 HWC RGB arrays")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    if pin_memory and str(processor.device).startswith("cuda"):
        tensor = tensor.pin_memory()
    return processor.transform(tensor.to(processor.device, non_blocking=pin_memory))


def _tensor_images(processor, images, pin_memory=False):
    """One packed uint8 transfer; native BCHW preprocessing per source shape.

    Variable-size images cannot form a rectangular tensor before resizing.
    Packing avoids padding and preserves native interpolation at image edges.
    Shape groups preserve sample order when reassembled for the encoder.
    """
    if not images:
        raise ValueError("nonempty image batch required")
    groups = {}
    offset = 0
    for index, rgb in enumerate(images):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected uint8 HWC RGB arrays")
        groups.setdefault(rgb.shape, []).append((index, offset, rgb.size))
        offset += rgb.size
    packed = torch.from_numpy(np.concatenate([rgb.reshape(-1) for rgb in images]))
    if pin_memory and str(processor.device).startswith("cuda"):
        packed = packed.pin_memory()
    packed = packed.to(processor.device, non_blocking=pin_memory)
    pixels = [None] * len(images)
    for shape, members in groups.items():
        batch = torch.stack([packed[start:start+length].reshape(shape)
                             for _, start, length in members]).permute(0, 3, 1, 2)
        transformed = processor.transform(batch)
        if transformed.ndim != 4 or len(transformed) != len(members):
            raise ValueError("native processor must preserve the image batch dimension")
        if len(groups) == 1:
            return transformed
        for (index, _, _), pixel in zip(members, transformed.unbind(0)):
            pixels[index] = pixel
    return torch.stack(pixels)


def _tensor_state(model, processor, images, pin_memory=False):
    """Match native processor preprocessing and its interactive feature projections.

    Meta's set_image_batch only accepts PIL images at the verified revision.
    Keep CHW uint8 tensors through transfer, then use its own transform. This
    avoids the PIL round trip and preserves its resize/normalization semantics.
    """
    batch = _tensor_images(processor, images, pin_memory)
    features = model.backbone.forward_image(batch)
    if "sam2_backbone_out" not in features:
        raise RuntimeError("native SAM3 returned no interactive backbone features")
    pyramid = features["sam2_backbone_out"]["backbone_fpn"]
    decoder = model.inst_interactive_predictor.model.sam_mask_decoder
    pyramid[0] = decoder.conv_s0(pyramid[0])
    pyramid[1] = decoder.conv_s1(pyramid[1])
    return {"original_heights": [im.shape[0] for im in images],
            "original_widths": [im.shape[1] for im in images], "backbone_out": features}


def _append_masks(proposals, masks, scores, points, offset, shape):
    # Native predictor squeezes the prompt dimension when its length is one.
    if masks.ndim == 3:
        masks, scores = masks[None], scores[None]
    if (masks.ndim != 4 or masks.shape[0] != len(points)
            or masks.shape[2:] != shape or scores.shape != masks.shape[:2]
            or not np.isfinite(scores).all() or not np.all((masks == 0) | (masks == 1))):
        raise ValueError("unaligned/nonbinary native batched predictions")
    for local, point in enumerate(points):
        for candidate, (mask, score) in enumerate(zip(masks[local], scores[local])):
            proposals.append(Proposal(np.packbits(mask.astype(bool)), shape, {
                "proposal_id": len(proposals), "grid_point_index": offset + local,
                "candidate_index": candidate, "point_xy": point.astype(float).tolist(),
                "point_label": 1, "predicted_iou_score": float(score),
                "area_pixels": int(np.count_nonzero(mask))}))


@dataclass
class BatchedSam3GridAdapter:
    model: object
    processor: object
    points_per_side: int = 3
    prompt_batch_size: int = 9
    pin_memory: bool = True
    synchronize: object = lambda: None

    @torch.inference_mode()
    def predict_batch(self, images):
        """Batch the backbone; batch independent one-point queries per image.

        The native decoder still iterates over images internally. A prompt
        tensor has shape (Q,1,2), never (1,Q,2): Q separate object queries.
        Timings are batch wall times amortized over images, not isolated latency.
        """
        if not images or self.prompt_batch_size < 1:
            raise ValueError("nonempty image batch and positive prompt_batch_size required")
        if not callable(getattr(self.model, "predict_inst_batch", None)):
            raise RuntimeError("installed SAM3 lacks predict_inst_batch; use the verified native revision")
        self.synchronize()
        start = time.perf_counter()
        state = _tensor_state(self.model, self.processor, images, self.pin_memory)
        self.synchronize()
        embedding = time.perf_counter() - start
        grids = [point_grid(*im.shape[:2], self.points_per_side) for im in images]
        outputs = [[] for _ in images]
        query_start = time.perf_counter()
        predictor = self.model.inst_interactive_predictor
        try:
            for offset in range(0, len(grids[0]), self.prompt_batch_size):
                points = [grid[offset:offset+self.prompt_batch_size] for grid in grids]
                masks, scores, low_resolution_logits = self.model.predict_inst_batch(
                    state, point_coords_batch=[p[:, None, :] for p in points],
                    point_labels_batch=[np.ones((len(p), 1), dtype=np.int32) for p in points],
                    multimask_output=True, return_logits=False, normalize_coords=True)
                if len(masks) != len(images) or len(scores) != len(images):
                    raise ValueError("native batch output count does not match input")
                for i, rgb in enumerate(images):
                    _append_masks(outputs[i], masks[i], scores[i], points[i], offset, rgb.shape[:2])
                del masks, scores, low_resolution_logits
            self.synchronize()
        finally:
            # Native implementation resets these only on successful prediction.
            predictor._features = None
            predictor._is_image_set = predictor._is_batch = False
        query = time.perf_counter() - query_start
        total = time.perf_counter() - start
        return [(proposals, {"grid_point_count": len(grids[i]),
                             "raw_proposal_count": len(proposals),
                             "image_embedding_seconds": embedding/len(images),
                             "point_query_seconds": query/len(images),
                             "collection_seconds": total/len(images),
                             "timing_scope": "batch wall time amortized over images",
                             "image_batch_size": len(images),
                             "prompt_batch_size": self.prompt_batch_size})
                for i, proposals in enumerate(outputs)]
