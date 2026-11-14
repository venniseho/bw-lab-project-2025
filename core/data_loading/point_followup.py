"""Select matched image budgets and exclude targets fully hidden by a fence."""

from copy import deepcopy
import random

import numpy as np
from pycocotools.coco import COCO

from core.stimuli.fence import fence_mask
from core.stimuli.point_transforms import point_transform_seed


def training_subset(data, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1]")
    ids = sorted(image["id"] for image in data["images"])
    random.Random(seed).shuffle(ids)
    count = int(len(ids) * fraction)
    if not count:
        raise ValueError("Training subset is empty")
    return ids[:count]


def filter_targets(data, image_ids, transform, parameters, seed):
    """Check labels only, using exactly the geometry applied by the runtime transform."""
    wanted = set(image_ids)
    images = {image["id"]: image for image in data["images"] if image["id"] in wanted}
    annotations = {ann["image_id"]: ann for ann in data["annotations"] if ann["image_id"] in wanted}
    if len(images) != len(image_ids) or len(annotations) != len(images):
        raise ValueError("Follow-up preparation requires one selected object per image")
    if sum(ann["image_id"] in wanted for ann in data["annotations"]) != len(images):
        raise ValueError("Multiple annotations per image: use a prepared single-target manifest")
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    kept, excluded = [], []
    for checked, image_id in enumerate(image_ids, 1):
        image, annotation = images[image_id], annotations[image_id]
        target = coco.annToMask(annotation).astype(bool)
        if target.shape != (image["height"], image["width"]) or not np.any(target):
            raise ValueError(f"Invalid prepared mask: {annotation['id']}")
        if transform == "fence":
            occlusion = fence_mask(
                image["height"],
                image["width"],
                seed=point_transform_seed(seed, image_id),
                **parameters,
            )
            if not np.any(target & ~occlusion):
                excluded.append(
                    {
                        "image_id": image_id,
                        "annotation_id": annotation["id"],
                        "reason": "fully_hidden_by_fence",
                    }
                )
                continue
        kept.append(image_id)
        if checked % 10000 == 0:
            print(
                f"Eligibility: {checked}/{len(image_ids)} checked; {len(excluded)} fully hidden",
                flush=True,
            )
    if not kept:
        raise ValueError("No visible targets remain")
    result = {
        "images": [deepcopy(images[key]) for key in sorted(kept)],
        "annotations": [deepcopy(annotations[key]) for key in sorted(kept)],
        "categories": deepcopy(data["categories"]),
        "selection_counts": {
            "candidate_images": len(image_ids),
            "excluded_fully_hidden": len(excluded),
            "selected_images": len(kept),
            "selected_masks": len(kept),
        },
        "selection_audit": excluded,
    }
    return result, kept
