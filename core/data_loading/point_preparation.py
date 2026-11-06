"""Select eligible source images and one reproducible target for each image."""

from copy import deepcopy
import random


def select_official_split(data, image_count, seed):
    groups = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd") == 0:
            groups.setdefault(ann["image_id"], []).append(ann)
    images = sorted(
        (image for image in data["images"] if image["id"] in groups), key=lambda image: image["id"]
    )
    random.Random(seed).shuffle(images)
    if image_count < 0 or image_count > len(images):
        raise ValueError("Requested image count exceeds eligible noncrowd source images")
    images = images[:image_count] if image_count else images
    images = sorted(images, key=lambda image: image["id"])
    annotations = [
        random.Random(f"{seed}:{image['id']}").choice(
            sorted(groups[image["id"]], key=lambda ann: ann["id"])
        )
        for image in images
    ]
    return {
        "images": deepcopy(images),
        "annotations": deepcopy(annotations),
        "categories": deepcopy(data["categories"]),
        "selection_counts": {
            "official_source_images": len(data["images"]),
            "eligible_source_images": len(groups),
            "excluded_without_noncrowd_target": len(data["images"]) - len(groups),
            "selected_images": len(images),
            "selected_masks": len(annotations),
        },
    }
