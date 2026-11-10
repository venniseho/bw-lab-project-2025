"""Select eligible source images and one reproducible target for each image."""

from copy import deepcopy
import random


def _mask_issue(coco, annotation, image):
    try:
        mask = coco.annToMask(annotation)
    except (ValueError, TypeError, IndexError, KeyError, AssertionError, OverflowError) as error:
        return f"decode_error:{type(error).__name__}"
    if mask.shape != (image["height"], image["width"]):
        return "mask_shape_mismatch"
    return None if mask.any() else "empty_mask"


def _valid_target(coco, candidates, image, seed, audit):
    """Preserve the previous first choice; try seeded alternatives only if invalid."""
    candidates = sorted(candidates, key=lambda ann: ann["id"])
    rng = random.Random(f"{seed}:{image['id']}")
    first = rng.choice(candidates)
    remaining = [ann for ann in candidates if ann["id"] != first["id"]]
    rng.shuffle(remaining)
    for index, annotation in enumerate([first, *remaining]):
        reason = _mask_issue(coco, annotation, image)
        if reason is None:
            if index:
                audit.append(
                    {
                        "image_id": image["id"],
                        "annotation_id": annotation["id"],
                        "action": "replacement_selected",
                        "reason": "valid_alternative",
                    }
                )
            return annotation
        audit.append(
            {
                "image_id": image["id"],
                "annotation_id": annotation["id"],
                "action": "rejected_annotation",
                "reason": reason,
            }
        )
    audit.append(
        {
            "image_id": image["id"],
            "annotation_id": None,
            "action": "excluded_image",
            "reason": "no_valid_noncrowd_mask",
        }
    )
    return None


def select_official_split(data, image_count, seed):
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = data
    coco.createIndex()
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
    audit = [
        {
            "image_id": image["id"],
            "annotation_id": None,
            "action": "excluded_image",
            "reason": "no_noncrowd_annotation",
        }
        for image in data["images"]
        if image["id"] not in groups
    ]
    selected, annotations = [], []
    for checked, image in enumerate(images, 1):
        annotation = _valid_target(coco, groups[image["id"]], image, seed, audit)
        if annotation is not None:
            selected.append(image)
            annotations.append(annotation)
        if checked % 10000 == 0:
            print(
                f"Checked masks for {checked} candidate images; {len(selected)} usable targets",
                flush=True,
            )
        if image_count and len(selected) == image_count:
            break
    if image_count and len(selected) < image_count:
        raise ValueError(
            f"Requested {image_count} images, but only {len(selected)} have valid noncrowd masks"
        )
    if not selected:
        raise ValueError("No images have a valid noncrowd target mask")
    images = sorted(selected, key=lambda image: image["id"])
    annotations = sorted(annotations, key=lambda ann: ann["image_id"])
    return {
        "images": deepcopy(images),
        "annotations": deepcopy(annotations),
        "categories": deepcopy(data["categories"]),
        "selection_audit": audit,
        "selection_counts": {
            "official_source_images": len(data["images"]),
            "eligible_source_images": len(groups),
            "excluded_without_noncrowd_target": len(data["images"]) - len(groups),
            "selected_images": len(images),
            "selected_masks": len(annotations),
            "excluded_without_valid_mask": sum(
                row["reason"] == "no_valid_noncrowd_mask" for row in audit
            ),
            "rejected_annotations": sum(row["action"] == "rejected_annotation" for row in audit),
            "replacement_target_images": sum(
                row["action"] == "replacement_selected" for row in audit
            ),
        },
    }
