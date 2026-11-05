"""Select disjoint COCO source images before expanding instance annotations."""

from copy import deepcopy
import random


def select_splits(data, train_count, heldout_count, seed):
    if train_count < 1 or heldout_count < 1:
        raise ValueError("Both split sizes must be positive")
    annotations = [ann for ann in data["annotations"] if ann.get("iscrowd") == 0]
    eligible = {ann["image_id"] for ann in annotations}
    images = sorted((im for im in data["images"] if im["id"] in eligible), key=lambda im: im["id"])
    if len(images) < train_count + heldout_count:
        raise ValueError("Not enough noncrowd training sources")
    chosen = random.Random(seed).sample(images, train_count + heldout_count)
    result = {}
    for name, selected in (("train", chosen[:train_count]), ("test", chosen[train_count:])):
        ids = {image["id"] for image in selected}
        result[name] = {
            "images": sorted(deepcopy(selected), key=lambda image: image["id"]),
            "annotations": deepcopy([ann for ann in annotations if ann["image_id"] in ids]),
            "categories": deepcopy(data["categories"]),
        }
    return result


def validate_masks(data):
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    for annotation in data["annotations"]:
        mask = coco.annToMask(annotation)
        image = coco.imgs[annotation["image_id"]]
        if mask.shape != (image["height"], image["width"]) or not mask.any():
            raise ValueError(f"Invalid/empty annotation mask: {annotation['id']}")
