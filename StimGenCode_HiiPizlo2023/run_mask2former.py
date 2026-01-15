# run_mask2former.py
import os, glob, cv2, numpy as np, argparse
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

def build_predictor(conf_threshold=0.5):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # COCO instance seg config + weights (Mask2Former R50)
    cfg.merge_from_file("configs/coco/instance-seg/maskformer2_swin_tiny_IN21k_384_bs16_50ep.yaml")
    # If the above config path is unavailable, try one of the Mask2Former example configs you have locally.
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/Mask2Former-R50/137849600/model_final_7c2ce3.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    return DefaultPredictor(cfg)

def overlay_mask(img, mask, alpha=0.6, color=(0,200,255)):
    over = img.copy()
    col = np.full_like(img, color, dtype=np.uint8)
    over = np.where(mask[...,None]>0, cv2.addWeighted(img,1-alpha,col,alpha,0), img)
    edges = cv2.Canny(mask, 0, 1)
    over[edges>0] = (0,0,0)
    return over

def run_folder(in_dir, out_dir, predictor):
    os.makedirs(out_dir, exist_ok=True)
    mdir = os.path.join(out_dir, "masks"); os.makedirs(mdir, exist_ok=True)
    vdir = os.path.join(out_dir, "viz");   os.makedirs(vdir, exist_ok=True)
    odir = os.path.join(out_dir, "overlays"); os.makedirs(odir, exist_ok=True)

    for p in sorted(glob.glob(os.path.join(in_dir, "*"))):
        if not p.lower().endswith((".png",".jpg",".jpeg")): continue
        name = Path(p).stem
        img = cv2.imread(p)
        if img is None: 
            print("skip unreadable:", p); continue

        out = predictor(img)
        inst = out["instances"].to("cpu")
        if len(inst) == 0 or not inst.has("pred_masks"):
            print("no instances:", p); continue

        masks = inst.pred_masks.numpy().astype(np.uint8)*255
        # choose largest instance area
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        mask = masks[int(np.argmax(areas))]

        mask_path = os.path.join(mdir, f"{name}_m2f.png")
        over_path = os.path.join(odir, f"{name}_m2f_overlay.png")
        viz_path  = os.path.join(vdir, f"{name}_m2f_viz.png")

        cv2.imwrite(mask_path, mask)
        over = overlay_mask(img, mask)
        cv2.imwrite(over_path, over)
        cv2.imwrite(viz_path, np.hstack([img, over]))
        print("Mask2Former:", name, "->", mask_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score", type=float, default=0.5)
    args = ap.parse_args()

    predictor = build_predictor(conf_threshold=args.score)
    run_folder(args.images, args.outdir, predictor)
