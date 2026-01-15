# run_sam.py
import os, sys, cv2, numpy as np, argparse, glob
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def overlay_mask(img, mask, alpha=0.6, color=(255,200,0)):
    over = img.copy()
    col = np.full_like(img, color, dtype=np.uint8)
    over = np.where(mask[...,None]>0, cv2.addWeighted(img, 1-alpha, col, alpha, 0), img)
    # crisp boundary
    edges = cv2.Canny(mask, 0, 1)
    over[edges>0] = (0,0,0)
    return over

def run_folder(in_dir, out_dir, sam_ckpt, model_type="vit_h", point_mode=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    ensure_dir(out_dir)
    for p in sorted(glob.glob(os.path.join(in_dir, "*"))):
        if not p.lower().endswith((".png",".jpg",".jpeg")): continue
        name = Path(p).stem
        img = cv2.imread(p)
        if img is None: 
            print("skip unreadable:", p); continue
        h, w = img.shape[:2]
        predictor.set_image(img)

        # prompting: a single positive point at center + whole-image box
        center = np.array([[w//2, h//2]])
        box = np.array([0,0,w,h])
        masks, _, _ = predictor.predict(
            point_coords=center if point_mode else None,
            point_labels=np.array([1]) if point_mode else None,
            box=box,
            multimask_output=True
        )
        # keep the largest mask
        areas = [m.sum() for m in masks]
        mask = masks[int(np.argmax(areas))].astype(np.uint8)*255

        # save
        mdir = os.path.join(out_dir, "masks"); ensure_dir(mdir)
        vdir = os.path.join(out_dir, "viz"); ensure_dir(vdir)
        odir = os.path.join(out_dir, "overlays"); ensure_dir(odir)

        mask_path = os.path.join(mdir, f"{name}_sam.png")
        viz_path  = os.path.join(vdir, f"{name}_sam_viz.png")
        over_path = os.path.join(odir, f"{name}_sam_overlay.png")

        cv2.imwrite(mask_path, mask)
        over = overlay_mask(img, mask)
        cv2.imwrite(over_path, over)
        canvas = np.hstack([img, over])
        cv2.imwrite(viz_path, canvas)
        print("SAM:", name, "->", mask_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="input folder of images")
    ap.add_argument("--outdir", required=True, help="output folder")
    ap.add_argument("--sam_ckpt", default="weights/sam_vit_h_4b8939.pth")
    ap.add_argument("--model", default="vit_h")
    args = ap.parse_args()
    run_folder(args.images, args.outdir, args.sam_ckpt, args.model)
