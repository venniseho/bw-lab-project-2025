# compare_segmentations.py
import os, glob, cv2, numpy as np, csv
from pathlib import Path

def iou_and_dice(m1, m2):
    m1 = (m1>0).astype(np.uint8); m2 = (m2>0).astype(np.uint8)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    iou = inter / max(1, union)
    dice = (2*inter) / max(1, m1.sum()+m2.sum())
    return float(iou), float(dice)

def load_mask(path, size=None):
    m = cv2.imread(path, 0)
    if m is None: return None
    if size is not None:
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return m

def pairwise_eval(model_name, orig_dir, frag_dir, out_csv):
    rows = [("model","name","iou","dice")]
    for p in sorted(glob.glob(os.path.join(orig_dir,"masks","*_*.png"))):
        name = Path(p).stem.replace(f"_{model_name}","")
        q = os.path.join(frag_dir,"masks", f"{name}_{model_name}.png")
        if not os.path.exists(q): 
            continue
        m1 = load_mask(p)
        m2 = load_mask(q, size=(m1.shape[1], m1.shape[0]))
        iou, dice = iou_and_dice(m1, m2)
        rows.append((model_name, name, iou, dice))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("wrote:", out_csv)

if __name__ == "__main__":
    pairwise_eval(
        model_name="sam",
        orig_dir="outputs/sam/originals",
        frag_dir="outputs/sam/fragmented",
        out_csv="outputs/sam/compare_sam.csv"
    )
    pairwise_eval(
        model_name="m2f",
        orig_dir="outputs/m2f/originals",
        frag_dir="outputs/m2f/fragmented",
        out_csv="outputs/m2f/compare_m2f.csv"
    )
 