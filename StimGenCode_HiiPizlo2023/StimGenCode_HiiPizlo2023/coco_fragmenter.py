import os, json, argparse, numpy as np, cv2
from pathlib import Path
from pycocotools.coco import COCO

from egg_dataset_helper import StimParam, EggParam, CanvasParam, ImageParam, VorParam

def largest_instance_mask(coco, img_info, ann_list):
    H, W = img_info['height'], img_info['width']
    if not ann_list:
        return None
    best = max(ann_list, key=lambda a: a.get('area',0))
    m = coco.annToMask(best)  # HxW {0,1}
    return (m*255).astype('uint8')

def run_one(image_path, mask, outdir, args):
    # set canvas to image size (keep your grid logic)
    H, W = mask.shape
    egg_param = EggParam(distorting_factor=0.08, direction=1, egg_size=0.3, jitter=args.jitter)
    canvas_param = CanvasParam(canvas_size=(H,W), egg_ecc=0, egg_center=np.array([H//2,W//2]),
                               egg_theta_StartEnd=(0,360), line_thickness=args.thickness,
                               edge_len_factor=args.edge_len_factor,
                               grid_size=np.array([args.grid, args.grid]), tot_gridNum=np.array([H//args.grid, W//args.grid]),
                               noise_offGrid=1.5)

    img_param = ImageParam(output_directory=outdir, file_id=0, set_dire="COCOFragments")
    vorEgg = VorParam(section_num=6, vor_imgsize=(H,W))
    vorBG  = VorParam(section_num=10, vor_imgsize=(H,W))

    stim = StimParam(egg_param, canvas_param, img_param, vorEgg, vorBG, 
                     vorCode={"In":"CC", "Out":"W"}, move_cent=0, donut_factor=1)

    # NEW: build segments from the object mask
    stim.addObjectEdgesFromMask(mask)

    # build mask_img from edges (same as draw_colorCanvas does)
    from egg_dataset_helper import viz_mat, fillArea_manualRaster
    mask_edges = viz_mat(stim.eggStim_mat, stim.canvas_param.canvas_size, connect=True, viz=False)
    stim.mask_img = fillArea_manualRaster(mask_edges, img_size=mask_edges.shape[:2])

    # colorize edges + add noise using existing code
    stim.draw_colorCanvas(randomSeedNum=args.seed, create_img=True)

    # save outputs
    base = Path(outdir)/"COCOFragments"
    base.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(base/f"{stem}_orig.png"), cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(base/f"{stem}_mask.png"), mask)
    cv2.imwrite(str(base/f"{stem}_frag.png"), cv2.cvtColor(stim.stim_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(base/f"{stem}_frag_eggOnly.png"), cv2.cvtColor(stim.stim_eggOnly, cv2.COLOR_RGB2BGR))
    np.save(str(base/f"{stem}_edges.npy"), stim.eggStim_mat)
    np.save(str(base/f"{stem}_noise_in.npy"), stim.noiseMat_InsideEgg)
    np.save(str(base/f"{stem}_noise_out.npy"), stim.noiseMat_OutsideEgg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", required=True, help="instances_val2017.json or similar")
    ap.add_argument("--coco_imgdir", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--edge_len_factor", type=float, default=0.6)  # 0.6 = big gaps, obvious closure
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--min_area", type=int, default=40000, help="filter for large objects")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--jitter", type=int, default=20)
    ap.add_argument("--outdir", default="ClosureCOCO")
    args = ap.parse_args()

    coco = COCO(args.coco_ann)
    ids = coco.getImgIds()
    picked = 0
    for img_id in ids:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns: 
            continue
        # take largest instance above area threshold
        big = [a for a in anns if a.get('area',0) >= args.min_area]
        if not big: 
            continue
        mask = largest_instance_mask(coco, img_info, big)
        img_path = os.path.join(args.coco_imgdir, img_info['file_name'])
        run_one(img_path, mask, args.outdir, args)
        picked += 1
        if picked >= args.num: 
            break

if __name__ == "__main__":
    main()
