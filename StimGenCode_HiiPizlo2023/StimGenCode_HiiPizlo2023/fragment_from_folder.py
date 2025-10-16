import os, cv2, numpy as np
from egg_dataset_helper import StimParam, EggParam, CanvasParam, ImageParam, VorParam, viz_mat, fillArea_manualRaster

in_dir   = "images"
mask_dir = "masks"
out_dir  = "output"
os.makedirs(out_dir, exist_ok=True)

# Process each image in the input directory
for name in os.listdir(in_dir):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Define paths
    img_path  = os.path.join(in_dir, name)
    mask_path = os.path.join(mask_dir, name.rsplit('.',1)[0] + "_mask.png")

    if not os.path.exists(mask_path):
        print(f"No mask for {name}")
        continue

    # Read image and mask
    mask = cv2.imread(mask_path, 0)
    H, W = mask.shape

    # Set parameters
    egg   = EggParam(distorting_factor=0.08, direction=1, egg_size=0.3, jitter=20)
    canvas = CanvasParam(canvas_size=(H, W), edge_len_factor=0.6,
                         grid_size=(40, 40), tot_gridNum=(H//40, W//40))
    imgp  = ImageParam(output_directory=out_dir, file_id=0, set_dire="manual")
    vorE, vorB = VorParam(section_num=1, vor_imgsize=(H, W)), VorParam(section_num=1, vor_imgsize=(H, W))

    # Create stimulus
    stim = StimParam(egg, canvas, imgp, vorE, vorB, vorCode={"In":"CC", "Out":"W"})
    stim.addObjectEdgesFromMask(mask)

    # Generate fragmented image
    edge_img = viz_mat(stim.eggStim_mat, stim.canvas_param.canvas_size, connect=True, viz=False)
    stim.mask_img = fillArea_manualRaster(edge_img, img_size=edge_img.shape[:2])
    stim.draw_colorCanvas(randomSeedNum=123, create_img=True)

    # Save output
    out_file = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_fragmented.png")
    cv2.imwrite(out_file, cv2.cvtColor(stim.stim_img, cv2.COLOR_RGB2BGR))
    print("Fragmented:", out_file)
