import cv2
import numpy as np
import os

# input and output folders
in_dir = "images"
out_dir = "masks"
os.makedirs(out_dir, exist_ok=True)

# loop through every image in your images folder
for name in os.listdir(in_dir):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(in_dir, name)
    img = cv2.imread(path)

    if img is None:
        print(f"Could not read {name}")
        continue

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # anything brighter than ~10 becomes white (foreground)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    out_path = os.path.join(out_dir, name.rsplit(".", 1)[0] + "_mask.png")
    cv2.imwrite(out_path, mask)
    print(f"Saved binary mask: {out_path}")

print("\nAll masks created. in 'masks/' folder.")
