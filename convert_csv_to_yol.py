import os
import json
import pandas as pd
import cv2

# paths
CSV_PATH = "jarlids_annots.csv"
IMAGE_ROOT = "images"
LABEL_ROOT = "labels"

# class mapping
CLASS_MAP = {
    "intact": 0,
    "damaged": 1
}

os.makedirs(LABEL_ROOT, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("okk")

for _, row in df.iterrows():
    filename = row["filename"]
    shape_attr = json.loads(row["region_shape_attributes"])
    region_attr = json.loads(row["region_attributes"])

    if shape_attr.get("name") != "rect":
        continue

    x = shape_attr["x"]
    y = shape_attr["y"]
    w = shape_attr["width"]
    h = shape_attr["height"]

    label_name = region_attr["type"]
    class_id = CLASS_MAP[label_name]

    # find image (train/val/test)
    img_path = None
    for split in ["train", "val", "test"]:
        candidate = os.path.join(IMAGE_ROOT, split, filename)
        if os.path.exists(candidate):
            img_path = candidate
            label_dir = os.path.join(LABEL_ROOT, split)
            break

    if img_path is None:
        print("not ok")
        continue

    os.makedirs(label_dir, exist_ok=True)

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    label_file = os.path.join(label_dir, filename.replace(".JPG", ".txt"))

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

print("CSV successfully converted to YOLO format.")
