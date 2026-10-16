import json
import os
import random
import shutil
from glob import glob

BASE_DIR = r"C:\Users\VANSH\Projects\solar-sweeper-ai"
DATASET_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")
CLASS_MAP = {"debris": 0, "dirt": 1}

# 1. Gather all JSON files in the project data directory
all_jsons = glob(os.path.join(BASE_DIR, "data", "**", "*.json"), recursive=True)
# Exclude any JSONs that might already be inside yolo_dataset
all_jsons = [j for j in all_jsons if "yolo_dataset" not in j]

print(f"Total JSON annotations found: {len(all_jsons)}")

if not all_jsons:
    print(
        "ERROR: No JSON files found. Please verify your Labelme annotations exist under the 'data' folder."
    )
    exit(1)

# 2. Build Image Index across entire project
valid_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp"]
image_index = {}
for root, _, files in os.walk(os.path.join(BASE_DIR, "data")):
    if "yolo_dataset" in root:
        continue
    for f in files:
        name, ext = os.path.splitext(f)
        if ext in valid_exts:
            image_index[name.lower()] = os.path.join(root, f)

print(f"Total source images indexed: {len(image_index)}")

# 3. Setup YOLO directory layout
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

# 4. Shuffle and split 80% train / 20% val
random.seed(42)
random.shuffle(all_jsons)
split_idx = int(0.8 * len(all_jsons))
splits = {"train": all_jsons[:split_idx], "val": all_jsons[split_idx:]}

success_count = {"train": 0, "val": 0}

for split_type, json_list in splits.items():
    for json_path in json_list:
        base_name = os.path.splitext(os.path.basename(json_path))[0]

        # Match corresponding image
        img_path = image_index.get(base_name.lower())
        if not img_path:
            print(f"Warning: Missing image for label '{base_name}.json'")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")

        if not img_w or not img_h:
            # Fallback using PIL if Labelme didn't embed dimensions
            from PIL import Image

            with Image.open(img_path) as im:
                img_w, img_h = im.size

        yolo_lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[label]
            points = shape.get("points", [])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # Clamp coordinates to image boundaries
            xmin = max(0.0, float(xmin))
            ymin = max(0.0, float(ymin))
            xmax = min(float(img_w), float(xmax))
            ymax = min(float(img_h), float(ymax))

            # Normalize to YOLO format
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Write label text file
        out_txt = os.path.join(
            DATASET_DIR, "labels", split_type, f"{base_name}.txt"
        )
        with open(out_txt, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(yolo_lines))

        # Copy image file
        out_img = os.path.join(
            DATASET_DIR, "images", split_type, os.path.basename(img_path)
        )
        shutil.copy(img_path, out_img)
        success_count[split_type] += 1

print("\n--- Summary ---")
print(f"Train images processed: {success_count['train']}")
print(f"Val images processed:   {success_count['val']}")
print("Dataset generation complete!")