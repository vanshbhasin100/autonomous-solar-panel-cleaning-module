import os
import csv
import shutil
from pathlib import Path

# CONFIG
DATA_DIR = Path("data/solar_final_resplit")
SPLITS = ["train", "valid", "test"]
AREA_THRESHOLD = 0.25  # box covers >25% of image area = "large"
OUTPUT_DIR = Path("data/ground_truth_audit/large_boxes_v2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / "large_boxes_list.csv"

rows = []

for split in SPLITS:
    labels_dir = DATA_DIR / split / "labels"
    images_dir = DATA_DIR / split / "images"

    if not labels_dir.exists():
        continue

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            w, h = float(w), float(h)
            area = w * h  # YOLO format is already normalized 0-1

            if area > AREA_THRESHOLD:
                img_name = label_file.stem
                # try common extensions
                img_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = images_dir / (img_name + ext)
                    if candidate.exists():
                        img_path = candidate
                        break

                class_name = "debris" if cls == "0" else "dust"
                rows.append({
                    "split": split,
                    "image": img_name,
                    "class": class_name,
                    "box_area_pct": round(area * 100, 1),
                    "label_file": str(label_file)
                })

                if img_path:
                    shutil.copy(img_path, OUTPUT_DIR / f"{split}_{img_path.name}")

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "image", "class", "box_area_pct", "label_file"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Found {len(rows)} large-box annotations.")
print(f"List saved to: {csv_path}")
print(f"Images copied to: {OUTPUT_DIR}")