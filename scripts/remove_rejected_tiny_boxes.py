import csv
from pathlib import Path

DATA_DIR = Path("data/solar_final_resplit")
DECISIONS_CSV = Path("data/ground_truth_audit/tiny_boxes_decisions.csv")
TINY_LIST_CSV = Path("data/ground_truth_audit/tiny_boxes/tiny_boxes_list.csv")

# Load decisions
rejected_names = set()
with open(DECISIONS_CSV, "r") as f:
    for row in csv.DictReader(f):
        if row["decision"] == "reject":
            rejected_names.add(row["image"])  # e.g. "train_img123.jpg"

# Reconstruct which exact box (w,h) was shown for each image
# This mirrors the "first match" logic used when generating the zoomed crops
seen_images = set()
rejected_specs = []
with open(TINY_LIST_CSV, "r") as f:
    for row in csv.DictReader(f):
        vis_name = f"{row['split']}_{row['image']}.jpg"
        if vis_name in seen_images:
            continue  # only first box per image was used for the crop
        seen_images.add(vis_name)

        if vis_name in rejected_names:
            rejected_specs.append({
                "split": row["split"],
                "image": row["image"],
                "w": round(float(row["w"]), 6),
                "h": round(float(row["h"]), 6),
            })

print(f"Found {len(rejected_specs)} rejected tiny box specs to remove.")

removed_count = 0
for spec in rejected_specs:
    label_path = DATA_DIR / spec["split"] / "labels" / f"{spec['image']}.txt"
    if not label_path.exists():
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    removed_this_file = False
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            new_lines.append(line)
            continue
        cls, x, y, w, h = parts
        w_r, h_r = round(float(w), 6), round(float(h), 6)

        if w_r == spec["w"] and h_r == spec["h"] and not removed_this_file:
            removed_count += 1
            removed_this_file = True
            continue
        new_lines.append(line)

    with open(label_path, "w") as f:
        f.writelines(new_lines)

print(f"Removed {removed_count} bad tiny box lines.")