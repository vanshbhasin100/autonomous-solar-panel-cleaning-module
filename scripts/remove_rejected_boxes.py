import csv
from pathlib import Path

DATA_DIR = Path("data/solar_final_resplit")
DECISIONS_CSV = Path("data/ground_truth_audit/oversized_debris_decisions.csv")
OVERSIZED_LIST_CSV = Path("data/ground_truth_audit/oversized_debris/oversized_debris_list.csv")

# Load which images/boxes were rejected
rejected_images = set()
with open(DECISIONS_CSV, "r") as f:
    for row in csv.DictReader(f):
        if row["decision"] == "reject":
            # filename format is "{split}_{original_name}.jpg"
            rejected_images.add(row["image"])

# Load the original box info (split, image stem, w, h) so we can find the exact matching line
rejected_box_specs = []
with open(OVERSIZED_LIST_CSV, "r") as f:
    for row in csv.DictReader(f):
        vis_name = f"{row['split']}_{row['image']}.jpg"
        if vis_name in rejected_images:
            rejected_box_specs.append({
                "split": row["split"],
                "image": row["image"],
                "w": round(float(row["w"]), 6),
                "h": round(float(row["h"]), 6),
            })

print(f"Found {len(rejected_box_specs)} rejected box specs to remove.")

removed_count = 0
for spec in rejected_box_specs:
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

        # match debris class (0) with matching w/h to identify the exact bad box
        if cls == "0" and w_r == spec["w"] and h_r == spec["h"] and not removed_this_file:
            removed_count += 1
            removed_this_file = True
            continue  # skip = remove this line
        new_lines.append(line)

    with open(label_path, "w") as f:
        f.writelines(new_lines)

print(f"Removed {removed_count} bad debris box lines.")
print("Backup is safely stored in data/solar_final_resplit_labels_backup/")