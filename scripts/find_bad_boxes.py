import csv
import shutil
from pathlib import Path

DATA_DIR = Path("data/solar_final_resplit")
SPLITS = ["train", "valid", "test"]

TINY_THRESHOLD = 0.5      # % area below this = likely degenerate box
DEBRIS_LARGE_THRESHOLD = 40  # % area above this for DEBRIS ONLY = suspicious

OUT_TINY = Path("data/ground_truth_audit/tiny_boxes")
OUT_DEBRIS_LARGE = Path("data/ground_truth_audit/oversized_debris")
OUT_TINY.mkdir(parents=True, exist_ok=True)
OUT_DEBRIS_LARGE.mkdir(parents=True, exist_ok=True)

tiny_rows = []
debris_large_rows = []

for split in SPLITS:
    labels_dir = DATA_DIR / split / "labels"
    images_dir = DATA_DIR / split / "images"
    if not labels_dir.exists():
        continue

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / (label_file.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            w, h = float(w), float(h)
            area_pct = w * h * 100
            class_name = "debris" if cls == "0" else "dust"

            row = {
                "split": split, "image": label_file.stem, "class": class_name,
                "box_area_pct": round(area_pct, 3), "w": w, "h": h
            }

            if area_pct < TINY_THRESHOLD:
                tiny_rows.append(row)
                if img_path:
                    shutil.copy(img_path, OUT_TINY / f"{split}_{img_path.name}")

            if class_name == "debris" and area_pct > DEBRIS_LARGE_THRESHOLD:
                debris_large_rows.append(row)
                if img_path:
                    shutil.copy(img_path, OUT_DEBRIS_LARGE / f"{split}_{img_path.name}")

# Save CSVs
for rows, path in [
    (tiny_rows, OUT_TINY / "tiny_boxes_list.csv"),
    (debris_large_rows, OUT_DEBRIS_LARGE / "oversized_debris_list.csv"),
]:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "class", "box_area_pct", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)

print(f"Tiny boxes (<{TINY_THRESHOLD}% area): {len(tiny_rows)} -> {OUT_TINY}")
print(f"Oversized DEBRIS boxes (>{DEBRIS_LARGE_THRESHOLD}% area): {len(debris_large_rows)} -> {OUT_DEBRIS_LARGE}")