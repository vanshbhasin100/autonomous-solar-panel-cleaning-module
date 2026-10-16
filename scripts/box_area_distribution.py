import csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/solar_final_resplit")
SPLITS = ["train", "valid", "test"]

# bucket boxes by size range
buckets = defaultdict(int)
class_areas = {"debris": [], "dust": []}

def bucket_for(area_pct):
    if area_pct < 5: return "0-5%"
    if area_pct < 10: return "5-10%"
    if area_pct < 15: return "10-15%"
    if area_pct < 25: return "15-25%"
    if area_pct < 40: return "25-40%"
    if area_pct < 60: return "40-60%"
    return "60-100%"

total_boxes = 0

for split in SPLITS:
    labels_dir = DATA_DIR / split / "labels"
    if not labels_dir.exists():
        continue
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                w, h = float(w), float(h)
                area_pct = w * h * 100
                total_boxes += 1
                buckets[bucket_for(area_pct)] += 1
                class_name = "debris" if cls == "0" else "dust"
                class_areas[class_name].append(area_pct)

print(f"Total boxes scanned: {total_boxes}\n")
print("Box size distribution (% of image area):")
for b in ["0-5%", "5-10%", "10-15%", "15-25%", "25-40%", "40-60%", "60-100%"]:
    count = buckets[b]
    pct = round(count / total_boxes * 100, 1)
    print(f"  {b:10s}: {count:5d} boxes ({pct}%)")

print("\nPer-class average box area:")
for cls_name, areas in class_areas.items():
    if areas:
        avg = sum(areas) / len(areas)
        print(f"  {cls_name}: avg {avg:.1f}%, max {max(areas):.1f}%, min {min(areas):.1f}%")