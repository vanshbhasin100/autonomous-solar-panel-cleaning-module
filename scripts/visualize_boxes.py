import csv
import cv2
from pathlib import Path

DATA_DIR = Path("data/solar_final_resplit")

def draw_boxes_for_csv(csv_path, out_dir, sample_size=20):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    rows = rows[:sample_size]  # just take first N for a quick look

    for row in rows:
        split = row["split"]
        img_stem = row["image"]
        images_dir = DATA_DIR / split / "images"
        labels_dir = DATA_DIR / split / "labels"

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / (img_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        label_path = labels_dir / (img_stem + ".txt")
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                cls, x, y, w, h = int(cls), float(x), float(y), float(w), float(h)

                x1 = int((x - w/2) * w_img)
                y1 = int((y - h/2) * h_img)
                x2 = int((x + w/2) * w_img)
                y2 = int((y + h/2) * h_img)

                color = (0, 0, 255) if cls == 0 else (255, 0, 0)  # red=debris, blue=dust
                label = "debris" if cls == 0 else "dust"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_path = out_dir / f"{split}_{img_stem}.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"Saved {len(rows)} annotated images to {out_dir}")

# Tiny boxes sample
draw_boxes_for_csv(
    "data/ground_truth_audit/tiny_boxes/tiny_boxes_list.csv",
    "data/ground_truth_audit/tiny_boxes_visualized",
    sample_size=20
)

# Oversized debris sample (do all 177 since it's a manageable number)
draw_boxes_for_csv(
    "data/ground_truth_audit/oversized_debris/oversized_debris_list.csv",
    "data/ground_truth_audit/oversized_debris_visualized",
    sample_size=177
)