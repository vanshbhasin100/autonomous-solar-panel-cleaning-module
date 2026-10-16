import csv
import cv2
from pathlib import Path

DATA_DIR = Path("data/solar_final_resplit")
CSV_PATH = Path("data/ground_truth_audit/tiny_boxes/tiny_boxes_list.csv")
OUT_DIR = Path("data/ground_truth_audit/tiny_boxes_zoomed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAD_FACTOR = 6  # how much extra context around the tiny box to show (bigger = more zoomed out)

with open(CSV_PATH, "r") as f:
    rows = list(csv.DictReader(f))

print(f"Generating zoomed crops for {len(rows)} tiny boxes...")

for row in rows:
    split = row["split"]
    img_stem = row["image"]
    w, h = float(row["w"]), float(row["h"])

    images_dir = DATA_DIR / split / "images"
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_dir / (img_stem + ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        continue

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    label_path = DATA_DIR / split / "labels" / f"{img_stem}.txt"
    with open(label_path, "r") as f:
        lines = f.readlines()

    # find the matching tiny box line (match by w,h)
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, bw, bh = parts
        x, y, bw, bh = float(x), float(y), float(bw), float(bh)
        if round(bw, 6) != round(w, 6) or round(bh, 6) != round(h, 6):
            continue

        cx, cy = int(x * img_w), int(y * img_h)
        box_w_px, box_h_px = int(bw * img_w), int(bh * img_h)

        crop_w = max(box_w_px * PAD_FACTOR, 150)
        crop_h = max(box_h_px * PAD_FACTOR, 150)

        x1 = max(cx - crop_w // 2, 0)
        y1 = max(cy - crop_h // 2, 0)
        x2 = min(cx + crop_w // 2, img_w)
        y2 = min(cy + crop_h // 2, img_h)

        crop = img[y1:y2, x1:x2].copy()

        # draw box relative to crop
        bx1 = int((x - bw/2) * img_w) - x1
        by1 = int((y - bh/2) * img_h) - y1
        bx2 = int((x + bw/2) * img_w) - x1
        by2 = int((y + bh/2) * img_h) - y1
        color = (0, 0, 255) if cls == "0" else (255, 0, 0)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 1)

        # upscale for visibility
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        out_name = f"{split}_{img_stem}.jpg"
        cv2.imwrite(str(OUT_DIR / out_name), crop)
        break  # only first match per row

print(f"Done. Saved to {OUT_DIR}")