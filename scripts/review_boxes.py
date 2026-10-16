import csv
from pathlib import Path
import cv2

VIS_DIR = Path("data/ground_truth_audit/tiny_boxes_zoomed")
DECISIONS_CSV = Path("data/ground_truth_audit/tiny_boxes_decisions.csv")

images = sorted(VIS_DIR.glob("*.jpg"))
print(f"Found {len(images)} images to review.")
print("Controls: [k] keep as-is   [x] reject/remove   [f] flag for manual fix   [q] quit & save")

decisions = {}
if DECISIONS_CSV.exists():
    with open(DECISIONS_CSV, "r") as f:
        for row in csv.DictReader(f):
            decisions[row["image"]] = row["decision"]

for img_path in images:
    name = img_path.name
    if name in decisions:
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    cv2.imshow("Review - k=keep x=reject f=flag q=quit", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('k'):
        decisions[name] = "keep"
    elif key == ord('x'):
        decisions[name] = "reject"
    elif key == ord('f'):
        decisions[name] = "flag_manual_fix"
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

with open(DECISIONS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "decision"])
    for name, decision in decisions.items():
        writer.writerow([name, decision])

print(f"Saved {len(decisions)} decisions to {DECISIONS_CSV}")