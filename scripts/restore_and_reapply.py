import shutil
from pathlib import Path

DATA_DIR = Path("data/solar_final_resplit")
BACKUP_DIR = Path("data/solar_final_resplit_labels_backup")

SPLITS = ["train", "valid", "test"]

# Step 1: restore all labels from the clean backup (undoes BOTH removals)
for split in SPLITS:
    src = BACKUP_DIR / split / "labels"
    dst = DATA_DIR / split / "labels"
    if not src.exists():
        print(f"Warning: no backup found for {split}")
        continue
    shutil.rmtree(dst)
    shutil.copytree(src, dst)

print("Restored all labels from backup (tiny-box removals undone).")
print("Now re-run scripts\\remove_rejected_boxes.py to reapply ONLY the oversized-debris cleanup.")