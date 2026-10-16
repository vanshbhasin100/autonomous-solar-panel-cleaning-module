"""Build the MobileNet binary-classification dataset (clean vs contaminated).

Sources from the LEAKAGE-FREE grouped YOLO split (data/solar_grouped), not the
old leaked solar_final_resplit. An image is "contaminated" when its YOLO label
file is non-empty, else "clean".

Split mapping: grouped train -> binary train, valid -> val, test -> test.
Rebuilds data/binary_classification from scratch (images are copies, so this
is reproducible and non-lossy).
"""
import shutil
from pathlib import Path

YOLO_ROOT = Path(r"C:\Users\VANSH\Projects\solar-sweeper-ai\data\solar_grouped")
BINARY_ROOT = Path(r"C:\Users\VANSH\Projects\solar-sweeper-ai\data\binary_classification")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# grouped split -> binary split
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def is_contaminated(lbl_path: Path) -> bool:
    return lbl_path.exists() and lbl_path.read_text().strip() != ""


def main():
    if BINARY_ROOT.exists():
        print(f"Removing stale (leaked) binary dataset: {BINARY_ROOT}")
        shutil.rmtree(BINARY_ROOT)

    for ysplit, bsplit in SPLIT_MAP.items():
        img_dir = YOLO_ROOT / ysplit / "images"
        lbl_dir = YOLO_ROOT / ysplit / "labels"
        clean_out = BINARY_ROOT / bsplit / "clean"
        contam_out = BINARY_ROOT / bsplit / "contaminated"
        clean_out.mkdir(parents=True, exist_ok=True)
        contam_out.mkdir(parents=True, exist_ok=True)

        n_clean = n_contam = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if is_contaminated(lbl_path):
                shutil.copy2(img_path, contam_out / img_path.name)
                n_contam += 1
            else:
                shutil.copy2(img_path, clean_out / img_path.name)
                n_clean += 1

        print(f"[{bsplit:5}] clean: {n_clean:5} | contaminated: {n_contam:5}")

    print("\nBinary dataset rebuilt under:", BINARY_ROOT)


if __name__ == "__main__":
    main()
