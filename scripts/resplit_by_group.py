"""Leakage-free re-split for the solar sweeper YOLO dataset.

The previous split (resplit_dataset.py) shuffled individual files, so Roboflow
augmentations (.rf.<hash>) of the SAME original photo ended up in different
splits -> ~80% train/test leakage. This script pools every file, groups all
augmentations by their original source image, and splits at the GROUP level so
a source image lands in exactly one split.

Non-destructive: reads from solar_final_resplit, writes to solar_grouped.
"""
import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

BASE = Path("data")
SOURCE = BASE / "solar_final_resplit"
DEST = BASE / "solar_grouped"
SEED = 42
SPLITS = {"train": 0.80, "valid": 0.10, "test": 0.10}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def src_id(filename: str) -> str:
    """Original source image id, stripping augmentation + extension."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"_jpg$", "", name)
    return name


def category(label_path: Path) -> str:
    if not label_path.exists():
        return "negative"
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return "negative"
    classes = set()
    for line in lines:
        parts = line.split()
        if parts:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    if 0 in classes and 1 in classes:
        return "mixed"
    if 0 in classes:
        return "debris"
    if 1 in classes:
        return "dust"
    return "negative"


def pool_items():
    """Collect every (image, label) across the old splits, deduped by filename."""
    items = {}
    for split in ["train", "valid", "test"]:
        img_dir = SOURCE / split / "images"
        lbl_dir = SOURCE / split / "labels"
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in IMG_EXT:
                continue
            if p.name in items:
                continue
            lbl = lbl_dir / f"{p.stem}.txt"
            items[p.name] = {"image": p, "label": lbl, "src": src_id(p.name)}
    return list(items.values())


def main():
    random.seed(SEED)
    items = pool_items()
    print(f"Pooled {len(items)} unique files from {SOURCE}")

    # group by source image; category is consistent across a source's augmentations
    groups = defaultdict(list)
    for it in items:
        groups[it["src"]].append(it)
    print(f"Grouped into {len(groups)} unique source images")

    # stratify groups by category
    by_cat = defaultdict(list)
    for src, members in groups.items():
        by_cat[category(members[0]["label"])].append(src)
    for c in sorted(by_cat):
        print(f"  {c:9}: {len(by_cat[c])} source images")

    if DEST.exists():
        print(f"\nRemoving existing {DEST}")
        shutil.rmtree(DEST)
    for split in SPLITS:
        (DEST / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST / split / "labels").mkdir(parents=True, exist_ok=True)

    assignment = {}  # src -> split
    for c, srcs in by_cat.items():
        srcs = sorted(srcs)
        random.shuffle(srcs)
        n = len(srcs)
        n_tr = int(n * SPLITS["train"])
        n_va = int(n * SPLITS["valid"])
        for s in srcs[:n_tr]:
            assignment[s] = "train"
        for s in srcs[n_tr:n_tr + n_va]:
            assignment[s] = "valid"
        for s in srcs[n_tr + n_va:]:
            assignment[s] = "test"

    counts = defaultdict(int)
    for it in items:
        split = assignment[it["src"]]
        shutil.copy2(it["image"], DEST / split / "images" / it["image"].name)
        dst_lbl = DEST / split / "labels" / f"{it['image'].stem}.txt"
        if it["label"].exists():
            shutil.copy2(it["label"], dst_lbl)
        else:
            dst_lbl.touch()
        counts[split] += 1

    # write a data.yaml for the new split
    (DEST / "data.yaml").write_text(
        f"path: {DEST.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n  0: debris\n  1: dirt\n"
    )

    print("\nNew grouped split:")
    for split in ["train", "valid", "test"]:
        print(f"  {split:6}: {counts[split]} files")
    print(f"\nWrote {DEST/'data.yaml'}")


if __name__ == "__main__":
    main()
