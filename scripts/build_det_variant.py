"""Build a detection-focused train split to fix the recall collapse.

Diagnosis (2026-10-08): the converged model misses 84% of debris and 70% of
dirt (see val-4/confusion_matrix_normalized.png) while class confusion is ~1%.
43% of train images are background (clean panels), far above the ~0-10%
Ultralytics recommends, which teaches the model to predict nothing.

This script creates data/solar_grouped_det with:
  - train: ALL labeled images + a seeded random ~10%-of-labeled subset of the
    background images (single-variable change)
  - valid / test: hardlinked byte-identical copies, so metrics stay comparable

Original data/solar_grouped is never modified.
"""
import os
import random
import shutil
from pathlib import Path

SRC = Path("data/solar_grouped")
DST = Path("data/solar_grouped_det")
BG_FRACTION = 0.10
SEED = 42


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stem_to_files(split, stem):
    out = []
    img = next(SRC.glob(f"{split}/images/{stem}.*"), None)
    if img:
        out.append(img)
    lbl = SRC / split / "labels" / f"{stem}.txt"
    if lbl.exists():
        out.append(lbl)
    return out


def main():
    random.seed(SEED)
    train_imgs = SRC / "train" / "images"
    stems = sorted(p.stem for p in train_imgs.glob("*"))
    labeled, background = [], []
    for s in stems:
        lbl = SRC / "train" / "labels" / f"{s}.txt"
        has_box = lbl.exists() and lbl.read_text(errors="ignore").strip() != ""
        (labeled if has_box else background).append(s)

    keep_bg = int(BG_FRACTION * len(labeled))
    chosen_bg = random.sample(background, keep_bg)
    dropped = len(background) - keep_bg
    print(f"train: labeled={len(labeled)} background={len(background)} "
          f"-> keep {keep_bg} background, drop {dropped}")

    for split in ("train", "valid", "test"):
        if split == "train":
            use = labeled + chosen_bg
        else:
            use = sorted(p.stem for p in (SRC / split / "images").glob("*"))
        for s in use:
            for f in stem_to_files(split, s):
                rel = f.relative_to(SRC)
                link(f, DST / rel)
        n_img = len(list((DST / split / "images").glob("*")))
        print(f"  {split}: wrote {n_img} images")

    yaml_text = f"""path: {DST.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: debris
  1: dirt
"""
    (DST / "data.yaml").write_text(yaml_text)
    print(f"wrote {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
