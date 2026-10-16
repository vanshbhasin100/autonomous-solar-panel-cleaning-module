"""Remove degenerate bounding boxes (pure-noise annotations).

A box is degenerate when its shorter side is under ~5px at the 640px training
size (min(w,h) < 0.0078 in normalized coords). These trace watermark edges or
noise and teach the detector nothing. Full-frame boxes are intentionally kept.

Operates in place on data/solar_grouped. Rewrites label files, dropping only
the degenerate lines.
"""
import glob
import os
from pathlib import Path

BASE = Path("data/solar_grouped")
MIN_SIDE = 5 / 640  # ~5px at training resolution


def main():
    removed_total = 0
    emptied = 0
    for split in ["train", "valid", "test"]:
        removed = 0
        for lbl in glob.glob(str(BASE / split / "labels" / "*.txt")):
            lines = Path(lbl).read_text().strip().splitlines()
            if not lines:
                continue
            kept = []
            for line in lines:
                p = line.split()
                if len(p) != 5:
                    kept.append(line)
                    continue
                try:
                    _, _, _, w, h = p[0], *map(float, p[1:])
                except ValueError:
                    kept.append(line)
                    continue
                if min(w, h) < MIN_SIDE:
                    removed += 1
                else:
                    kept.append(line)
            if removed and len(kept) != len(lines):
                Path(lbl).write_text("\n".join(kept) + ("\n" if kept else ""))
                if not kept:
                    emptied += 1
        removed_total += removed
        print(f"{split:6}: removed {removed} degenerate boxes")
    print(f"\nTotal removed: {removed_total}  (label files now empty: {emptied})")


if __name__ == "__main__":
    main()
