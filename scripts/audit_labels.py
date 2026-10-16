"""Quantify label-quality problems in data/solar_grouped.

Reports exact duplicate boxes, near-duplicate boxes (same class, IoU >= 0.9),
per-split class counts, and box-size distribution so YOLO improvements can be
prioritised against measured noise rather than guesswork.
"""
from collections import Counter
from pathlib import Path

ROOT = Path("data/solar_grouped")
SPLITS = ("train", "valid", "test")
NEAR_IOU = 0.9


def read_boxes(txt):
    out = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                out.append(tuple(float(p) for p in parts[:5]))
            except ValueError:
                pass
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = a[3] * a[4] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def main():
    grand_exact = grand_near = grand_boxes = 0
    sizes = []
    for split in SPLITS:
        label_dir = ROOT / split / "labels"
        if not label_dir.is_dir():
            continue
        cls_count = Counter()
        exact_dup = near_dup = n_boxes = 0
        files_with_dups = 0
        n_files = 0
        for txt in label_dir.glob("*.txt"):
            n_files += 1
            boxes = read_boxes(txt)
            n_boxes += len(boxes)
            for b in boxes:
                cls_count[int(b[0])] += 1
                sizes.append(b[3] * b[4])
            exact = len(boxes) - len(set(boxes))
            exact_dup += exact
            near = 0
            uniq = sorted(set(boxes))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    if uniq[i][0] == uniq[j][0] and iou(uniq[i], uniq[j]) >= NEAR_IOU:
                        near += 1
            near_dup += near
            if exact or near:
                files_with_dups += 1
        grand_exact += exact_dup
        grand_near += near_dup
        grand_boxes += n_boxes
        print(f"[{split}] files={n_files} boxes={n_boxes} "
              f"exact_dups={exact_dup} near_dups(>={NEAR_IOU})={near_dup} "
              f"files_affected={files_with_dups}")
        print(f"         class_counts={dict(sorted(cls_count.items()))}")

    print(f"\nTOTAL boxes={grand_boxes} exact_dups={grand_exact} "
          f"({100 * grand_exact / max(grand_boxes,1):.2f}%) near_dups={grand_near}")

    if sizes:
        sizes.sort()
        n = len(sizes)
        small = sum(1 for s in sizes if s < 0.001)
        med = sum(1 for s in sizes if 0.001 <= s < 0.01)
        large = sum(1 for s in sizes if s >= 0.01)
        full = sum(1 for s in sizes if s >= 0.9)
        print(f"box area (normalised) median={sizes[n//2]:.5f} p90={sizes[int(n*0.9)]:.4f} "
              f"max={sizes[-1]:.4f}")
        print(f"  <0.1% of frame (tiny)   : {small} ({100*small/n:.1f}%)")
        print(f"  0.1-1% of frame (small) : {med} ({100*med/n:.1f}%)")
        print(f"  >1% of frame (medium+)  : {large} ({100*large/n:.1f}%)")
        print(f"  >=90% of frame (full)   : {full}")


if __name__ == "__main__":
    main()
