"""Stream the newest Ultralytics results.csv into TensorBoard event files.

Runs started before tensorboard was installed write no native TB events, so
this polls runs/train for the most recently updated solar_nano* results.csv
and mirrors new epoch rows as TB scalars for the dashboard at localhost:6006.
If a newer run is started later, the bridge follows it automatically.
"""
import csv
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

RUNS_ROOT = Path(r"C:/Users/VANSH/runs/detect/runs/train")
OUT = RUNS_ROOT / "tb_bridge_scalars"
POLL = 15

TAGS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "train/box_loss": "train/box_loss",
    "train/cls_loss": "train/cls_loss",
    "train/dfl_loss": "train/dfl_loss",
    "val/box_loss": "val/box_loss",
    "val/cls_loss": "val/cls_loss",
    "val/dfl_loss": "val/dfl_loss",
}


def newest_results_csv():
    candidates = [p for p in RUNS_ROOT.glob("solar_nano*/results.csv") if p.exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def main():
    writer = SummaryWriter(str(OUT))
    seen = {}
    print(f"bridging newest results.csv under {RUNS_ROOT} -> {OUT}")
    while True:
        csv_path = newest_results_csv()
        if csv_path:
            rows = list(csv.DictReader(csv_path.open()))
            n = seen.get(csv_path, 0)
            if len(rows) > n:
                for row in rows[n:]:
                    step = int(row["epoch"])
                    for col, tag in TAGS.items():
                        v = row.get(col, "")
                        if v != "":
                            try:
                                writer.add_scalar(tag, float(v), step)
                            except ValueError:
                                pass
                writer.flush()
                print(f"synced epoch {int(rows[-1]['epoch'])} from {csv_path.parent.name}")
                seen[csv_path] = len(rows)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
