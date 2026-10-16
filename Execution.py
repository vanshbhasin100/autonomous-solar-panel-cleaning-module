"""Solar Sweeper demo: pick photos/videos, get labelled detection boxes.

Opens a file picker, runs the trained YOLOv8n dust/debris detector over each
selected image or video, draws bounding boxes with class + confidence, shows
the result live, and saves annotated copies to demo_output/.

Usage:
    python Execution.py                 # pick files with a dialog
    python Execution.py --camera        # live webcam feed instead
    python Execution.py --conf 0.4      # raise/lower detection threshold
    python Execution.py --model path    # override which weights to load
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OUT_DIR = Path("demo_output")
WINDOW = "Solar Sweeper - Execution (press Q to stop a video)"

MODEL_CANDIDATES = [
    r"C:/Users/VANSH/runs/detect/runs/train/solar_nano_grouped_v2-2/weights/best.pt",
    "runs/train/solar_nano_grouped_v2-2/weights/best.pt",
    r"C:/Users/VANSH/runs/detect/runs/train/solar_nano_det/weights/best.pt",
    "runs/train/solar_nano_det/weights/best.pt",
    "yolov8n.pt",
]


def resolve_model(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"model not found: {p}")
        return p
    for c in MODEL_CANDIDATES:
        if Path(c).exists():
            return Path(c)
    raise SystemExit("no trained weights found; expected one of:\n  "
                     + "\n  ".join(MODEL_CANDIDATES))


def pick_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Select photos and/or videos",
        filetypes=[
            ("Images & videos", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm"),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return [Path(p) for p in paths]


def summarize(results):
    counts = {}
    for r in results:
        for cls in r.boxes.cls.int().tolist():
            name = r.names[cls]
            counts[name] = counts.get(name, 0) + 1
    return counts


def run_image(model, path, conf):
    results = model.predict(str(path), conf=conf, verbose=False)
    frame = results[0].plot()
    out = OUT_DIR / f"{path.stem}_annotated{path.suffix}"
    cv2.imwrite(str(out), frame)
    print(f"  {path.name}: {summarize(results)} -> saved {out}")
    cv2.imshow(WINDOW, frame)
    print("  [image] press any key in the preview window to continue")
    cv2.waitKey(0)
    cv2.destroyWindow(WINDOW)


def run_video(model, path, conf):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  {path.name}: could not open video, skipped")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = OUT_DIR / f"{path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)
        cv2.imshow(WINDOW, annotated)
        n += 1
        if n % 30 == 0:
            print(f"  {path.name}: frame {n} ({n / max(time.time() - t0, 1e-6):.1f} fps)")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  stopped early by user (Q)")
            break
    cap.release()
    writer.release()
    cv2.destroyWindow(WINDOW)
    print(f"  {path.name}: {n} frames -> saved {out}")


def run_camera(model, conf):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam 0")
    print("[camera] press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        cv2.imshow(WINDOW, results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyWindow(WINDOW)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="path to best.pt weights")
    ap.add_argument("--conf", type=float, default=0.1, help="confidence threshold")
    ap.add_argument("--camera", action="store_true", help="use webcam instead of files")
    args = ap.parse_args()

    weights = resolve_model(args.model)
    print(f"loading model: {weights}")
    model = YOLO(str(weights))
    OUT_DIR.mkdir(exist_ok=True)

    if args.camera:
        run_camera(model, args.conf)
        return

    files = pick_files()
    if not files:
        print("no files selected - nothing to do")
        return
    print(f"selected {len(files)} file(s)")
    for path in files:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            run_image(model, path, args.conf)
        elif ext in VIDEO_EXTS:
            run_video(model, path, args.conf)
        else:
            print(f"  {path.name}: unsupported type, skipped")
    print("done. annotated outputs in", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
