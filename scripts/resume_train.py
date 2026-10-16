"""Resume the interrupted YOLOv8n run from its last checkpoint (epoch 55)."""
from ultralytics import YOLO

LAST = r"C:/Users/VANSH/runs/detect/runs/train/solar_nano_grouped/weights/last.pt"

YOLO(LAST).train(resume=True)
