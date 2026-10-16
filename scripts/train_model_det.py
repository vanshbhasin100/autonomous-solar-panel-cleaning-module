"""Retrain yolov8n on the detection-focused split (background capped to ~10%).

Single-variable A/B vs scripts/train_model.py: only the train split changes
(data/solar_grouped_det); valid/test are identical so mAP compares directly
against the previous run's test mAP50 0.203 / val mAP50 0.208.

Run only when the GPU is free (MobileNet must not be training concurrently).
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/solar_grouped_det/data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        project="runs/train",
        name="solar_nano_det",

        pretrained=True,
        patience=50,
        save=True,
        plots=True,
        seed=42,
    )

    print("\nTraining complete!")
    print("Best model: runs/train/solar_nano_det/weights/best.pt")


if __name__ == "__main__":
    main()
