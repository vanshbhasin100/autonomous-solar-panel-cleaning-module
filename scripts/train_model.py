"""Train the final YOLOv8-nano detector on the leakage-free grouped dataset."""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/solar_grouped/data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        project="runs/train",
        name="solar_nano_grouped_v2",

        pretrained=True,
        patience=50,
        save=True,
        plots=True,
        seed=42,
    )

    print("\nTraining complete!")
    print("Best model: runs/train/solar_nano_grouped_v2/weights/best.pt")


if __name__ == "__main__":
    main()
