from ultralytics import YOLO

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="data/solar_final_resplit/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,

    project="runs/solar_sweeper",
    name="yolov8n_baseline",

    patience=20,
    pretrained=True,
    optimizer="auto",
    seed=42,

    save=True,
    plots=True,
)