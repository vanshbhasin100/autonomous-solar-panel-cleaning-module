from ultralytics import YOLO


def main():
    model = YOLO(
        r"C:\Users\VANSH\runs\detect\runs\train\solar_sweeper_yolo\weights\best.pt"
    )

    results = model.val(
        data="data/solar_final_resplit/data.yaml",
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        plots=True,
        project="runs/test",
        name="solar_sweeper_test"
    )


if __name__ == "__main__":
    main()