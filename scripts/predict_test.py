from ultralytics import YOLO
from pathlib import Path

def main():
    # Load best trained model
    model = YOLO(
        r"C:\Users\VANSH\runs\detect\runs\train\solar_sweeper_yolo\weights\best.pt"
    )

    # Test images folder
    source = Path(
        r"C:\Users\VANSH\Projects\solar-sweeper-ai\data\solar_final_resplit\test\images"
    )

    # Run predictions
    model.predict(
        source=str(source),
        conf=0.25,
        save=True,
        project=r"C:\Users\VANSH\Projects\solar-sweeper-ai\runs\predict",
        name="test_predictions",
        exist_ok=True
    )

    print("\nPredictions complete!")
    print(r"Check: C:\Users\VANSH\Projects\solar-sweeper-ai\runs\predict\test_predictions")


if __name__ == "__main__":
    main()