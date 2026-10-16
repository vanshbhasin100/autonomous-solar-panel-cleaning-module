from pathlib import Path
import random
import cv2

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "solar_final"
OUTPUT = PROJECT / "data" / "visualization"

SPLITS = ["train", "valid", "test"]

# Number of images to inspect from each split
SAMPLES_PER_SPLIT = 20

# Class names
CLASS_NAMES = {
    0: "debris",
    1: "dust"
}


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

if OUTPUT.exists():
    import shutil
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DRAW YOLO BOXES
# --------------------------------------------------

def draw_annotations(image_path, label_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return None

    height, width = image.shape[:2]

    # Read label file
    if label_path.exists():
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    else:
        lines = []

    # If image has no annotations, mark it as CLEAN
    if not lines:
        cv2.putText(
            image,
            "NEGATIVE / NO CONTAMINATION LABEL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Draw each annotation
    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
        except ValueError:
            continue

        # Convert normalized YOLO coordinates to pixels
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # Draw label
        label = f"{class_id}: {class_name}"

        cv2.putText(
            image,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return image


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

random.seed(42)

total_saved = 0

for split in SPLITS:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not images:
        print(f"No images found in {split}")
        continue

    sample_count = min(SAMPLES_PER_SPLIT, len(images))

    selected_images = random.sample(
        images,
        sample_count
    )

    split_output = OUTPUT / split
    split_output.mkdir(exist_ok=True)

    print(f"\nProcessing {split}: {sample_count} images")

    for image_path in selected_images:

        label_path = (
            label_dir /
            f"{image_path.stem}.txt"
        )

        annotated_image = draw_annotations(
            image_path,
            label_path
        )

        if annotated_image is not None:

            output_path = (
                split_output /
                image_path.name
            )

            cv2.imwrite(
                str(output_path),
                annotated_image
            )

            total_saved += 1

print("\n" + "=" * 50)
print("VISUALIZATION COMPLETE")
print("=" * 50)
print(f"Images saved: {total_saved}")
print(f"Location: {OUTPUT}")