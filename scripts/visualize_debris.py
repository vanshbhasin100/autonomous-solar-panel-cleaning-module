from pathlib import Path
import random
import cv2

# Project paths
DATASET = Path("data/solar_final")
OUTPUT = Path("data/visualization_debris")

# Class mapping from our final dataset
CLASS_NAMES = {
    0: "debris",
    1: "dust"
}

# Number of debris images to visualize from each split
MAX_IMAGES = 10


def get_image_files(images_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in extensions
    ]


def get_label_path(image_path, split):
    return DATASET / split / "labels" / f"{image_path.stem}.txt"


def read_labels(label_path):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                boxes.append(
                    (class_id, x_center, y_center, width, height)
                )

            except ValueError:
                continue

    return boxes


def draw_boxes(image, boxes):
    height, width = image.shape[:2]

    for class_id, x_center, y_center, box_width, box_height in boxes:

        # Convert YOLO normalized coordinates to pixels
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Red for debris
        color = (0, 0, 255)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"{class_id}: {CLASS_NAMES.get(class_id, 'unknown')}"

        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return image


def process_split(split):
    images_dir = DATASET / split / "images"
    output_dir = OUTPUT / split

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(images_dir)

    debris_images = []
    debris_boxes = 0

    print(f"\nScanning {split}...")

    for image_path in image_files:

        label_path = get_label_path(image_path, split)
        boxes = read_labels(label_path)

        # Find class 0 = debris
        debris_only = [
            box for box in boxes
            if box[0] == 0
        ]

        if debris_only:
            debris_images.append(
                (image_path, boxes)
            )

            debris_boxes += len(debris_only)

    print(f"Total images: {len(image_files)}")
    print(f"Images containing debris: {len(debris_images)}")
    print(f"Total debris boxes: {debris_boxes}")

    if not debris_images:
        print("WARNING: NO DEBRIS IMAGES FOUND!")
        return

    # Randomly select images
    selected = random.sample(
        debris_images,
        min(MAX_IMAGES, len(debris_images))
    )

    for i, (image_path, boxes) in enumerate(selected, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        # Draw ALL boxes in the image
        image = draw_boxes(image, boxes)

        output_path = output_dir / f"debris_{i}_{image_path.name}"

        cv2.imwrite(
            str(output_path),
            image
        )

    print(f"Saved {len(selected)} images to:")
    print(output_dir)


def main():

    print("=" * 55)
    print("SOLAR SWEEPER - DEBRIS DATASET CHECK")
    print("=" * 55)

    for split in ["train", "valid", "test"]:
        process_split(split)

    print("\n" + "=" * 55)
    print("CHECK COMPLETE")
    print("=" * 55)

    print("\nOpen this folder to see the results:")
    print(OUTPUT)


if __name__ == "__main__":
    main()