from pathlib import Path
import cv2
import random

DATASET = Path("data/solar_final_resplit")
OUTPUT = Path("data/visualization_resplit")

CLASSES = {
    0: ("debris", (0, 0, 255)),
    1: ("dust", (255, 0, 0))
}

SAMPLES_PER_CLASS = 10


def get_class_ids(label_path):
    if not label_path.exists():
        return set()

    classes = set()

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) >= 5:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                pass

    return classes


def draw_boxes(image, label_path):
    height, width = image.shape[:2]

    if not label_path.exists():
        return image

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
        except ValueError:
            continue

        if class_id not in CLASSES:
            continue

        class_name, color = CLASSES[class_id]

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label = f"{class_id}: {class_name}"

        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    return image


def visualize_split(split_name):
    image_dir = DATASET / split_name / "images"
    label_dir = DATASET / split_name / "labels"

    debris_images = []
    dust_images = []
    negative_images = []

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        class_ids = get_class_ids(label_path)

        if 0 in class_ids:
            debris_images.append(image_path)
        elif 1 in class_ids:
            dust_images.append(image_path)
        else:
            negative_images.append(image_path)

    print(f"\n{split_name.upper()}")
    print(f"Debris images   : {len(debris_images)}")
    print(f"Dust images     : {len(dust_images)}")
    print(f"Negative images : {len(negative_images)}")

    categories = {
        "debris": debris_images,
        "dust": dust_images,
        "negative": negative_images
    }

    for category, images in categories.items():

        output_dir = OUTPUT / split_name / category
        output_dir.mkdir(parents=True, exist_ok=True)

        if not images:
            continue

        random.shuffle(images)

        selected = images[:SAMPLES_PER_CLASS]

        for i, image_path in enumerate(selected):

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if category != "negative":
                image = draw_boxes(image, label_path)
            else:
                cv2.putText(
                    image,
                    "NEGATIVE / NO CONTAMINATION",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

            output_path = (
                output_dir /
                f"{category}_{i + 1}_{image_path.name}"
            )

            cv2.imwrite(str(output_path), image)


def main():

    print("=" * 60)
    print("SOLAR SWEEPER - RESPLIT DATASET VISUAL CHECK")
    print("=" * 60)

    random.seed(42)

    # We especially care about validation and test
    visualize_split("valid")
    visualize_split("test")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    print(f"\nOpen:\n{OUTPUT.resolve()}")


if __name__ == "__main__":
    main()