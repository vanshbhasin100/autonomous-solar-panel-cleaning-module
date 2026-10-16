from pathlib import Path
import random
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"C:\Users\VANSH\Projects\solar-sweeper-ai\data\solar_final_resplit"
)

OUTPUT = Path(
    r"C:\Users\VANSH\Projects\solar-sweeper-ai\data\ground_truth_audit"
)

SPLIT = "train"

# Number of random annotated images to inspect
NUM_IMAGES = 100

CLASS_NAMES = {
    0: "debris",
    1: "dust"
}

# ============================================================
# HELPERS
# ============================================================

def find_image(images_dir, stem):

    extensions = [
        ".jpg", ".jpeg", ".png",
        ".JPG", ".JPEG", ".PNG"
    ]

    for ext in extensions:
        image_path = images_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


def get_box_area(parts):

    if len(parts) != 5:
        return 0

    try:
        w = float(parts[3])
        h = float(parts[4])

        return w * h

    except ValueError:
        return 0


def draw_annotations(image, label_path):

    height, width = image.shape[:2]

    if not label_path.exists():
        return image, []

    classes_found = []

    with open(label_path, "r") as file:
        lines = file.readlines()

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

        if class_id not in CLASS_NAMES:
            continue

        classes_found.append(class_id)

        # YOLO normalized coordinates -> pixel coordinates
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        class_name = CLASS_NAMES[class_id]
        area = box_width * box_height

        # Red = debris
        # Yellow = dust
        if class_id == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{class_name} | area={area:.3f}"

        cv2.putText(
            image,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image, classes_found


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SOLAR SWEEPER - GROUND TRUTH ANNOTATION AUDIT")
    print("=" * 65)

    images_dir = DATASET / SPLIT / "images"
    labels_dir = DATASET / SPLIT / "labels"

    debris_output = OUTPUT / SPLIT / "debris"
    dust_output = OUTPUT / SPLIT / "dust"
    mixed_output = OUTPUT / SPLIT / "mixed"
    large_output = OUTPUT / SPLIT / "large_boxes"

    for folder in [
        debris_output,
        dust_output,
        mixed_output,
        large_output
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    label_files = list(labels_dir.glob("*.txt"))

    # Only images that actually contain annotations
    annotated_files = []

    for label_path in label_files:

        content = label_path.read_text().strip()

        if content:
            annotated_files.append(label_path)

    print(f"\nAnnotated training images found: {len(annotated_files)}")

    random.seed(42)

    if len(annotated_files) > NUM_IMAGES:
        selected_files = random.sample(
            annotated_files,
            NUM_IMAGES
        )
    else:
        selected_files = annotated_files

    debris_count = 0
    dust_count = 0
    mixed_count = 0
    large_count = 0

    for label_path in selected_files:

        image_path = find_image(
            images_dir,
            label_path.stem
        )

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        annotated_image, classes_found = draw_annotations(
            image.copy(),
            label_path
        )

        class_set = set(classes_found)

        # Decide category
        if class_set == {0}:
            output_folder = debris_output
            debris_count += 1

        elif class_set == {1}:
            output_folder = dust_output
            dust_count += 1

        else:
            output_folder = mixed_output
            mixed_count += 1

        # Save normal categorized image
        output_path = (
            output_folder /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            annotated_image
        )

        # Check for suspiciously large boxes
        lines = label_path.read_text().strip().splitlines()

        has_large_box = False

        for line in lines:

            parts = line.strip().split()

            area = get_box_area(parts)

            # Box covers more than 30% of image
            if area > 0.30:
                has_large_box = True
                break

        if has_large_box:

            cv2.imwrite(
                str(large_output / image_path.name),
                annotated_image
            )

            large_count += 1

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)

    print(f"\nRandom images inspected : {len(selected_files)}")
    print(f"Debris-only images      : {debris_count}")
    print(f"Dust-only images        : {dust_count}")
    print(f"Mixed images            : {mixed_count}")
    print(f"Large-box images        : {large_count}")

    print("\nOpen this folder:")
    print(OUTPUT)


if __name__ == "__main__":
    main()