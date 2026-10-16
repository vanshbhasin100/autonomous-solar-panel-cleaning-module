from pathlib import Path
import shutil
import hashlib

PROJECT = Path(__file__).resolve().parent.parent

DATASET_1 = PROJECT / "raw_datasets" / "dusty_bird_droppings" / "extracted"
DATASET_2 = PROJECT / "raw_datasets" / "solar_panel_detect" / "extracted"

OUTPUT = PROJECT / "data" / "solar_final"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# CREATE CLEAN OUTPUT
# --------------------------------------------------

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in SPLITS:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DUPLICATE DETECTION
# --------------------------------------------------

def image_hash(image_path):
    hasher = hashlib.sha256()

    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# --------------------------------------------------
# NORMAL YOLO BOX
# --------------------------------------------------

def parse_box(parts):
    if len(parts) != 5:
        return None

    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        if (
            0 <= x <= 1 and
            0 <= y <= 1 and
            0 < w <= 1 and
            0 < h <= 1
        ):
            return class_id, x, y, w, h

    except ValueError:
        pass

    return None


# --------------------------------------------------
# YOLO POLYGON -> YOLO BOUNDING BOX
# --------------------------------------------------

def polygon_to_box(parts):
    try:
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Need x,y coordinate pairs
        if len(coords) < 6 or len(coords) % 2 != 0:
            return None

        xs = coords[0::2]
        ys = coords[1::2]

        # Coordinates must be normalized
        if any(x < 0 or x > 1 for x in xs):
            return None

        if any(y < 0 or y > 1 for y in ys):
            return None

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            return None

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        return class_id, x_center, y_center, width, height

    except (ValueError, IndexError):
        return None


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

def process_dataset(
    dataset_path,
    dataset_number,
    class_mapping,
    seen_hashes,
    stats
):

    for split in SPLITS:

        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            stats["images_seen"] += 1

            # Skip exact duplicates
            file_hash = image_hash(image_path)

            if file_hash in seen_hashes:
                stats["duplicates_skipped"] += 1
                continue

            seen_hashes.add(file_hash)

            new_name = (
                f"ds{dataset_number}_{split}_{image_path.name}"
            )

            output_image = (
                OUTPUT / split / "images" / new_name
            )

            output_label = (
                OUTPUT
                / split
                / "labels"
                / f"{Path(new_name).stem}.txt"
            )

            source_label = (
                label_dir / f"{image_path.stem}.txt"
            )

            final_lines = []

            if source_label.exists():

                lines = source_label.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()

                for line in lines:

                    parts = line.strip().split()

                    if not parts:
                        continue

                    # Try normal YOLO bounding box first
                    result = parse_box(parts)

                    annotation_type = "box"

                    # If not a normal box, try polygon
                    if result is None:
                        result = polygon_to_box(parts)
                        annotation_type = "polygon"

                    # Truly invalid annotation
                    if result is None:
                        stats["invalid_labels_skipped"] += 1
                        continue

                    old_class, x, y, w, h = result

                    # Map relevant classes
                    if old_class in class_mapping:

                        new_class = class_mapping[old_class]

                        final_lines.append(
                            f"{new_class} "
                            f"{x:.6f} "
                            f"{y:.6f} "
                            f"{w:.6f} "
                            f"{h:.6f}"
                        )

                        if annotation_type == "polygon":
                            stats["polygons_converted"] += 1

                        if new_class == 0:
                            stats["debris_boxes"] += 1
                        else:
                            stats["dust_boxes"] += 1

                    else:
                        # Dataset 2 clean class
                        stats["clean_annotations_removed"] += 1

            shutil.copy2(image_path, output_image)

            # Empty label file = negative example
            output_label.write_text(
                "\n".join(final_lines),
                encoding="utf-8"
            )

            stats["images_copied"] += 1

            if not final_lines:
                stats["negative_images"] += 1


# --------------------------------------------------
# RUN
# --------------------------------------------------

stats = {
    "images_seen": 0,
    "images_copied": 0,
    "duplicates_skipped": 0,
    "invalid_labels_skipped": 0,
    "polygons_converted": 0,
    "clean_annotations_removed": 0,
    "negative_images": 0,
    "debris_boxes": 0,
    "dust_boxes": 0,
}

seen_hashes = set()

print("\nProcessing Dataset 1...")

process_dataset(
    DATASET_1,
    dataset_number=1,
    class_mapping={
        0: 0,  # bird-droppings -> debris
        1: 1,  # dusdusty -> dust
    },
    seen_hashes=seen_hashes,
    stats=stats,
)

print("Processing Dataset 2...")

process_dataset(
    DATASET_2,
    dataset_number=2,
    class_mapping={
        0: 0,  # bird-drop -> debris
        2: 1,  # dust -> dust
    },
    seen_hashes=seen_hashes,
    stats=stats,
)


# --------------------------------------------------
# CREATE DATA.YAML
# --------------------------------------------------

yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: debris
  1: dust
"""

(OUTPUT / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL DATASET CREATED")
print("=" * 55)

for key, value in stats.items():
    print(f"{key}: {value}")

print("\nFinal dataset:")

for split in SPLITS:

    image_count = len(
        list((OUTPUT / split / "images").glob("*"))
    )

    label_count = len(
        list((OUTPUT / split / "labels").glob("*.txt"))
    )

    print(
        f"{split}: "
        f"{image_count} images, "
        f"{label_count} label files"
    )