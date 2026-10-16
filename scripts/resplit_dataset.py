from pathlib import Path
import shutil
import random
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path("data/solar_final")
DESTINATION = Path("data/solar_final_resplit")

SEED = 42

SPLITS = {
    "train": 0.80,
    "valid": 0.10,
    "test": 0.10
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ============================================================
# FUNCTIONS
# ============================================================

def get_image_type(label_path):
    """
    Class 0 = debris
    Class 1 = dust

    Returns:
        negative
        debris
        dust
        mixed
    """

    if not label_path.exists():
        return "negative"

    try:
        lines = label_path.read_text().strip().splitlines()
    except Exception:
        return "negative"

    if not lines:
        return "negative"

    classes = set()

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        try:
            class_id = int(float(parts[0]))
            classes.add(class_id)
        except ValueError:
            continue

    has_debris = 0 in classes
    has_dust = 1 in classes

    if has_debris and has_dust:
        return "mixed"
    elif has_debris:
        return "debris"
    elif has_dust:
        return "dust"
    else:
        return "negative"


def find_all_images():
    """
    Collect all images from the existing
    train, valid and test folders.
    """

    items = []

    for split in ["train", "valid", "test"]:

        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        if not image_dir.exists():
            print(f"WARNING: Missing folder: {image_dir}")
            continue

        for image_path in image_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            image_type = get_image_type(label_path)

            items.append({
                "image": image_path,
                "label": label_path,
                "type": image_type
            })

    return items


def prepare_destination():
    """
    Delete old resplit dataset if it exists,
    then create fresh folder structure.
    """

    if DESTINATION.exists():
        print(f"\nRemoving old folder: {DESTINATION}")
        shutil.rmtree(DESTINATION)

    for split in SPLITS:
        (DESTINATION / split / "images").mkdir(
            parents=True,
            exist_ok=True
        )
        (DESTINATION / split / "labels").mkdir(
            parents=True,
            exist_ok=True
        )


def calculate_split_counts(total):
    """
    Calculate approximately:
    80% train
    10% valid
    10% test
    """

    train_count = int(total * 0.80)
    valid_count = int(total * 0.10)

    test_count = total - train_count - valid_count

    return train_count, valid_count, test_count


def copy_item(item, split):
    """
    Copy image and label into the new split.
    """

    source_image = item["image"]
    source_label = item["label"]

    destination_image = (
        DESTINATION / split / "images" / source_image.name
    )

    destination_label = (
        DESTINATION / split / "labels" /
        f"{source_image.stem}.txt"
    )

    shutil.copy2(source_image, destination_image)

    # Every image gets a label file.
    # Negative images get an empty .txt file.
    if source_label.exists():
        shutil.copy2(source_label, destination_label)
    else:
        destination_label.touch()


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    print("=" * 60)
    print("SOLAR SWEEPER - STRATIFIED DATASET RESPLIT")
    print("=" * 60)

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    print("\nScanning existing dataset...")

    items = find_all_images()

    print(f"Total images found: {len(items)}")

    # --------------------------------------------------------
    # Group by annotation type
    # --------------------------------------------------------

    groups = defaultdict(list)

    for item in items:
        groups[item["type"]].append(item)

    print("\nCurrent dataset categories:")

    for category in ["negative", "debris", "dust", "mixed"]:
        print(f"{category:10}: {len(groups[category])}")

    # --------------------------------------------------------
    # Prepare destination
    # --------------------------------------------------------

    prepare_destination()

    # --------------------------------------------------------
    # Split each category separately
    # --------------------------------------------------------

    final_counts = {
        "train": defaultdict(int),
        "valid": defaultdict(int),
        "test": defaultdict(int)
    }

    for category in ["negative", "debris", "dust", "mixed"]:

        category_items = groups[category]

        random.shuffle(category_items)

        total = len(category_items)

        train_count, valid_count, test_count = \
            calculate_split_counts(total)

        train_items = category_items[:train_count]

        valid_items = category_items[
            train_count:train_count + valid_count
        ]

        test_items = category_items[
            train_count + valid_count:
        ]

        print(f"\nSplitting {category}:")
        print(f"  Train: {len(train_items)}")
        print(f"  Valid: {len(valid_items)}")
        print(f"  Test : {len(test_items)}")

        for item in train_items:
            copy_item(item, "train")
            final_counts["train"][category] += 1

        for item in valid_items:
            copy_item(item, "valid")
            final_counts["valid"][category] += 1

        for item in test_items:
            copy_item(item, "test")
            final_counts["test"][category] += 1

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("NEW DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    for split in ["train", "valid", "test"]:

        total = sum(final_counts[split].values())

        print(f"\n{split.upper()}")

        print(f"  Total    : {total}")
        print(f"  Negative : {final_counts[split]['negative']}")
        print(f"  Debris   : {final_counts[split]['debris']}")
        print(f"  Dust     : {final_counts[split]['dust']}")
        print(f"  Mixed    : {final_counts[split]['mixed']}")

    print("\nNew dataset location:")
    print(DESTINATION.resolve())

    print("\nIMPORTANT:")
    print("Your original data/solar_final folder was NOT modified.")


if __name__ == "__main__":
    main()