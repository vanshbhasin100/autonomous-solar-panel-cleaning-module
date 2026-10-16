from pathlib import Path

DATASET = Path("data/solar_final_resplit")

print("=" * 60)
print("SOLAR SWEEPER - NEGATIVE IMAGE CHECK")
print("=" * 60)

for split in ["train", "valid", "test"]:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*.*"))

    negative_count = 0
    object_count = 0
    missing_labels = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            negative_count += 1
            continue

        content = label_path.read_text().strip()

        if not content:
            negative_count += 1
        else:
            object_count += 1

    print(f"\n{split.upper()}")
    print(f"Total images           : {len(images)}")
    print(f"Negative / clean images: {negative_count}")
    print(f"Images with objects    : {object_count}")
    print(f"Missing label files    : {missing_labels}")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)