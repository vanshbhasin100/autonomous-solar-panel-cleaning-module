from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent

DATASETS = {
    "dataset_1": PROJECT / "raw_datasets" / "dusty_bird_droppings" / "extracted",
    "dataset_2": PROJECT / "raw_datasets" / "solar_panel_detect" / "extracted",
}

for dataset_name, dataset_path in DATASETS.items():

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    length_counts = Counter()
    examples = {}

    for split in ["train", "valid", "test"]:

        label_dir = dataset_path / split / "labels"

        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):

            for line in label_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines():

                parts = line.strip().split()

                if not parts:
                    continue

                length = len(parts)

                length_counts[length] += 1

                if length not in examples:
                    examples[length] = {
                        "file": str(label_file.relative_to(PROJECT)),
                        "line": line[:200]
                    }

    print("\nNumber of values per annotation:")

    for length, count in sorted(length_counts.items()):
        print(f"{length} values: {count}")

    print("\nExamples of unusual annotation formats:")

    for length in sorted(examples):

        if length != 5:

            print(f"\n--- {length} values ---")
            print("File:", examples[length]["file"])
            print("Line:", examples[length]["line"])