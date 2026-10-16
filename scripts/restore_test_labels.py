import os
import shutil

# Define paths relative to the project root
MASTER_TEST_LABELS = r"data\solar_final_resplit_labels_backup\test\labels"  # or direct from solar_final if structured
MASTER_SOURCE = r"data\solar_final"
CURRENT_TEST_LABELS = r"data\solar_final_resplit\test\labels"

# Verify current files and restore
if not os.path.exists(CURRENT_TEST_LABELS):
    print(f"Error: Target path {CURRENT_TEST_LABELS} not found.")
else:
    print(f"Target test labels folder located: {CURRENT_TEST_LABELS}")
    # Let's verify file count
    current_files = os.listdir(CURRENT_TEST_LABELS)
    print(f"Current test label files count: {len(current_files)}")

print(
    "\nReady to verify master source. Please ensure data/solar_final path is accessible."
)