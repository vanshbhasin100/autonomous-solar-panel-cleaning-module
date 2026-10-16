import os
import shutil

# Paths
master_dir = os.path.join("data", "solar_final")
test_images_dir = os.path.join("data", "solar_final_resplit", "test", "images")
test_labels_dir = os.path.join("data", "solar_final_resplit", "test", "labels")

# 1. Verify source and destination exist
if not os.path.exists(master_dir):
    raise FileNotFoundError(f"Master directory '{master_dir}' does not exist.")
if not os.path.exists(test_images_dir):
    raise FileNotFoundError(f"Test images directory '{test_images_dir}' does not exist.")
if not os.path.exists(test_labels_dir):
    raise FileNotFoundError(f"Test labels directory '{test_labels_dir}' does not exist.")

# 2. Find all master label files (search recursively across solar_final)
master_labels_map = {}
for root, dirs, files in os.walk(master_dir):
    if os.path.basename(root) == "labels":
        for f in files:
            if f.endswith(".txt"):
                master_labels_map[f] = os.path.join(root, f)

print(f"Found {len(master_labels_map)} label files in master dataset '{master_dir}'.")

# 3. Match against test images
test_image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(test_image_files)} test images in '{test_images_dir}'.")

restored_count = 0
missing_count = 0

for img_file in test_image_files:
    base_name = os.path.splitext(img_file)[0]
    label_file = f"{base_name}.txt"
    target_path = os.path.join(test_labels_dir, label_file)
    
    if label_file in master_labels_map:
        source_path = master_labels_map[label_file]
        shutil.copy2(source_path, target_path)
        restored_count += 1
    else:
        # If it's a negative/clean image, ensure an empty txt or handle accordingly
        missing_count += 1

print(f"\nRestoration Complete:")
print(f"  - Restored/Overwritten: {restored_count} label files from master.")
print(f"  - Unmatched in master: {missing_count} files.")