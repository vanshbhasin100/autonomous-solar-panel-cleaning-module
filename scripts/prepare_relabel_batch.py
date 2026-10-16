import os
import shutil
import csv

csv_path = os.path.join("data", "ground_truth_audit", "oversized_debris_decisions.csv")
target_dir = os.path.join("data", "relabel_batch_81")
dataset_root = os.path.join("data", "solar_final_resplit")

os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)

# 1. Print CSV header & sample row to inspect
with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"CSV Header columns: {header}")
    for i in range(3):
        try:
            print(f"Sample row {i+1}: {next(reader)}")
        except StopIteration:
            break

# 2. Index all images across train, valid, test in solar_final_resplit
print("\nIndexing images and labels across solar_final_resplit...")
all_images = {}
all_labels = {}

for split in ["train", "valid", "test"]:
    img_split_dir = os.path.join(dataset_root, split, "images")
    lbl_split_dir = os.path.join(dataset_root, split, "labels")
    
    if os.path.exists(img_split_dir):
        for f in os.listdir(img_split_dir):
            base = os.path.splitext(f)[0].lower()
            all_images[base] = os.path.join(img_split_dir, f)
            
    if os.path.exists(lbl_split_dir):
        for f in os.listdir(lbl_split_dir):
            base = os.path.splitext(f)[0].lower()
            all_labels[base] = os.path.join(lbl_split_dir, f)

print(f"Indexed {len(all_images)} total images and {len(all_labels)} total label files.")

# 3. Process matches
copied_count = 0
with open(csv_path, mode="r", encoding="utf-8") as f:
    dict_reader = csv.DictReader(f)
    for row in dict_reader:
        decision = next((v for k, v in row.items() if "decision" in k.lower()), "").strip().lower()
        
        if decision == "flag_manual_fix":
            # Extract possible identifier
            val = next((v for k, v in row.items() if any(x in k.lower() for x in ["file", "img", "name", "image"])), None)
            if not val:
                val = list(row.values())[0]
                
            raw_name = os.path.splitext(os.path.basename(str(val).strip()))[0].lower()
            
            # Match directly or by checking if raw_name is substring/contains base
            matched_base = None
            if raw_name in all_images:
                matched_base = raw_name
            else:
                for base in all_images:
                    if base in raw_name or raw_name in base:
                        matched_base = base
                        break
            
            if matched_base:
                src_img = all_images[matched_base]
                src_lbl = all_labels.get(matched_base)
                
                shutil.copy2(src_img, os.path.join(target_dir, "images", os.path.basename(src_img)))
                if src_lbl and os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(target_dir, "labels", os.path.basename(src_lbl)))
                copied_count += 1

print(f"\nSuccessfully copied {copied_count} of 81 target images into '{target_dir}'.")