import os

data_dir = "data"

if not os.path.exists(data_dir):
    print(f"Directory '{data_dir}' does not exist.")
else:
    print(f"Contents of '{data_dir}':")
    for item in os.listdir(data_dir):
        full_path = os.path.join(data_dir, item)
        if os.path.isdir(full_path):
            print(f"  [FOLDER] {item}")
        else:
            print(f"  [FILE]   {item}")