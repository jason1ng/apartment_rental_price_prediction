import os
import shutil
import kagglehub

DATASET = "shashanks1202/apartment-rent-data"
DATA_DIR = "data"

cache_path = kagglehub.dataset_download(DATASET)

os.makedirs(DATA_DIR, exist_ok=True)

for item in os.listdir(cache_path):
    src = os.path.join(cache_path, item)
    dst = os.path.join(DATA_DIR, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied folder: {item}")
    else:
        shutil.copy2(src, dst)
        print(f"Copied file: {item}")

print("Done!")