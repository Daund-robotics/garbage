import os
import shutil

DATASET_ROOT = r"D:\garbage updated\dataset"
IMAGES_TRAIN = os.path.join(DATASET_ROOT, "images", "train")
IMAGES_VAL = os.path.join(DATASET_ROOT, "images", "val")
LABELS_TRAIN = os.path.join(DATASET_ROOT, "labels", "train")
LABELS_VAL = os.path.join(DATASET_ROOT, "labels", "val")

def create_structure():
    for path in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

def move_images():
    # Move all .png files from root to images/train
    files = [f for f in os.listdir(DATASET_ROOT) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for f in files:
        src = os.path.join(DATASET_ROOT, f)
        dst = os.path.join(IMAGES_TRAIN, f)
        shutil.move(src, dst)
        print(f"Moved {f} -> {IMAGES_TRAIN}")

def create_yaml():
    yaml_content = f"""path: {DATASET_ROOT}
train: images/train
val: images/train  # Use train images for validation for now
nc: 1
names: ['custom_garbage']
"""
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created {yaml_path}")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        create_structure()
        move_images()
        create_yaml()
        print("Dataset setup complete.")
    else:
        print(f"Error: {DATASET_ROOT} does not exist.")
