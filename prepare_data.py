import os
import shutil
import random

# Point this at the extracted dataset
SOURCE_TRAIN = r"C:\Users\rsree\Downloads\archive\fruit_ripeness_dataset\archive (1)\dataset\dataset\train"
SOURCE_TEST  = r"C:\Users\rsree\Downloads\archive\fruit_ripeness_dataset\archive (1)\dataset\dataset\test"
DEST         = r"C:\Users\rsree\Desktop\ML model\fruit_ripeness_classifier\data"

# Map original folder names → our 3 classes
CLASS_MAP = {
    "freshapples":   "ripe",
    "freshbanana":   "ripe",
    "freshoranges":  "ripe",
    "rottenapples":  "rotten",
    "rottenbanana":  "rotten",
    "rottenoranges": "rotten",
    "unripe apple":  "unripe",
    "unripe banana": "unripe",
    "unripe orange": "unripe",
}

VAL_SPLIT = 0.15
random.seed(42)

def copy_images(source_dir, split):
    for original_class, mapped_class in CLASS_MAP.items():
        src_folder = os.path.join(source_dir, original_class)
        if not os.path.exists(src_folder):
            print(f"Skipping missing folder: {src_folder}")
            continue

        images = [f for f in os.listdir(src_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if split == "train":
            # Carve out val from train
            random.shuffle(images)
            n_val = int(len(images) * VAL_SPLIT)
            val_images   = images[:n_val]
            train_images = images[n_val:]

            for img in train_images:
                dest = os.path.join(DEST, "train", mapped_class)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(os.path.join(src_folder, img), dest)

            for img in val_images:
                dest = os.path.join(DEST, "val", mapped_class)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(os.path.join(src_folder, img), dest)

        elif split == "test":
            for img in images:
                dest = os.path.join(DEST, "test", mapped_class)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(os.path.join(src_folder, img), dest)

print("Copying training + val images...")
copy_images(SOURCE_TRAIN, "train")

print("Copying test images...")
copy_images(SOURCE_TEST, "test")

print("\nDone! Counting images per split...")
for split in ["train", "val", "test"]:
    for cls in ["unripe", "ripe", "rotten"]:
        folder = os.path.join(DEST, split, cls)
        if os.path.exists(folder):
            count = len(os.listdir(folder))
            print(f"  {split}/{cls}: {count} images")