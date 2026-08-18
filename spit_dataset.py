import os
import random
import shutil

# Configurations matching your exact folder names
SOURCE_DIR = "nepali_dal_bhat_dataset"  # Folder containing merged images and labels
OUTPUT_DIR = "final_yolo_model_dataset"  # Target directory for YOLO train/val structure
VAL_RATIO = 0.2                                         # 20% validation split

images_dir = os.path.join(SOURCE_DIR, "images")
labels_dir = os.path.join(SOURCE_DIR, "labels")

# Get list of image filenames
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42)
random.shuffle(image_files)

# Calculate train/val split
val_count = int(len(image_files) * VAL_RATIO)
val_images = set(image_files[:val_count])

# Create destination directories inside nepali_dal_bhat_dataset
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# Copy images and matching label files
for img_name in image_files:
    split = 'val' if img_name in val_images else 'train'
    
    base_name = os.path.splitext(img_name)[0]
    label_name = f"{base_name}.txt"
    
    src_img = os.path.join(images_dir, img_name)
    src_lbl = os.path.join(labels_dir, label_name)
    
    dst_img = os.path.join(OUTPUT_DIR, 'images', split, img_name)
    dst_lbl = os.path.join(OUTPUT_DIR, 'labels', split, label_name)
    
    shutil.copy(src_img, dst_img)
    
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, dst_lbl)

print(f"Dataset successfully organized into '{OUTPUT_DIR}'!")