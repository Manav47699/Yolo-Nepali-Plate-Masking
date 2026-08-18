import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash

INPUT_DIR = "./Dalbhat"                      # Your current low-res folder
OUTPUT_DIR = "./parikar_cleaned"    # Target clean high-res directory
MIN_RES = (600, 600)                        # Required YOLO threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_enhance():
    print(f"[1/2] Processing images from '{INPUT_DIR}'...")
    
    seen_hashes = set()
    valid_count = 0
    upscaled_count = 0
    rejected_count = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                continue

            file_path = os.path.join(root, file)

            try:
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    continue

                h, w = cv_img.shape[:2]

                # 1. Deduplication Check
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                p_hash = imagehash.phash(pil_img)
                if p_hash in seen_hashes:
                    continue
                seen_hashes.add(p_hash)

                # 2. Check Resolution & Upscale if below threshold
                if w < MIN_RES[0] or h < MIN_RES[1]:
                    # Calculate scale factor to hit target size
                    scale_w = MIN_RES[0] / w
                    scale_h = MIN_RES[1] / h
                    scale = max(scale_w, scale_h)

                    # High-quality Lanczos bicubic interpolation for clean edges
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    upscaled_count += 1
                
                # 3. Fast Sharpness Filter (Reject excessively blurry outputs)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < 80.0:
                    rejected_count += 1
                    continue

                # 4. Save to Output Directory
                valid_count += 1
                out_name = f"dal_bhat_hd_{valid_count:04d}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            except Exception as e:
                continue

    print("\n[2/2] Processing Complete!")
    print(f" -> High-Res Images Ready for YOLO: {valid_count}")
    print(f" -> Images Upscaled to 600x600+: {upscaled_count}")
    print(f" -> Blurry Images Dropped: {rejected_count}")
    print(f" -> Output Directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    process_and_enhance()