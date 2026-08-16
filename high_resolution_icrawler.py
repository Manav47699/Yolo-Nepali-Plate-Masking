import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler

OUTPUT_DIR = "./nepali_dal_bhat_dataset"
RAW_DIR = "./temp_raw_downloads"

# Filter 1: High Resolution Enforcer (Rejects small icons/thumbnails)
MIN_RES = (800, 800)

# Targeted search queries focusing on top-down views and clean plates
QUERIES = [
    "Nepali Dal Bhat top view",
    "Thakali Khana set top view flat lay",
    "Nepali Thali plate directly above view",
    "Dal Bhat Tarkari top view high resolution",
    "Nepali food plate top view",
    "Authentic Nepali Khana top down photo",
    "Nepali Non Veg Khana Set top view",
    "Thakali Thali set photo top view",
    "Nepali veg khana set plate top view",
    "Nepali chicken dal bhat plate top view",
    "Nepali village dal bhat top view",
    "Authentic Nepali lunch plate top view",
    "Nepali restaurant dal bhat set top view"
]

def setup_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def crawl_images():
    print("[1/3] Mining Google, Bing, and Baidu engines...")
    
    for idx, q in enumerate(QUERIES):
        print(f" -> Mining query: '{q}'")
        
        # 1. Bing Crawl
        bing = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"q_{idx}_b")}
        )
        bing.crawl(keyword=q, max_num=1000, min_size=MIN_RES)
        
        # 2. Google Crawl
        google = GoogleImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"q_{idx}_g")}
        )
        google.crawl(keyword=q, max_num=1000, min_size=MIN_RES)

        # 3. Baidu Crawl (Additional pool for uncompressed food images)
        baidu = BaiduImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"q_{idx}_ba")}
        )
        baidu.crawl(keyword=q, max_num=1000, min_size=MIN_RES)

def Fast_Filter_and_Save():
    print("\n[2/3] Processing images (Sharpness, Resolution, Deduplication)...")
    
    hashes = set()
    saved_count = 0
    duplicate_count = 0
    low_quality_count = 0

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                # 1. Verify Image Integrity & Resolution
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    continue
                
                h, w = cv_img.shape[:2]
                if h < MIN_RES[1] or w < MIN_RES[0]:
                    low_quality_count += 1
                    continue

                # 2. Fast Blur Detection (Laplacian Variance check)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < 100.0:  # Drop blurry photos
                    low_quality_count += 1
                    continue

                # 3. Fast Deduplication (Perceptual Hash)
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                p_hash = imagehash.phash(pil_img)
                if p_hash in hashes:
                    duplicate_count += 1
                    continue
                
                hashes.add(p_hash)

                # 4. Save clean image with standardized naming
                saved_count += 1
                out_name = f"dal_bhat_topview_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            except Exception:
                continue

    print(f"\n[3/3] Done!")
    print(f" -> High-Resolution Clean Images Saved: {saved_count}")
    print(f" -> Duplicates Dropped: {duplicate_count}")
    print(f" -> Blurry / Low-Res Dropped: {low_quality_count}")
    print(f" -> Output Directory: {os.path.abspath(OUTPUT_DIR)}")

    # Cleanup temporary downloads
    shutil.rmtree(RAW_DIR, ignore_errors=True)

if __name__ == "__main__":
    setup_dirs()
    crawl_images()
    Fast_Filter_and_Save()