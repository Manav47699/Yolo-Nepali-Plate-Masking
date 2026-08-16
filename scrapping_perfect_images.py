import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
import easyocr
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# Target directories
RAW_DIR = "./temp_raw"
FINAL_DIR = "./perfect_nepali_plate_dataset"
TARGET_COUNT = 1200

# Strict search queries targeting top-down views of Nepali plates
QUERIES = [
    "Nepali Dal Bhat top view",
    "Thakali Khana set top view flat lay",
    "Nepali Thali plate directly above view",
    "Dal Bhat Tarkari top view high resolution",
    "Nepali food plate top view",
    "Authentic Nepali Khana top down photo",
    "Nepali Non Veg Khana Set top view"
]

print("[1/4] Initializing EasyOCR for watermark/text detection...")
# CPU bound text detector to reject blog images with text overlays
reader = easyocr.Reader(['en'], gpu=False)

def setup_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

def crawl_raw_data():
    print("[2/4] Scraper active: Collecting images from search engines...")
    for idx, q in enumerate(QUERIES):
        print(f" -> Mining query: '{q}'")
        
        # Bing (Higher retention of uncompressed photography)
        bing = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"q_{idx}_b")}
        )
        bing.crawl(keyword=q, max_num=350, min_size=(800, 800))
        
        # Google
        google = GoogleImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"q_{idx}_g")}
        )
        google.crawl(keyword=q, max_num=250, min_size=(800, 800))

def is_top_view(cv_img):
    """
    Checks if the main object (plate) is roughly circular (Top-View Perspective).
    Side angles distort circular plates into flat ellipses.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circular contours via Hough Transformation
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=int(min(cv_img.shape[:2]) * 0.25),
        maxRadius=int(min(cv_img.shape[:2]) * 0.48)
    )
    return circles is not None

def contains_text(img_path):
    """Detects text overlays/watermarks on the image."""
    results = reader.readtext(img_path)
    # Rejects image if OCR detects watermarks or recipes text
    return len(results) > 0

def process_and_filter():
    print("\n[3/4] AI Engine Running: Filtering Top-Views, Sharpness & Watermarks...")
    
    hashes = set()
    saved_count = 0

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if saved_count >= TARGET_COUNT:
                break

            file_path = os.path.join(root, file)

            try:
                # 1. Image Load & Resolution Check (> 800x800)
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    continue
                
                h, w = cv_img.shape[:2]
                if h < 800 or w < 800:
                    continue

                # 2. Sharpness / Blur Detection (Laplacian Variance)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < 120.0:  # Drop blurry or low-quality uploads
                    continue

                # 3. Top-View Geometry Verification
                if not is_top_view(cv_img):
                    continue

                # 4. Watermark and Text Removal
                if contains_text(file_path):
                    continue

                # 5. Deduplication using Perceptual Hash
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                p_hash = imagehash.phash(pil_img)
                if p_hash in hashes:
                    continue
                
                hashes.add(p_hash)

                # 6. Save cleanly formatted final image
                saved_count += 1
                out_name = f"nepali_plate_topview_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(FINAL_DIR, out_name), cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
                print(f"Accepted ({saved_count}/{TARGET_COUNT}): {out_name}")

            except Exception:
                continue

    print(f"\n[4/4] Complete! {saved_count} clean top-view images saved to '{FINAL_DIR}'.")
    shutil.rmtree(RAW_DIR, ignore_errors=True)

if __name__ == "__main__":
    setup_dirs()
    crawl_raw_data()
    process_and_filter()