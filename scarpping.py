import os
import shutil
from PIL import Image
import imagehash
import cv2
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, FlickrImageCrawler

OUTPUT_DIR = "./nepali_dal_bhat_dataset"
RAW_DIR = "./temp_raw_downloads"
MIN_RES = (600, 600)  # Rejects anything under 600x600 px for YOLO quality

# Targeted search queries for maximum high-quality yield
QUERIES = [
    "Nepali Dal Bhat plate",
    "Thakali Khana set high resolution",
    "Nepali Thali plate close up",
    "Dal Bhat Tarkari Nepali food",
    "Nepali Non Veg Khana Set",
    "Nepali authentic food plate"
]

def create_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def crawl_images():
    print("[1/3] Crawling Bing & Google for high-res images...")
    
    for idx, query in enumerate(QUERIES):
        print(f" -> Searching query: '{query}'")
        
        # Crawl Bing (High yield for uncompressed photos)
        bing_crawler = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"query_{idx}_bing")}
        )
        bing_crawler.crawl(
            keyword=query,
            max_num=250,
            min_size=MIN_RES
        )

        # Crawl Google
        google_crawler = GoogleImageCrawler(
            downloader_threads=4,
            storage={'root_dir': os.path.join(RAW_DIR, f"query_{idx}_google")}
        )
        google_crawler.crawl(
            keyword=query,
            max_num=150,
            min_size=MIN_RES
        )

def process_and_deduplicate():
    print("\n[2/3] Filtering resolution and deleting duplicates...")
    
    seen_hashes = set()
    valid_count = 0
    duplicate_count = 0
    corrupt_count = 0

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 1. Check if valid image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Ensure file is not corrupt
                
                # Re-open for hash/size checks (verify closes file context)
                with Image.open(file_path) as img:
                    w, h = img.size
                    
                    # Double check minimum bounds
                    if w < MIN_RES[0] or h < MIN_RES[1]:
                        corrupt_count += 1
                        continue

                    # 2. Perceptual Hashing for deduplication
                    p_hash = imagehash.phash(img)
                    if p_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                    
                    seen_hashes.add(p_hash)

                # 3. Save clean image with standardized naming
                dest_filename = f"dal_bhat_{valid_count + 1:04d}.jpg"
                dest_path = os.path.join(OUTPUT_DIR, dest_filename)
                
                # Convert to standard RGB JPEG via OpenCV
                cv_img = cv2.imread(file_path)
                if cv_img is not None:
                    cv2.imwrite(dest_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    valid_count += 1

            except Exception:
                corrupt_count += 1
                continue

    print(f"\n[3/3] Done!")
    print(f" -> Clean High-Res Images Saved: {valid_count}")
    print(f" -> Duplicates Dropped: {duplicate_count}")
    print(f" -> Low-Res / Corrupt Dropped: {corrupt_count}")
    print(f" -> Final Dataset Directory: {os.path.abspath(OUTPUT_DIR)}")

    # Clean up temporary raw download directory
    shutil.rmtree(RAW_DIR, ignore_errors=True)

if __name__ == "__main__":
    create_dirs()
    crawl_images()
    process_and_deduplicate()