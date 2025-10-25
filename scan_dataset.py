import os
from PIL import Image
import logging

# === CONFIG ===
DATASET_DIR = "curacha_dataset"
LOG_FILE = "scan_results.log"

# === LOGGING SETUP ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("🔍 Scanning dataset for corrupted or unreadable images...")
logging.info("=== DATASET SCAN START ===")

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

total_files = 0
valid_count = 0
corrupted_count = 0
non_image_count = 0

for root, _, files in os.walk(DATASET_DIR):
    for filename in files:
        total_files += 1
        path = os.path.join(root, filename)

        if not filename.lower().endswith(valid_exts):
            non_image_count += 1
            logging.warning(f"Skipping non-image file: {path}")
            continue

        try:
            # Attempt to open and verify image integrity
            with Image.open(path) as img:
                img.verify()
            valid_count += 1
        except Exception as e:
            corrupted_count += 1
            logging.error(f"❌ Corrupted or unreadable: {path} | Error: {e}")

# === SUMMARY ===
summary = (
    f"\n=== SCAN COMPLETE ===\n"
    f"📁 Dataset directory: {DATASET_DIR}\n"
    f"🖼️ Total files scanned: {total_files}\n"
    f"✅ Valid images: {valid_count}\n"
    f"❌ Corrupted images: {corrupted_count}\n"
    f"⚠️ Non-image files skipped: {non_image_count}\n"
    f"🧾 Log file saved to: {LOG_FILE}\n"
)
print(summary)
logging.info(summary)
