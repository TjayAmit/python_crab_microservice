import os
import logging
import re

# === CONFIG ===
LOG_FILE = "scan_results.log"   # must exist from scan_dataset.py
DELETE_LOG = "delete_results.log"

# === LOGGING SETUP ===
logging.basicConfig(
    filename=DELETE_LOG,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_corrupted_paths(log_file):
    """
    Parse scan_results.log and extract corrupted file paths.
    Handles Unicode escape sequences and Windows-style paths.
    """
    corrupted_files = []

    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return corrupted_files

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            # Match lines with "Corrupted or unreadable:"
            if "Corrupted or unreadable:" in line:
                # Extract path between markers
                match = re.search(r"Corrupted or unreadable:\s*(.*?)\s*\|", line)
                if match:
                    raw_path = match.group(1)
                    # Decode escaped Unicode like \u274c
                    decoded_line = raw_path.encode().decode('unicode_escape')
                    # Normalize slashes for current OS
                    norm_path = os.path.normpath(decoded_line.strip())
                    corrupted_files.append(norm_path)
    return corrupted_files

def delete_files(file_list):
    """Delete all corrupted files safely"""
    deleted, failed = 0, 0

    for path in file_list:
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted += 1
                print(f"🗑️ Deleted: {path}")
                logging.info(f"🗑️ Deleted: {path}")
            except Exception as e:
                failed += 1
                print(f"⚠️ Failed to delete {path}: {e}")
                logging.error(f"⚠️ Failed to delete {path}: {e}")
        else:
            print(f"⚠️ File not found: {path}")
            logging.warning(f"⚠️ File not found: {path}")

    return deleted, failed


# === MAIN ===
if __name__ == "__main__":
    print("🧹 Reading scan_results.log for corrupted images...")
    corrupted_files = extract_corrupted_paths(LOG_FILE)

    if not corrupted_files:
        print("✅ No corrupted files found or log is empty.")
    else:
        print(f"Found {len(corrupted_files)} corrupted files. Proceeding with deletion...\n")
        deleted, failed = delete_files(corrupted_files)

        print("\n=== DELETE SUMMARY ===")
        print(f"🗑️ Deleted files: {deleted}")
        print(f"⚠️ Errors: {failed}")
        print(f"🧾 Log saved to: {DELETE_LOG}")

        logging.info(f"=== DELETE SUMMARY === Deleted: {deleted}, Errors: {failed}")
