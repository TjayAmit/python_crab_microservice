import json
import os

# === CONFIG ===
INPUT_JSON = "coco/result.json"       # Path to original Label Studio result.json
OUTPUT_JSON = "coco/result_fixed.json"  # Path to save updated JSON
IMAGE_FOLDER = "images"                # Folder for the images

# Ensure folder exists
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load JSON
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Update file_name in "images" array
for image in data.get("images", []):
    original_path = image.get("file_name", "")
    # Get the base filename
    filename = os.path.basename(original_path)
    # Remove hash: keep everything after the first '-'
    if "-" in filename:
        filename = filename.split("-", 1)[1]  # Split once, take second part
    # Update path
    image["file_name"] = f"{IMAGE_FOLDER}/{filename}"

# Save updated JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Updated file_name paths saved to {OUTPUT_JSON}")
