import json
from urllib.parse import unquote

# Read the COCO JSON file
with open('coco/result.json', 'r') as f:
    coco_data = json.load(f)

# Create category_id to category_name mapping
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Map category names to actual directory names
category_to_dir = {
    "Blue Crab Female": "500 Blue crab female",
    "Blue Crab Gay": "500 Blue crab gay",
    "Blue Crab Male": "500 Blue crab male",
    "Curacha Female": "500 Curacha f",
    "Curacha Male": "500 Curacha M",
    "Killer Crab": "killer crab",
    "Mud Crab Female": "500 mud crab female",
    "Mud Crab Gay": "500 Mud crab gay",
    "Mud Crab Male": "500 Mud crab male"
}

# Create image_id to category_id mapping from annotations
image_to_category = {}
for ann in coco_data['annotations']:
    image_to_category[ann['image_id']] = ann['category_id']

# Update file_name for each image
for img in coco_data['images']:
    img_id = img['id']
    file_name = img['file_name']
    
    # Handle URL-encoded Label Studio format
    if '/data/local-files/?d=' in file_name:
        # Extract and decode the path
        encoded_path = file_name.split('?d=')[1]
        decoded_path = unquote(encoded_path)
        # Get just the filename
        filename = decoded_path.split('\\')[-1]
        
        # Get the category for this image
        category_id = image_to_category.get(img_id)
        if category_id is not None:
            category_name = categories[category_id]
            dir_name = category_to_dir.get(category_name, category_name)
            # Update to new path format: datasets/[dir name]/filename
            new_filename = f"datasets/{dir_name}/{filename}"
            img['file_name'] = new_filename
            print(f"Updated: {file_name[:50]}... -> {new_filename}")

# Write the updated JSON back
with open('coco/result.json', 'w') as f:
    json.dump(coco_data, f, indent=2)

print(f"\n✅ Updated {len(coco_data['images'])} image paths in coco/result.json")
