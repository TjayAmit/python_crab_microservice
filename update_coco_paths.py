import json

# Read the COCO JSON file
with open('cocos/result.json', 'r') as f:
    coco_data = json.load(f)

# Create category_id to category_name mapping
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Create image_id to category_id mapping from annotations
image_to_category = {}
for ann in coco_data['annotations']:
    image_to_category[ann['image_id']] = ann['category_id']

# Update file_name for each image
for img in coco_data['images']:
    img_id = img['id']
    old_filename = img['file_name']
    
    # Get the category for this image
    category_id = image_to_category.get(img_id)
    if category_id is not None:
        category_name = categories[category_id]
        # Update to new path format: datasets/[category name]/filename
        new_filename = f"datasets/{category_name}/{old_filename}"
        img['file_name'] = new_filename
        print(f"Updated: {old_filename} -> {new_filename}")

# Write the updated JSON back
with open('cocos/result.json', 'w') as f:
    json.dump(coco_data, f, indent=2)

print(f"\n✅ Updated {len(coco_data['images'])} image paths in cocos/result.json")
