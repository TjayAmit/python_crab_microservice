import json
from urllib.parse import unquote
import os

with open('coco/result.json', 'r') as f:
    coco_data = json.load(f)

print("First 5 image file_name patterns:")
print("="*60)
for i, img in enumerate(coco_data['images'][:5]):
    file_name = img['file_name']
    print(f"\n{i+1}. Original: {file_name}")
    
    if '/data/local-files/?d=' in file_name:
        encoded_path = file_name.split('?d=')[1]
        decoded_path = unquote(encoded_path)
        decoded_path = decoded_path.replace('/', '\\')
        print(f"   Decoded: {decoded_path}")
        print(f"   Exists: {os.path.exists(decoded_path)}")
