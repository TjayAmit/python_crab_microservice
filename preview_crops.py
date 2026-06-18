"""
preview_crops.py
----------------
Dumps a few CROPPED samples per class to model/crop_preview/<class>/ so you
can verify that the bbox crop captures the feature you labeled (e.g. the tail)
before kicking off a full training run.

Crop logic mirrors train_model_cropped.py exactly (same 10% margin).

Usage:
    python preview_crops.py            # 6 samples per class
    python preview_crops.py 10         # 10 samples per class
"""

import os
import sys
import json
from collections import defaultdict

from PIL import Image

COCO_DIR = "coco"
IMAGES_DIR = os.path.join(COCO_DIR, "images")
ANNOTATIONS_FILE = os.path.join(COCO_DIR, "result.json")
OUT_DIR = os.path.join("model", "crop_preview")
BOX_MARGIN = 0.10
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def resolve_path(file_name):
    for p in (os.path.join(COCO_DIR, file_name),
              os.path.join(IMAGES_DIR, file_name),
              file_name):
        if os.path.exists(p):
            return p
    return None


def safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def main():
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    cats = {c['id']: c['name'] for c in coco['categories']}
    img_path = {im['id']: resolve_path(im['file_name']) for im in coco['images']}

    by_class = defaultdict(list)
    for ann in coco['annotations']:
        by_class[ann['category_id']].append(ann)

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for cat_id, anns in by_class.items():
        cls = cats[cat_id]
        cdir = os.path.join(OUT_DIR, safe(cls))
        os.makedirs(cdir, exist_ok=True)
        saved = 0
        for ann in anns:
            if saved >= PER_CLASS:
                break
            path = img_path.get(ann['image_id'])
            if not path:
                continue
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            W, H = im.size
            x, y, bw, bh = ann['bbox']
            mx, my = bw * BOX_MARGIN, bh * BOX_MARGIN
            x1 = max(0, int(x - mx)); y1 = max(0, int(y - my))
            x2 = min(W, int(x + bw + mx)); y2 = min(H, int(y + bh + my))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = im.crop((x1, y1, x2, y2)).resize((224, 224))
            crop.save(os.path.join(cdir, f"{ann['id']}.jpg"), quality=90)
            saved += 1
            total += 1
        print(f"  {cls:<22} {saved} crops -> {cdir}")

    print(f"\nDone. {total} cropped samples written under: {OUT_DIR}")
    print("Open that folder and confirm each crop clearly shows the feature you labeled.")


if __name__ == "__main__":
    main()
