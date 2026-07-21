import json
from pathlib import Path

import cv2
import numpy as np


ANNOTATIONS_DIR = Path("datasets/segmentation/annotations")
MASKS_DIR = Path("datasets/segmentation/masks")

MASKS_DIR.mkdir(parents=True, exist_ok=True)

json_files = list(ANNOTATIONS_DIR.glob("*.json"))

created = 0
skipped = 0

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data.get("imageHeight")
    width = data.get("imageWidth")

    if not height or not width:
        print(f"Skipped: {json_file.name} — missing image size")
        skipped += 1
        continue

    mask = np.zeros((height, width), dtype=np.uint8)

    found_document = False

    for shape in data.get("shapes", []):
        if shape.get("label") != "document":
            continue

        points = np.array(shape["points"], dtype=np.int32)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)
            found_document = True

    if not found_document:
        print(f"Skipped: {json_file.name} — no 'document' polygon")
        skipped += 1
        continue

    output_path = MASKS_DIR / f"{json_file.stem}.png"
    cv2.imwrite(str(output_path), mask)

    created += 1
    print(f"Created: {output_path.name}")

print("\nFinished")
print(f"Masks created: {created}")
print(f"Skipped: {skipped}")