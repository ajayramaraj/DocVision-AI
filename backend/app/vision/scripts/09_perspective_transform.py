import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

image_path = PROJECT_ROOT / "backend" / "app" / "vision" / "samples" / "invoices" / "invoice1.jpg"

output_dir = PROJECT_ROOT / "backend" / "app" / "vision" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Image not found!")

height, width = image.shape[:2]

# Example corner points (replace later with detected contour corners)
pts1 = np.float32([
    [0,0],
    [width-1,0],
    [0,height-1],
    [width-1,height-1]
])

pts2 = np.float32([
    [0,0],
    [800,0],
    [0,1000],
    [800,1000]
])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

scanned = cv2.warpPerspective(image, matrix, (800,1000))

output_path = output_dir / "invoice1_scanned.jpg"

cv2.imwrite(str(output_path), scanned)

print("="*50)
print("Perspective Transform")
print("="*50)

print(f"Output Shape : {scanned.shape}")

print(f"\nSaved To:\n{output_path}")