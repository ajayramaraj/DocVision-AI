import cv2
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Input image
image_path = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "vision"
    / "samples"
    / "invoices"
    / "invoice1.jpg"
)

# Output directory
output_dir = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "vision"
    / "outputs"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Read image
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Original size
height, width = image.shape[:2]

print("=" * 50)
print("Original Image")
print("=" * 50)
print(f"Width  : {width}")
print(f"Height : {height}")

# Resize
new_width = 800
new_height = 600

resized = cv2.resize(image, (new_width, new_height))

# Save image
output_path = output_dir / "invoice1_resized.jpg"
cv2.imwrite(str(output_path), resized)

print("\nAfter Resize")
print("=" * 50)
print(f"Width  : {resized.shape[1]}")
print(f"Height : {resized.shape[0]}")

print(f"\nSaved to:\n{output_path}")