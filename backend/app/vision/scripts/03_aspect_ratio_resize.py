import cv2
from pathlib import Path

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

image_path = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "vision"
    / "samples"
    / "invoices"
    / "invoice1.jpg"
)

output_dir = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "vision"
    / "outputs"
)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Read Image
# ---------------------------------------------------

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# ---------------------------------------------------
# Original Size
# ---------------------------------------------------

height, width = image.shape[:2]

print("=" * 50)
print("Original Image")
print("=" * 50)
print(f"Width  : {width}")
print(f"Height : {height}")

# ---------------------------------------------------
# Resize while maintaining aspect ratio
# ---------------------------------------------------

target_width = 800

scale = target_width / width

target_height = int(height * scale)

resized = cv2.resize(image, (target_width, target_height))

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path = output_dir / "invoice1_aspect_resize.jpg"

cv2.imwrite(str(output_path), resized)

# ---------------------------------------------------
# Output Information
# ---------------------------------------------------

print("\nAspect Ratio Resize")
print("=" * 50)

print(f"Scale Factor : {scale:.2f}")

print(f"New Width    : {resized.shape[1]}")
print(f"New Height   : {resized.shape[0]}")

print(f"\nSaved To:\n{output_path}")