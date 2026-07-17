from pathlib import Path
import cv2

current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[4]

image_path = PROJECT_ROOT / "backend" / "app" / "vision" / "samples" / "invoices" / "invoice1.jpg"

output_dir = PROJECT_ROOT / "backend" / "app" / "vision" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Read image
image = cv2.imread(str(image_path))

# Check if image loaded
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Print image information
print("=" * 50)
print("Image Loaded Successfully")
print("=" * 50)

print(f"Image Path : {image_path.name}")
print(f"Shape      : {image.shape}")
print(f"Height     : {image.shape[0]} pixels")
print(f"Width      : {image.shape[1]} pixels")
print(f"Channels   : {image.shape[2]}")
print(f"Data Type  : {image.dtype}")
print(f"Total Size : {image.size} values")

# Save a copy
output_path = output_dir / "invoice1_copy.jpg"
cv2.imwrite(str(output_path), image)

print(f"\nImage copy saved to:\n{output_path}")