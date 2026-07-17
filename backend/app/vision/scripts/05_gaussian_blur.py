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
    raise FileNotFoundError("Image not found!")

# ---------------------------------------------------
# Convert to Gray
# ---------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Gaussian Blur
# ---------------------------------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path = output_dir / "invoice1_blur.jpg"

cv2.imwrite(str(output_path), blur)

# ---------------------------------------------------
# Output
# ---------------------------------------------------

print("=" * 50)
print("Gaussian Blur")
print("=" * 50)

print(f"Gray Shape : {gray.shape}")
print(f"Blur Shape : {blur.shape}")

print("\nKernel Size : (5,5)")
print("Sigma       : 0")

print(f"\nSaved to:\n{output_path}")