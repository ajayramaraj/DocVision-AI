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
# Grayscale
# ---------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Gaussian Blur
# ---------------------------------------------------

blur = cv2.GaussianBlur(gray, (5,5), 0)

# ---------------------------------------------------
# Adaptive Threshold
# ---------------------------------------------------

threshold = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path = output_dir / "invoice1_threshold.jpg"

cv2.imwrite(str(output_path), threshold)

# ---------------------------------------------------
# Output
# ---------------------------------------------------

print("="*50)
print("Adaptive Threshold")
print("="*50)

print(f"Shape : {threshold.shape}")

print(f"\nSaved To:\n{output_path}")