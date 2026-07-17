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
# Gray
# ---------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Blur
# ---------------------------------------------------

blur = cv2.GaussianBlur(gray, (5,5), 0)

# ---------------------------------------------------
# Edge Detection
# ---------------------------------------------------

edges = cv2.Canny(
    blur,
    threshold1=75,
    threshold2=200
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path = output_dir / "invoice1_edges.jpg"

cv2.imwrite(str(output_path), edges)

# ---------------------------------------------------
# Output
# ---------------------------------------------------

print("="*50)
print("Canny Edge Detection")
print("="*50)

print(f"Shape : {edges.shape}")

print(f"\nSaved To:\n{output_path}")