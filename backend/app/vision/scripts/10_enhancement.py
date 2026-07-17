import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

image_path = PROJECT_ROOT / "backend" / "app" / "vision" / "samples" / "invoices" / "invoice1.jpg"

output_dir = PROJECT_ROOT / "backend" / "app" / "vision" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Image not found!")

# Brightness / Contrast
enhanced = cv2.convertScaleAbs(
    image,
    alpha=1.3,
    beta=20
)

output_path = output_dir / "invoice1_enhanced.jpg"

cv2.imwrite(str(output_path), enhanced)

print("="*50)
print("Image Enhancement")
print("="*50)

print("Alpha (Contrast)  : 1.3")
print("Beta (Brightness) : 20")

print(f"\nSaved To:\n{output_path}")