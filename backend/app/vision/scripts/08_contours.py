import cv2
from pathlib import Path

# ---------------- Project Paths ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

image_path = PROJECT_ROOT / "backend" / "app" / "vision" / "samples" / "invoices" / "invoice1.jpg"

output_dir = PROJECT_ROOT / "backend" / "app" / "vision" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------- Read ----------------

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Image not found!")

original = image.copy()

# ---------------- Gray ----------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------- Blur ----------------

blur = cv2.GaussianBlur(gray, (5,5), 0)

# ---------------- Edges ----------------

edges = cv2.Canny(blur, 75, 200)

# ---------------- Contours ----------------

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

cv2.drawContours(original, [largest], -1, (0,255,0), 3)

output_path = output_dir / "invoice1_contour.jpg"

cv2.imwrite(str(output_path), original)

print("="*50)
print("Contour Detection")
print("="*50)

print(f"Contours Found : {len(contours)}")
print(f"Largest Area   : {cv2.contourArea(largest):.2f}")

print(f"\nSaved To:\n{output_path}")