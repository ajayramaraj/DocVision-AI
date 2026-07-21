from pathlib import Path

# Folder containing your images
IMAGE_DIR = Path("datasets/segmentation/images")

# Supported image extensions
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Get all image files
images = sorted(
    [f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in EXTENSIONS]
)

print(f"Found {len(images)} images.")

# Rename sequentially
for i, image in enumerate(images, start=1):
    new_name = f"doc_{i:06d}{image.suffix.lower()}"
    new_path = IMAGE_DIR / new_name

    image.rename(new_path)
    print(f"{image.name}  -->  {new_name}")

print("\n✅ Renaming completed successfully!")