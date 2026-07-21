from pathlib import Path

import cv2
import numpy as np
import torch

from backend.app.segmentation.models.unet import UNet
from backend.app.segmentation.postprocess import clean_mask
from backend.app.segmentation.contours import detect_document_contour
from backend.app.segmentation.corners import detect_document_corners
from backend.app.segmentation.perspective import warp_document


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "weights" / "best_unet.pth"
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_SIZE = 256
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    model = UNet(
        in_channels=3,
        out_channels=1,
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original = image.copy()

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)

    return original, image.to(DEVICE)


def predict_mask(
    model,
    image_tensor,
    original_shape,
):
    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits)
        mask = (probability > THRESHOLD).float()

    mask = mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    height, width = original_shape[:2]

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    image_files = []

    for extension in (
        "*.png",
        "*.jpg",
        "*.jpeg",
    ):
        image_files.extend(
            INPUT_DIR.glob(extension)
        )

    if not image_files:
        print(
            f"No test images found inside: {INPUT_DIR}"
        )
        return

    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_PATH}")

    model = load_model()

    for image_path in image_files:
        original, image_tensor = preprocess_image(
            image_path
        )

        raw_mask = predict_mask(
            model,
            image_tensor,
            original.shape,
        )

        cleaned_mask = clean_mask(raw_mask)

        contour_image, contour = detect_document_contour(
            original,
            cleaned_mask,
        )

        corner_image, corners = detect_document_corners(
            contour_image,
            contour,
        )

        scanned_document = warp_document(
            original,
            corners,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{image_path.stem}_raw_mask.png"
            ),
            raw_mask,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{image_path.stem}_clean_mask.png"
            ),
            cleaned_mask,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{image_path.stem}_contour.png"
            ),
            contour_image,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{image_path.stem}_corners.png"
            ),
            corner_image,
        )

        if scanned_document is not None:
            scanned_path = (
                OUTPUT_DIR
                / f"{image_path.stem}_scanned.png"
            )

            cv2.imwrite(
                str(scanned_path),
                scanned_document,
            )

            print(f"Saved scanned document: {scanned_path}")
        else:
            print(
                f"Could not perform perspective transformation: "
                f"{image_path.name}"
            )

        print(f"Processed: {image_path.name}")

    print(
        "Perspective transformation completed successfully."
    )


if __name__ == "__main__":
    main()