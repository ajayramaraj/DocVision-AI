from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        transform=None,
        image_size=(256, 256),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size

        image_extensions = [".png", ".jpg", ".jpeg"]

        self.samples = []

        for mask_path in sorted(self.masks_dir.glob("*.png")):
            image_path = None

            for extension in image_extensions:
                possible_path = self.images_dir / f"{mask_path.stem}{extension}"

                if possible_path.exists():
                    image_path = possible_path
                    break

            if image_path is not None:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError("No matching image-mask pairs found.")

        print(f"Matching image-mask pairs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            width, height = self.image_size

            image = cv2.resize(image, (width, height))
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1) / 255.0

            mask = torch.tensor(mask, dtype=torch.float32)

        # Make sure mask is a tensor
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32)

        mask = mask.float()

        # Add channel dimension if missing
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Normalize mask to 0–1
        if mask.max() > 1:
            mask = mask / 255.0

        # Convert to binary mask
        mask = (mask > 0.5).float()

        return image, mask