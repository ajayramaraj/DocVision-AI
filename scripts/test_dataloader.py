import torch
from torch.utils.data import DataLoader, random_split

from backend.app.segmentation.dataset.dataset import DocumentDataset
from backend.app.segmentation.transforms import (
    get_train_transform,
    get_val_transform,
)


full_dataset = DocumentDataset(
    images_dir="datasets/segmentation/images",
    masks_dir="datasets/segmentation/masks",
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_indices, val_indices = random_split(
    range(len(full_dataset)),
    [train_size, val_size],
    generator=generator,
)

train_dataset = DocumentDataset(
    images_dir="datasets/segmentation/images",
    masks_dir="datasets/segmentation/masks",
    transform=get_train_transform(),
)

val_dataset = DocumentDataset(
    images_dir="datasets/segmentation/images",
    masks_dir="datasets/segmentation/masks",
    transform=get_val_transform(),
)

train_dataset.samples = [
    train_dataset.samples[i] for i in train_indices.indices
]

val_dataset.samples = [
    val_dataset.samples[i] for i in val_indices.indices
]

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

images, masks = next(iter(train_loader))

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Image batch shape:", images.shape)
print("Mask batch shape:", masks.shape)
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)
print("Mask values:", torch.unique(masks))