from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from backend.app.segmentation.dataset.dataset import DocumentDataset
from backend.app.segmentation.models.unet import UNet
from backend.app.segmentation.training.losses import BCEDiceLoss
from backend.app.segmentation.transforms import (
    get_train_transform,
    get_val_transform,
)


# ---------------- Configuration ----------------

IMAGES_DIR = "datasets/segmentation/images"
MASKS_DIR = "datasets/segmentation/masks"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42

MODEL_DIR = Path("backend/app/segmentation/outputs/models")
BEST_MODEL_PATH = MODEL_DIR / "best_unet.pth"


# ---------------- Metrics ----------------

def calculate_metrics(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        predictions.sum() + targets.sum() + 1e-6
    )

    union = predictions.sum() + targets.sum() - intersection

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return dice.item(), iou.item()


# ---------------- One Training Epoch ----------------

def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, masks)

        loss.backward()
        optimizer.step()

        dice, iou = calculate_metrics(logits.detach(), masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    number_of_batches = len(loader)

    return (
        total_loss / number_of_batches,
        total_dice / number_of_batches,
        total_iou / number_of_batches,
    )


# ---------------- Validation ----------------

def validate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_function(logits, masks)

            dice, iou = calculate_metrics(logits, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

    number_of_batches = len(loader)

    return (
        total_loss / number_of_batches,
        total_dice / number_of_batches,
        total_iou / number_of_batches,
    )


# ---------------- Main Training ----------------

def main():
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    base_dataset = DocumentDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    )

    dataset_size = len(base_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_full_dataset = DocumentDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        transform=get_train_transform(),
    )

    val_full_dataset = DocumentDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        transform=get_val_transform(),
    )

    train_dataset = Subset(
        train_full_dataset,
        train_indices,
    )

    val_dataset = Subset(
        val_full_dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("Training samples:", train_size)
    print("Validation samples:", val_size)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    model = UNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    loss_function = BCEDiceLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        val_loss, val_dice, val_iou = validate(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Dice: {train_dice:.4f} | "
            f"IoU: {train_iou:.4f}"
        )
        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"IoU: {val_iou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                BEST_MODEL_PATH,
            )

            print("Best model saved:", BEST_MODEL_PATH)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()