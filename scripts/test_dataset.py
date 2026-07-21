from backend.app.segmentation.dataset.dataset import DocumentDataset


dataset = DocumentDataset(
    images_dir="datasets/segmentation/images",
    masks_dir="datasets/segmentation/masks",
)

image, mask = dataset[0]

print("Dataset size:", len(dataset))
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Image range:", image.min().item(), image.max().item())
print("Mask values:", mask.unique())