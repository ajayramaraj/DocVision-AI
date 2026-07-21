import torch

from backend.app.segmentation.models.unet import UNet

model = UNet()

x = torch.randn(4, 3, 256, 256)

y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
print("Output range:", y.min().item(), y.max().item())