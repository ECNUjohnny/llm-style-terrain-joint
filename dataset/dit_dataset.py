"""DiT 数据集模块 — 直接复用 UNetDataset"""

from dataset.unet_dataset import UNetDataset

# DiT 与 UNet 共享相同的数据格式 (RGB+DEM+Prompt 三元组)
DiTDataset = UNetDataset
