"""
DiT 训练数据集

数据格式: data_root/{rgb/, dem/, txt/}
  - rgb/: 512×512 PNG (RGB)
  - dem/: 512×512 NPY (float32, 已归一化到 [0,1])
  - txt/: 文本 prompt
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Normalize


class DiTDataset(Dataset):
    """8 通道联合生成数据集 —— 简洁、预处理好后直接使用"""

    def __init__(
        self,
        data_root: str,
        *,
        augment: bool = False,
        metadata: list | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.augment = augment
        self.metadata = metadata if metadata is not None else self._scan()
        self.to_tensor = ToTensor()
        self.normalize_rgb = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # 文件扫描
    def _scan(self) -> list[dict]:
        """按文件名匹配 rgb/dem/txt 三元组"""
        rgb_dir = os.path.join(self.data_root, "rgb")
        dem_dir = os.path.join(self.data_root, "dem")
        txt_dir = os.path.join(self.data_root, "txt")

        entries = []
        for f in sorted(os.listdir(rgb_dir)):
            if not f.endswith(".png"):
                continue
            name = os.path.splitext(f)[0]

            dem_path = os.path.join(dem_dir, f"{name}.npy")
            if not os.path.exists(dem_path):
                continue

            txt_path = os.path.join(txt_dir, f"{name}.txt")

            entries.append(
                {
                    "rgb": os.path.join(rgb_dir, f),
                    "dem": dem_path,
                    "txt": txt_path,
                    "name": name,
                }
            )

        print(f"DiTDataset: {len(entries)} 组数据已匹配 ({self.data_root})")
        return entries

    # 读取
    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)  # [3, 512, 512], [0, 1]

    def _load_dem(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)  # [512, 512]
        return torch.from_numpy(arr).unsqueeze(0)  # [1, 512, 512]

    def _load_txt(self, path: str) -> str:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        return ""

    # 增强
    def _augment(self, rgb: torch.Tensor, dem: torch.Tensor):
        """同步随机翻转和旋转"""
        if torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, [-1])
            dem = torch.flip(dem, [-1])
        if torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, [-2])
            dem = torch.flip(dem, [-2])
        if torch.rand(1).item() < 0.5:
            k = torch.randint(0, 4, (1,)).item()
            if k:
                rgb = torch.rot90(rgb, k, [-2, -1])
                dem = torch.rot90(dem, k, [-2, -1])
        return rgb, dem

    # 接口
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        entry = self.metadata[idx]

        rgb = self._load_rgb(entry["rgb"])  # [3, 512, 512], [0,1]
        dem = self._load_dem(entry["dem"])  # [1, 512, 512], [0,1]
        prompt = self._load_txt(entry["txt"])

        if self.augment:
            rgb, dem = self._augment(rgb, dem)

        rgb = self.normalize_rgb(rgb)  # [3, 512, 512], [-1,1]

        return {"rgb": rgb, "dem": dem, "prompt": prompt, "basename": entry["name"]}
