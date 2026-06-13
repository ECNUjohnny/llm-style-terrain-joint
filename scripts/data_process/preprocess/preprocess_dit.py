"""
DiT 训练数据预处理脚本

将 origin 数据标准化为 DiT 训练格式：
  - RGB:  .tif → .png，统一缩放到 512×512
  - DEM:  .npy → .npy，统一缩放到 512×512（保持 float32）
  - TXT:  .txt → .txt，直接复制

输入: data/origin/dit/{rgb/, dem/, txt/}
输出: data/dit_training/{rgb/, dem/, txt/}

用法:
    uv run scripts/data_process/preprocess/preprocess_dit.py
"""

import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# 配置
ORIGIN_DIR = "./data/origin/dit"
OUTPUT_DIR = "./data/dit_training"
TARGET_SIZE = 512


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def process_rgb(origin_dir: str, output_dir: str):
    """RGB: .tif → .png, 确保 512×512"""
    subdirs = {"rgb": ".tif"}
    for sub, ext in subdirs.items():
        src_dir = os.path.join(origin_dir, sub)
        dst_dir = ensure_dir(os.path.join(output_dir, sub))

        files = sorted(f for f in os.listdir(src_dir) if f.endswith(ext))
        if not files:
            print(f"[RGB] 跳过: {src_dir} 中无 {ext} 文件")
            continue

        print(f"[RGB] {len(files)} 个 {ext} → png ({src_dir} → {dst_dir})")
        for fname in tqdm(files, desc="RGB"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png")

            img = Image.open(src).convert("RGB")
            if img.size != (TARGET_SIZE, TARGET_SIZE):
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            img.save(dst, "PNG")


def process_dem(origin_dir: str, output_dir: str):
    """DEM: .npy → .npy, 确保 512×512"""
    subdirs = {"dem": ".npy"}
    for sub, ext in subdirs.items():
        src_dir = os.path.join(origin_dir, sub)
        dst_dir = ensure_dir(os.path.join(output_dir, sub))

        files = sorted(f for f in os.listdir(src_dir) if f.endswith(ext))
        if not files:
            print(f"[DEM] 跳过: {src_dir} 中无 {ext} 文件")
            continue

        print(f"[DEM] {len(files)} 个 {ext} → npy ({src_dir} → {dst_dir})")
        for fname in tqdm(files, desc="DEM"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)

            arr = np.load(src).astype(np.float32)
            if arr.shape != (TARGET_SIZE, TARGET_SIZE):
                img = Image.fromarray(arr)
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32)
            np.save(dst, arr)


def process_txt(origin_dir: str, output_dir: str):
    """TXT: 直接复制"""
    subdirs = {"txt": ".txt"}
    for sub, ext in subdirs.items():
        src_dir = os.path.join(origin_dir, sub)
        dst_dir = ensure_dir(os.path.join(output_dir, sub))

        files = sorted(f for f in os.listdir(src_dir) if f.endswith(ext))
        if not files:
            print(f"[TXT] 跳过: {src_dir} 中无 {ext} 文件")
            continue

        print(f"[TXT] {len(files)} 个 {ext} → txt ({src_dir} → {dst_dir})")
        for fname in tqdm(files, desc="TXT"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    if not os.path.exists(ORIGIN_DIR):
        raise FileNotFoundError(f"Origin 目录不存在: {ORIGIN_DIR}")

    process_rgb(ORIGIN_DIR, OUTPUT_DIR)
    process_dem(ORIGIN_DIR, OUTPUT_DIR)
    process_txt(ORIGIN_DIR, OUTPUT_DIR)

    print(f"\n预处理完成 → {OUTPUT_DIR}/")
