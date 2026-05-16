"""
VAE 魔法数字 (Scale Factor) 计算脚本

用法:
    python ./scripts/unet/get_sigma.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 退三层，确保能找到项目根目录下的自定义模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dataset.height_map_dataset import HeightMapDataset
from models.vae.heightmap_vae import HeightMapVAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # 1. 配置路径 (与你之前的训练脚本保持一致)
    # ==========================================
    DATA_ROOT = "./data/process/heightmaps_hf"
    VAE_CKPT = "./data/vae_model_data/best_checkpoint.pt" # 你的 VAE 权重路径
    
    MAX_SAMPLES = 500  
    BATCH_SIZE = 16     # 推理不占显存，可以稍微开大点

    # ==========================================
    # 2. 加载你自己的 HeightMap VAE
    # ==========================================
    print("正在加载 HeightMap VAE...")
    vae = HeightMapVAE(block_out_channels=(128, 256, 512, 512)).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)["model_state_dict"])
    vae.eval()
    vae.requires_grad_(False)

    # ==========================================
    # 3. 加载数据集
    # ==========================================
    print(f"正在加载数据集: {DATA_ROOT}...")
    # 注意：这里不需要数据增强(augment=False)，我们需要最真实的数据分布
    dataset = HeightMapDataset(data_root=DATA_ROOT, augment=False) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 4. 收集隐向量 (Latents)
    # ==========================================
    all_latents = []
    num_collected = 0

    print(f"开始提取隐向量 (目标: {MAX_SAMPLES} 张)...")
    pbar = tqdm(dataloader)
    
    for batch in pbar:
        dem_pixels = batch[0].to(device)
        
        with torch.no_grad():
            # 极其关键：这里绝对不要乘任何缩放常数！我们要的是最原汁原味的输出！
            latents = vae.encode(dem_pixels).latent_dist.sample()
            
        # 移回 CPU 并保存，防止显存爆炸
        all_latents.append(latents.cpu())
        
        num_collected += dem_pixels.shape[0]
        if num_collected >= MAX_SAMPLES:
            break

    # ==========================================
    # 5. 见证奇迹：计算统计量
    # ==========================================
    # 把所有 batch 的隐向量拼成一个巨大的张量
    all_latents_tensor = torch.cat(all_latents, dim=0)
    
    # 算均值和标准差
    std = torch.std(all_latents_tensor).item()
    mean = torch.mean(all_latents_tensor).item()
    
    # 你的专属魔法数字就是标准差的倒数！
    scale_factor = 1.0 / std

    print("\n" + "="*40)
    print("VAE 隐空间统计结果分析")
    print("="*40)
    print(f"分析样本数  : {all_latents_tensor.shape[0]} 张高度图")
    print(f"隐空间均值  : {mean:.6f} (越接近 0 越好)")
    print(f"隐空间标准差: {std:.6f} (这就是 VAE 原本的分布宽度)")
    print("-" * 40)
    print(f"你的专属魔法数字 (1/Std) 为: [{scale_factor:.6f}]")
    print("="*40)
    
    print(f"\n下一步：请去 train_unet_full.py 中，把 dem_vae 处理高度图时的 0.18215 替换为 {scale_factor:.6f}")

if __name__ == "__main__":
    main()