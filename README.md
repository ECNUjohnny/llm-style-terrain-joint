# LLM 风格地形生成联合模型

根据文本 Prompt 同时生成配对的 512×512 高程图和纹理图。采用 Latent Diffusion Model (LDM) 范式，在压缩隐空间中进行扩散。

## 项目结构

```
.
├── main.py                          # 主入口 (骨架)
├── models/
│   ├── clip/text_encoder.py         # 双分支 CLIP 文本编码器 (HuggingFace, 冻结)
│   ├── unet/unet_8ch.py             # 8 通道 U-Net 去噪模型
│   ├── dit/dit_8ch.py               # 8 通道 DiT 去噪模型 (PixArt-α XL, 934M)
│   └── vae/heightmap_vae.py         # 高程图专用 VAE (继承 AutoencoderKL)
├── train/train_pipeline.py          # 训练流水线 (支持 UNet/DiT 切换)
├── inference/inference_pipeline.py  # 推理流水线 (骨架)
├── utils/latent_utils.py            # 隐空间工具 + DDIM 调度器 (骨架)
├── dataset/
│   ├── height_map_dataset.py        # 高度图数据集 (.npy)
│   └── unet_dataset.py              # 联合数据集 (rgb+dem+txt)
├── scripts/
│   ├── height_vae/                  # 高度图 VAE 训练脚本
│   ├── unet/                        # U-Net 训练脚本
│   ├── dit/                         # DiT 训练脚本
│   └── data_process/                # 数据预处理 + 校验
└── docs/
    ├── goals.md                     # 项目目标
    ├── roadmap.md                   # 技术路线
    └── now.md                       # 当前状态
```

## 快速开始

```bash
# 安装依赖 (uv, Python 3.11)
uv sync

# 高度图 VAE 训练
uv run python scripts/height_vae/train_height_vae.py --epochs 100

# U-Net 训练
uv run python scripts/unet/train_unet_full.py --epochs 50 --data_root ./data/unet_training

# DiT 训练 (PixArt-α XL, ~20GB 显存, B=2)
uv run python scripts/dit/train_dit_full.py --epochs 50 --data_root ./data/unet_training
```

## 核心流程

### 8 通道联合隐空间

核心思路：将高度隐向量和纹理隐向量按通道拼接，让去噪模型在扩散过程中学习跨模态关联。

```
torch.cat([height_latent, texture_latent], dim=1) → [B, 8, 64, 64]
  channels 0-3: 高度隐向量 (HeightMapVAE, 4×64×64)
  channels 4-7: 纹理隐向量 (SD VAE, 4×64×64, ×0.18215)
```

### 训练过程

1. **文本编码**：Prompt → 双分支 CLIP → 全局特征 [B,768] + 局部特征 [B,77,768]
2. **图像压缩**：高度图 + 纹理图 → 各自 VAE 编码 → 8 通道联合隐向量
3. **前向加噪**：随机时间步 t，将高斯噪声混入联合隐向量
4. **噪声预测**：去噪模型 (UNet 或 DiT) 根据文本条件预测噪声
5. **损失计算**：预测噪声 vs 真实噪声的 MSE，DEM 通道权重 1.5×

### 推理过程

1. 文本编码 → 向量指令
2. 随机 8×64×64 高斯噪声
3. DDIM 50 步循环去噪
4. 拆分 8 通道 → 高度/纹理分别 VAE 解码

## 双分支 CLIP 文本编码

- **全局分支**：CLIP `pooler_output` [B,768] → 叠加到时间步嵌入，全局调制
- **局部分支**：CLIP `last_hidden_state` [B,77,768] → 注入交叉注意力层

### UNet 注入方式

全局特征 → timestep embedding 叠加；局部特征 → encoder_hidden_states → cross-attn

### DiT 注入方式

全局特征 → adaLN shift/scale/gate 调制；局部特征 → 投影到 1152 → cross-attn K/V

## 去噪模型对比

| 特性 | UNet8Channel | DiT8Channel |
|------|-------------|-------------|
| 架构 | U-Net 2D (diffusers) | PixArt-α XL (Transformer) |
| 参数量 | ~100-200M | ~934M |
| Block 数 | 4 down + 1 mid + 4 up | 28 transformer blocks |
| 空间处理 | 逐层下采样/上采样 + 跳跃连接 | Patchify (2×2) + self-attention |
| 文本注入 | Cross-attn + timestep add | adaLN + cross-attn |
| 预训练 | 无 | PixArt-α XL (自然图像) |
| 训练显存 (B=2) | ~8 GB | ~20 GB |
| 训练入口 | `scripts/unet/train_unet_full.py` | `scripts/dit/train_dit_full.py` |

## 数据管线

```
1081×1081 uint16 PNG → 中心裁剪 1080×1080 → Area 缩放 512×512
  → 百分位截断 → 对数变换 → 线性映射 [0,1] → .npy float32
```

- Raw: `data/origin/heightmaps_hf/` → Processed: `data/process/heightmaps_hf/`
- 训练数据需 `data_root/{rgb/, dem/, txt/}` 三个子目录，文件名一一对应

## 关键注意事项

- **uv** 管理依赖 (非 pip)，Python 3.11
- **AMP + grad_checkpointing 互斥**，同时启用会破坏梯度流
- **fp16 下坡度和 KL 散度计算**必须用 `torch.autocast(enabled=False)` 包裹
- `torch.autocast` 会拦截 `F.conv2d` 即使显式创建 fp32 张量
- 配置文件硬编码在脚本中，不存在 `configs/` 目录
- 文档注释为中文，修改现有文件时保持中文
