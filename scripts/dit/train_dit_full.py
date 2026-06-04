"""
DiT 8 通道训练脚本

基于 PixArt-alpha XL 架构的 DiT，替代 UNet 进行纹理 + 高程联合去噪训练。

用法：
    # 全新训练
    python scripts/dit/train_dit_full.py --epochs 50

    # 断点续训
    python scripts/dit/train_dit_full.py --epochs 100 --checkpoint ./outputs/dit_8ch/checkpoint.pt

    # 测试噪声预测精度
    python scripts/dit/train_dit_full.py --mode test --checkpoint ./outputs/dit_8ch/checkpoint.pt

    # 指定 PixArt-alpha 预训练权重路径
    python scripts/dit/train_dit_full.py --epochs 50 --pretrained_path "PixArt-alpha/PixArt-XL-2-1024-MS"

数据目录结构：
    data_root/
      ├── rgb/      # RGB 纹理图 (.png / .jpg)
      ├── dem/      # DEM 高度图 (.npy / .png / .tif)
      └── txt/      # 文本提示词 (.txt)，与图片同名
"""

import argparse
import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

import torch

from train.train_pipeline import UNetTrainingPipeline, test_noise_prediction


# =============================================================================
# 默认超参数
# =============================================================================

_DEFAULT_DATA_ROOT = "./data/unet_training"
_DEFAULT_DEM_VAE_CKPT = "./data/vae_model_data/best_checkpoint.pt"
_DEFAULT_DIT_PRETRAINED = "PixArt-alpha/PixArt-XL-2-1024-MS"
_DEFAULT_OUTPUT_DIR = "/root/autodl-fs/dit_outputs"
_DEFAULT_EPOCHS = 50
_DEFAULT_BATCH_SIZE = 2
_DEFAULT_LEARNING_RATE = 5e-5
_DEFAULT_WEIGHT_DECAY = 1e-4
_DEFAULT_NUM_WORKERS = 2
_DEFAULT_USE_AMP = True
_DEFAULT_SAVE_STEPS = 1000
_DEFAULT_VIZ_INTERVAL = 1
_DEFAULT_WARMUP_EPOCHS = 5
_DEFAULT_STAGE1_EPOCHS = 10
_DEFAULT_SEED = 45
_DEFAULT_SPLIT = 0.05
_DEFAULT_NUM_SAMPLES = 5

_DEFAULT_CFG_DROP_RATE = 0.1
_DEFAULT_MIN_SNR_GAMMA = 5.0
_DEFAULT_GUIDANCE_SCALE = 4.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiT 8 通道联合去噪训练")

    parser.add_argument("--data_root", type=str, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--dem_vae_ckpt", type=str, default=_DEFAULT_DEM_VAE_CKPT,
                        help="高度图 VAE checkpoint 路径")
    parser.add_argument("--pretrained_path", type=str,
                        default=_DEFAULT_DIT_PRETRAINED,
                        help="PixArt-alpha 预训练权重路径")
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=_DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=_DEFAULT_NUM_WORKERS)
    parser.add_argument("--use_amp", action="store_true", default=_DEFAULT_USE_AMP)
    parser.add_argument("--save_steps", type=int, default=_DEFAULT_SAVE_STEPS)
    parser.add_argument("--viz_interval", type=int, default=_DEFAULT_VIZ_INTERVAL)
    parser.add_argument("--warmup_epochs", type=int, default=_DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="断点续训或测试使用的 checkpoint 路径")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"],
                        help="运行模式")
    parser.add_argument("--stage1_epochs", type=int, default=_DEFAULT_STAGE1_EPOCHS,
                        help="Stage 1 burn-in epoch 数（仅训练适配层）")

    parser.add_argument("--cfg_drop_rate", type=float, default=_DEFAULT_CFG_DROP_RATE,
                        help="无分类器引导(CFG)的条件丢弃概率 (默认 0.1)")
    parser.add_argument("--min_snr_gamma", type=float, default=_DEFAULT_MIN_SNR_GAMMA,
                        help="Min-SNR 加权策略的截断阈值 (默认 5.0，设为 0 关闭)")
    
    parser.add_argument("--seed", type=int, default = _DEFAULT_SEED)

    parser.add_argument("--split", type=float, default = _DEFAULT_SPLIT)

    parser.add_argument("--num_sample", type=int, default = _DEFAULT_NUM_SAMPLES)

    return parser


def _get_adapter_param_names(model: torch.nn.Module) -> set:
    """返回适配层参数名的集合（用于 10x lr）。

    PixArt 预训练层：transformer_blocks.*, pos_embed, time_proj, time_embedding,
                     patch_embed (部分), final_norm, final_linear (部分)
    CLIP 适配层：global_text_proj, local_text_proj
    """
    adapter_patterns = [
        "global_text_proj",
        "local_text_proj",
    ]
    adapter_params = set()
    for name, _ in model.named_parameters():
        if any(pat in name for pat in adapter_patterns):
            adapter_params.add(name)
    return adapter_params


def setup_staged_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    stage: int = 1,
) -> torch.optim.AdamW:
    """构建分阶段优化器。

    Stage 1: 冻结预训练层（transformer_blocks, pos_embed, time_*, patch_embed,
             final_norm, final_linear），仅训练 adapter 层（global_text_proj,
             local_text_proj），使用 10x lr。
    Stage 2: 全部解冻，adapter 层用 10x lr，预训练层用 base lr。
    """
    adapter_names = _get_adapter_param_names(model)

    if stage == 1:
        # Freeze all, then unfreeze only adapters
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name in adapter_names:
                param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        params_to_opt = [{"params": trainable, "lr": lr * 10}]
        print(f"Stage 1: 训练 {len(trainable)}/"
              f"{sum(1 for _ in model.parameters())} 个参数组（仅适配层）")

    else:
        for param in model.parameters():
            param.requires_grad = True

        adapter_param_objs = []
        pretrained_param_objs = []
        for name, param in model.named_parameters():
            if name in adapter_names:
                adapter_param_objs.append(param)
            else:
                pretrained_param_objs.append(param)

        params_to_opt = [
            {"params": pretrained_param_objs, "lr": lr},
            {"params": adapter_param_objs, "lr": lr * 10},
        ]
        print(f"Stage 2: 全部参数可训练，adapter 层 {len(adapter_param_objs)} 组 "
              f"@ {lr*10:.2e}, pretrained {len(pretrained_param_objs)} 组 @ {lr:.2e}")

    return torch.optim.AdamW(params_to_opt, weight_decay=weight_decay)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # DiT-specific flags
    args.model_type = "dit"
    args.dit_pretrained_path = args.pretrained_path

    # --- Test mode ---
    if args.mode == "test":
        if not args.checkpoint:
            parser.error("测试模式需要 --checkpoint 参数")
        pipeline = UNetTrainingPipeline(args)
        pipeline.load_checkpoint(args.checkpoint)
        test_noise_prediction(pipeline)

        pipeline.generate_validation_samples(epoch_or_name="test_inference", guidance_scale=_DEFAULT_GUIDANCE_SCALE)

        return

    # --- Train mode ---
    pipeline = UNetTrainingPipeline(args)

    # 1. 第一步：仅在 CPU 内存中“偷看” Checkpoint (耗时几秒，0 显存消耗)
    if args.checkpoint:
        print(f"正在读取 Checkpoint 元数据以确定训练阶段...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        start_epoch = ckpt["epoch"] + 1
    else:
        start_epoch = 0
        ckpt = None

    # 2. 第二步：根据读到的 epoch，推断当前属于哪个 Stage
    current_stage = 1 if (start_epoch < args.stage1_epochs) else 2
    
    print(f"\n{'='*60}")
    stage_name = "Burn-in 适配层" if current_stage == 1 else "全量微调"
    print(f"Stage {current_stage}: {stage_name} (从 Epoch {start_epoch} 开始)")
    print(f"{'='*60}")

    # 3. 第三步：构建与当前 Stage 匹配的优化器和调度器结构
    pipeline.optimizer = setup_staged_optimizer(
        pipeline.unet, args.learning_rate, args.weight_decay, stage=current_stage
    )
    
    total_stage_epochs = args.stage1_epochs if current_stage == 1 else (args.epochs - args.warmup_epochs)
    pipeline.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        pipeline.optimizer,
        T_max=max(1, total_stage_epochs - start_epoch),
        eta_min=args.learning_rate * 1e-3,
    )

    # 4. 第四步：结构就绪，安全地把数据灌入 GPU 模型和优化器中
    if ckpt:
        print("正在将权重拷贝至 GPU 显存...")
        # load_state_dict 会自动把 CPU 里的权重 copy 到模型所在的 GPU 上，极其安全
        pipeline.unet.load_state_dict(ckpt["model_state_dict"])
        pipeline.global_step = ckpt.get("global_step", 0)
        pipeline.best_loss = ckpt.get("best_loss", float("inf"))
        pipeline.loss_history = ckpt.get("loss_history", [])
        
        if pipeline.scaler and "scaler_state_dict" in ckpt:
            pipeline.scaler.load_state_dict(ckpt["scaler_state_dict"])

        try:
            pipeline.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            pipeline.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"成功恢复 Stage {current_stage} 的优化器和调度器动量状态！")
        except ValueError as e:
            print(f"[警告] 参数组发生变化（可能跨阶段），安全跳过优化器状态恢复：{e}")
            
        # 极其重要：手工释放内存中的 5GB 字典，防止内存泄漏
        del ckpt 

    # 5. 第五步：正式起飞
    if current_stage == 1:
        pipeline.train(start_epoch=start_epoch, end_epoch=args.stage1_epochs)
        
        # 如果 Stage 1 跑完，目标 epoch 还没到，丝滑切入 Stage 2
        if args.stage1_epochs < args.epochs:
            print(f"\n{'='*60}")
            print(f"Stage 2: 全量微调 (Epochs {args.stage1_epochs}-{args.epochs - 1})")
            print(f"{'='*60}")
            pipeline.optimizer = setup_staged_optimizer(
                pipeline.unet, args.learning_rate, args.weight_decay, stage=2
            )
            pipeline.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                pipeline.optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs - args.stage1_epochs),
                eta_min=args.learning_rate * 1e-3,
            )
            pipeline.train(start_epoch=args.stage1_epochs)
    else:
        # 如果断点本身就在 Stage 2，直接跑完剩下的
        pipeline.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
