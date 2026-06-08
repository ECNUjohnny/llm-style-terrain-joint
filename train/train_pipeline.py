"""
8 通道去噪模型训练流水线 — 已搁置

该文件包含
独立的 UNetTrainer 类和完整训练逻辑。本模块保留为占位，供未来统一训练
接口时参考或重构。

历史使用方式（已不再支持）：
    python scripts/unet/train_unet_full.py --epochs 50   # UNet
    python scripts/dit/train_dit_full.py --epochs 50     # DiT
"""


class UNetTrainingPipeline:
    """
    训练流水线占位。

    该流水线已搁置。U-Net 训练请使用 scripts/unet/unet_full.py；
    DiT 训练请使用 scripts/dit/train_dit_full.py。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "UNetTrainingPipeline 已搁置。\n"
            "U-Net 训练请使用: python scripts/unet/unet_full.py --mode train\n"
            "DiT 训练请使用:  python scripts/dit/train_dit_full.py --mode train"
        )


def test_noise_prediction(*args, **kwargs):
    """
    噪声预测测试占位。

    该函数已搁置。测试请使用对应训练脚本的 --mode test 参数：
        python scripts/unet/unet_full.py --mode test --checkpoint <path>
    """
    raise NotImplementedError(
        "test_noise_prediction 已搁置。\n"
        "请使用: python scripts/unet/unet_full.py --mode test --checkpoint <path>"
    )
