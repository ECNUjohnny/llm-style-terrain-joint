"""
推理流水线 — 已搁置

本模块为骨架占位。训练后的模型推理请直接使用训练脚本的 --mode test:
    python scripts/unet/unet_full.py --mode test --checkpoint <path>

历史设计文档保留如下供参考。
"""


class InferencePipeline:
    """
    推理流水线骨架 (已搁置)。

    原始设计目标:
        输入: Prompt 字符串
        输出: (512x512 高度图, 512x512 纹理图)

    流程: 文本编码 → 初始化噪声 → DDIM 循环降噪 → VAE 解码
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "InferencePipeline 已搁置。\n"
            "请使用: python scripts/unet/unet_full.py --mode test --checkpoint <path>"
        )
