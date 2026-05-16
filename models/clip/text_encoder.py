"""
双分支 CLIP 文本编码器
"""

import torch
import torch.nn as nn
from typing import Tuple, List
# 引入 Hugging Face 的分词器和 CLIP 文本模型
from transformers import CLIPTokenizer, CLIPTextModel

class DualBranchCLIPEncoder(nn.Module):
    """
    双分支 CLIP 文本编码器

    将文本 Prompt 编码为两种特征：
    1. 全局特征向量：捕捉整体语义
    2. 细节特征向量：捕捉局部细节
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        global_dim: int = 768,
        local_dim: int = 768,
    ):
        """
        初始化双分支 CLIP 编码器
        """
        super().__init__()

        # 1. 加载预训练的 Tokenizer 和 CLIP 模型
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip_model = CLIPTextModel.from_pretrained(model_name)

        # 关键点：冻结 CLIP 模型的参数，防止在训练 UNet 时更新它，节省大量显存
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 获取 CLIP 模型默认的隐藏层维度（base 通常是 512，large 通常是 768）
        hidden_size = self.clip_model.config.hidden_size

        # 2. 全局特征投影层
        self.global_proj = nn.Linear(hidden_size, global_dim)

        # 3. 细节特征投影层
        self.local_proj = nn.Linear(hidden_size, local_dim)

    def forward(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            prompts: Prompt 字符串列表
        Returns:
            global_features: 全局特征向量 [B, global_dim]
            local_features: 细节特征向量 [B, N, local_dim]
        """
        
        # 1. 获取当前模型所在的设备 (GPU/CPU)
        device = self.clip_model.device

        # 2. 分词：将文本字符串变成模型能看懂的 Token ID
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77, # CLIP 的标准最大长度
            return_tensors="pt",
        ).to(device)       # 确保数据和模型在同一个显卡上

        # 3. 送入 CLIP 提取特征（使用 torch.no_grad() 进一步确保不计算梯度）
        with torch.no_grad():
            outputs = self.clip_model(**text_inputs)

        # 4. 提取并映射【全局特征】
        # pooler_output 通常是句子结尾符 [EOS] 经过特定映射后的结果，代表全句语义
        global_features = outputs.pooler_output  # 形状: [B, hidden_size]
        global_features = self.global_proj(global_features)  # 形状: [B, global_dim]

        # 5. 提取并映射【细节特征】
        # last_hidden_state 包含了输入句子中所有 77 个 Token 的独立特征
        local_features = outputs.last_hidden_state  # 形状: [B, 77, hidden_size]
        local_features = self.local_proj(local_features)  # 形状: [B, 77, local_dim]

        return global_features, local_features


def build_text_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
) -> DualBranchCLIPEncoder:
    """
    构建文本编码器工厂函数
    """
    return DualBranchCLIPEncoder(model_name=model_name)