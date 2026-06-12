"""
8 通道联合隐向量 DiT 扩散模型

Facebook DiT 架构 (Peebles & Xie, 2023)，adaLN-Zero 初始化 + Flow Matching。
专为 joint latent [B,8,64,64] 设计，从零训练，无预训练权重依赖。

条件注入: CLIP pooler [B,768] → adaLN 调制
          CLIP hidden [B,77,768] → cross-attention K/V

用法:
    dit = DiT(in_channels=8, out_channels=8, depth=18, hidden_size=1024, num_heads=16)
    output = dit(sample=noisy_latent, timestep=t, encoder_hidden_states=hidden, pooler_output=pooler)
    velocity_pred = output.sample  # [B, 8, 64, 64]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# 工具函数
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN 调制: x * (1 + scale) + shift，支持 [B,D] 广播到 [B,N,D]"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# 模型组件
# =============================================================================

class TimestepEmbedder(nn.Module):
    """正弦时间编码 + 2 层 MLP 投影"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class PatchEmbed(nn.Module):
    """8 通道 → patch token 序列: Conv2d(k=2,s=2) → [B, 1024, hidden_size]"""

    def __init__(self, in_channels: int = 8, hidden_size: int = 1024, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (64 // patch_size) ** 2  # 32x32 = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                     # [B, hidden, 32, 32]
        x = x.flatten(2).transpose(1, 2)     # [B, 1024, hidden]
        return x


class AdaLNModulation(nn.Module):
    """adaLN-Zero: 从条件向量 c 生成 6 组调制参数，最后层零初始化"""

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 6 * hidden_size)

    def forward(self, c: torch.Tensor):
        # c: [B, hidden]
        x = self.silu(self.linear_1(c))
        x = self.linear_2(x)  # [B, 6*hidden]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = x.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DiTBlock(nn.Module):
    """DiT Block: Self-Attn (gated) + Cross-Attn + MLP (gated)，均 pre-norm"""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        # adaLN-Zero 调制
        self.adaLN = AdaLNModulation(hidden_size)

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=0.0, batch_first=True,
        )

        # Cross-attention (K/V from projected CLIP hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            kdim=hidden_size, vdim=hidden_size,
            dropout=0.0, batch_first=True,
        )

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, N, hidden], c: [B, hidden], ctx: [B, L, hidden] or None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)

        # Self-attention (gated)
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Cross-attention (non-gated, skip if no context)
        if ctx is not None:
            ca_out, _ = self.cross_attn(self.norm2(x), ctx, ctx, need_weights=False)
            x = x + ca_out

        # MLP (gated)
        norm_x = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class FinalLayer(nn.Module):
    """输出层: LayerNorm → Linear → unpatchify，零初始化"""

    def __init__(self, hidden_size: int = 1024, out_channels: int = 8, patch_size: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels * patch_size * patch_size)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.spatial_size = 64 // patch_size  # 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1024, hidden] → [B, 1024, C*P²] → unpatchify → [B, C, 64, 64]
        x = self.norm(x)
        x = self.linear(x)  # [B, 1024, 32]
        B, N, _ = x.shape
        P = self.patch_size
        C = self.out_channels
        H = W = self.spatial_size
        x = x.reshape(B, H, W, P, P, C)       # [B, H, W, P, P, C]
        x = x.permute(0, 5, 1, 3, 2, 4)       # [B, C, H, P, W, P]
        x = x.reshape(B, C, H * P, W * P)     # [B, C, 64, 64]
        return x


class DiTOutput:
    """兼容 UNet2DConditionModel 输出接口"""
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


# =============================================================================
# 顶层 DiT 模型
# =============================================================================

class DiT(nn.Module):
    """
    8 通道联合隐向量 DiT 扩散模型

    Args:
        in_channels:  输入通道数 (8: RGB 4ch + DEM 4ch)
        out_channels: 输出通道数 (8)
        hidden_size:  Transformer 隐藏维度
        depth:        DiT block 层数
        num_heads:    注意力头数
        mlp_ratio:    MLP 扩展比
        patch_size:   patch 大小
        cross_attention_dim: CLIP 特征维度 (ViT-L/14 = 768)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        hidden_size: int = 1024,
        depth: int = 18,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        cross_attention_dim: int = 768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        # 输入
        self.patch_embed = PatchEmbed(in_channels, hidden_size, patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, hidden_size)
        )

        # 时间编码
        self.time_embed = TimestepEmbedder(hidden_size)

        # CLIP 文本投影 (768 → hidden_size)
        self.global_text_proj = nn.Linear(cross_attention_dim, hidden_size)
        self.local_text_proj = nn.Linear(cross_attention_dim, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 输出
        self.final_layer = FinalLayer(hidden_size, out_channels, patch_size)

        self._init_weights()

    def _init_weights(self):
        # 1. 标准初始化: Conv2d, Linear, LayerNorm
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 2. Position embedding: 较小 std
        nn.init.normal_(self.pos_embed, std=0.02)

        # 3. adaLN-Zero: 每个 block 的调制最后层归零
        for block in self.blocks:
            nn.init.constant_(block.adaLN.linear_2.weight, 0)
            nn.init.constant_(block.adaLN.linear_2.bias, 0)

        # 4. 输出层归零
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooler_output: Optional[torch.Tensor] = None,
    ) -> DiTOutput:
        """
        Args:
            sample:               带噪联合隐向量 [B, 8, 64, 64]
            timestep:             连续时间 t ∈ [0, 1], shape [B]
            encoder_hidden_states: CLIP last_hidden_state [B, 77, 768]
            pooler_output:         CLIP pooler_output [B, 768] (可选，CFG 时可为 None)
        Returns:
            DiTOutput(sample=velocity_pred) [B, 8, 64, 64]
        """
        # Patch embed
        x = self.patch_embed(sample)       # [B, 1024, 1024]
        x = x + self.pos_embed

        # Timestep condition
        c = self.time_embed(timestep)      # [B, 1024]

        # adaLN modulation from global CLIP pooler
        if pooler_output is not None:
            c = c + self.global_text_proj(pooler_output)

        # Cross-attn context from local CLIP hidden
        ctx = None
        if encoder_hidden_states is not None:
            ctx = self.local_text_proj(encoder_hidden_states)  # [B, 77, 1024]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ctx)

        # Output
        x = self.final_layer(x)
        return DiTOutput(sample=x)

    def loss(self, velocity_pred: torch.Tensor, velocity_target: torch.Tensor) -> tuple:
        """
        Flow Matching 速度场损失，DEM 通道 1.5x 权重。

        Returns:
            (total_loss, loss_rgb, loss_dem)
        """
        loss_rgb = F.mse_loss(velocity_pred[:, :4], velocity_target[:, :4])
        loss_dem = F.mse_loss(velocity_pred[:, 4:], velocity_target[:, 4:])
        total_loss = loss_rgb + 1.5 * loss_dem
        return total_loss, loss_rgb, loss_dem
