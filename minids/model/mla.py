"""MLA（Multi-head Latent Attention），DeepSeek-V2 的注意力。

与普通多头注意力的区别：K 和 V 不直接从输入投影，而是先压缩成一个
低维潜向量 c_kv，再升维出每个头的 K（不带位置部分）和 V。位置信息走一条
单独的"解耦"路径：所有头共用一个带 RoPE 的位置键 k_pe，Q 也拆出对应的
q_pe 加 RoPE。推理时只需缓存 c_kv 和 k_pe，这就是它省显存的原因。

本实现不做 Q 的低秩压缩，也不做推理期的权重吸收。
头的排布是 head-major：投影输出 view 成 (B, T, n_head, per_head)。
增长算子依赖这一点，也依赖下面这些属性名：wq、wdkv、kv_norm、wukv、wkpe、wo。
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minids.config import ModelConfig
from minids.model.norm import RMSNorm
from minids.model.rope import apply_rope


class MLA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = cfg.qk_head_dim
        self.v_head_dim = cfg.v_head_dim

        d, h = cfg.d_model, cfg.n_head
        self.wq = nn.Linear(d, h * self.qk_head_dim, bias=False)
        self.wdkv = nn.Linear(d, cfg.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)
        self.wukv = nn.Linear(cfg.kv_lora_rank, h * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wkpe = nn.Linear(d, self.qk_rope_head_dim, bias=False)
        self.wo = nn.Linear(h * self.v_head_dim, d, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """x: (B, T, d_model)；cos/sin: (T, qk_rope_head_dim)。"""
        B, T, _ = x.shape
        h = self.n_head

        # Q：拆成不带位置的 q_nope 和带 RoPE 的 q_pe
        q = self.wq(x).view(B, T, h, self.qk_head_dim).transpose(1, 2)  # (B, h, T, qk)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, cos, sin)

        # KV：压缩 → 归一化 → 升维
        c_kv = self.kv_norm(self.wdkv(x))  # (B, T, kv_lora_rank)
        kv = self.wukv(c_kv).view(B, T, h, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 解耦的位置键：所有头共用
        k_pe = apply_rope(self.wkpe(x).unsqueeze(1), cos, sin)  # (B, 1, T, rope)
        k_pe = k_pe.expand(B, h, T, self.qk_rope_head_dim)

        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe], dim=-1)
        out = F.scaled_dot_product_attention(
            q_full, k_full, v, is_causal=True, scale=1.0 / math.sqrt(self.qk_head_dim)
        )  # (B, h, T, v_head_dim)
        out = out.transpose(1, 2).reshape(B, T, h * self.v_head_dim)
        return self.wo(out)
