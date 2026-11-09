"""旋转位置编码（RoPE）。

只作用于 MLA 中解耦出来的 rope 维度。采用 rotate-half 约定：
把向量前后两半配对旋转，第 i 维与第 i + dim/2 维共用同一个频率。
"""

import torch
from torch import Tensor

RopeCache = tuple[Tensor, Tensor]  # (cos, sin)，形状各为 (seq_len, dim)


def precompute_rope(
    dim: int,
    seq_len: int,
    base: float = 10000.0,
    device=None,
    dtype=torch.float32,
) -> RopeCache:
    if dim % 2 != 0:
        raise ValueError(f"RoPE 维度必须是偶数，得到 {dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, h, T, dim)；cos/sin: (T, dim)，调用方已切到当前长度。"""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + _rotate_half(x) * sin
