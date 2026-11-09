"""RMSNorm：按均方根归一化，再乘可学习的缩放参数。

计算在 fp32 里做，避免 bf16 下平方求和的精度损失，结果转回输入 dtype。
"""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)
