"""优化器：AdamW，β1 0.9、β2 0.95，权重衰减 0.1；词嵌入与 RMSNorm 缩放参数不衰减（spec 8.2）。

分组规则：二维以上的权重（各线性层，含输出头）衰减；一维参数（RMSNorm 缩放）和词嵌入不衰减。
参数在 GPU 上时用 fused 实现。每级开始时重新构造，优化器状态不跨级迁移（spec 7.4）。
"""

from __future__ import annotations

import torch

from minids.model.transformer import Transformer


def build_optimizer(
    model: Transformer,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    emb = model.tok_emb.weight
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2 and p is not emb]
    no_decay = [p for p in params if p.ndim < 2 or p is emb]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=betas,
        fused=params[0].is_cuda,
    )
