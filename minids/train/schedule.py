"""学习率调度：线性预热到峰值，再余弦衰减到峰值的 min_ratio（spec 8.2 为 10%）。

无状态：学习率只是 step 的函数，恢复训练不需要保存调度器状态。
"""

from __future__ import annotations

import math


def warmup_steps(total_steps: int, warmup_frac: float) -> int:
    """预热步数，至少 1 步（级 0 占 2%，其余级占 5%）。"""
    return max(1, round(total_steps * warmup_frac))


def lr_at(step: int, total_steps: int, peak_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """step 从 0 计；预热段 peak × (step + 1) / warmup，最后一个预热步恰好到峰值；之后余弦到 min_ratio × peak，超出 total 后钳在最小值。"""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    floor = min_ratio * peak_lr
    return floor + 0.5 * (peak_lr - floor) * (1.0 + math.cos(math.pi * progress))
