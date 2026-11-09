"""显存守卫。

Windows 上 CUDA 超过物理显存不会抛 OutOfMemoryError，而是把一部分搬到系统内存，
速度掉一到两个数量级。所以训练和基准都要自己盯住峰值显存。
看 reserved（驱动实际持有的）而不是 allocated：allocated 会低估真实占用。
"""

import torch


def cuda_memory_snapshot() -> dict[str, float]:
    """返回自上次 reset_peak_memory_stats 以来的峰值 allocated / reserved 和物理显存总量，单位 GB。没有 CUDA 时全为 0。"""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def exceeds_memory_budget(reserved_gb: float, total_gb: float, max_frac: float = 0.8) -> bool:
    """reserved 超过物理显存的 max_frac 就判定为即将或已经溢出。"""
    return total_gb > 0 and reserved_gb > max_frac * total_gb
