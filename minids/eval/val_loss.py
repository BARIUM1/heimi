"""验证损失：web 与 code 各取固定的顺序窗口（spec 10：各 200 个，约 20 万 token），分别报告。

窗口从验证分片开头按顺序切，不随机，所以每次评测比的是同一批 token。
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F

from minids.data.loader import ShardSet
from minids.model.transformer import Transformer


def fixed_val_windows(index_path: Path, source: str, n_windows: int, seq_len: int) -> torch.LongTensor:
    """(n_windows, seq_len + 1) 的 int64 张量，来自该来源验证分片的开头。"""
    shards = ShardSet.from_index(Path(index_path), source, "val", seq_len + 1)
    return torch.from_numpy(shards.sequential_windows(seq_len + 1, n_windows))


@torch.no_grad()
def val_loss(model: Transformer, windows: torch.LongTensor, device: str, micro_batch: int) -> float:
    """所有窗口所有位置的平均交叉熵。cuda 上用 bf16 autocast；结束后恢复原来的训练模式。"""
    was_training = model.training
    model.eval()
    use_autocast = str(device).startswith("cuda")
    total, count = 0.0, 0
    for i in range(0, windows.shape[0], micro_batch):
        batch = windows[i : i + micro_batch].to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()
        with ctx:
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total += loss.item()
        count += y.numel()
    model.train(was_training)
    return total / count
