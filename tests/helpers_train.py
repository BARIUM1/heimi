"""训练与评测测试共用的小工具：两个桩模型（本文件后半部分在 Task 8 补数据与配置构造）。"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import LongTensor, nn


class ScriptedModel(nn.Module):
    """generate 每次调用按脚本追加下一个 token，脚本用尽后重复最后一个。只用于测试停止规则。"""

    def __init__(self, script: list[int], seq_len: int = 64):
        super().__init__()
        self.script = list(script)
        self.pos = 0
        self.cfg = SimpleNamespace(seq_len=seq_len)
        self._dummy = nn.Parameter(torch.zeros(1))

    def generate(self, idx: LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> LongTensor:
        out = idx
        for _ in range(max_new_tokens):
            tok = self.script[min(self.pos, len(self.script) - 1)]
            self.pos += 1
            out = torch.cat([out, torch.tensor([[tok]], dtype=out.dtype, device=out.device)], dim=1)
        return out


class CountingModel(nn.Module):
    """forward 返回 (logits, None)，logits 在 下一个 token = 当前 token + 1 处为 10，其余为 0。用于 HellaSwag 测试。"""

    def __init__(self, vocab_size: int, seq_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.cfg = SimpleNamespace(seq_len=seq_len, vocab_size=vocab_size)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx: LongTensor, targets: LongTensor | None = None):
        nxt = (idx + 1) % self.vocab_size
        logits = torch.zeros(*idx.shape, self.vocab_size, device=idx.device)
        logits.scatter_(-1, nxt.unsqueeze(-1), 10.0)
        return logits, None
