"""官方评测测试共用：手写词表的分词器、按脚本打分的假模型、迷你代码评测数据、造记录的小函数。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from minids.official.codedata import INFILL_LOCAL, REPOBENCH_LOCAL


class PieceTok:
    """手写词表的迷你分词器：id 0-15 是特殊 token，其余就是给定的字符串；encode 每次贪心取最长的那块。

    用来精确造出"换行 + 缩进是一个 token"这类情形，真分词器训出来的小词表控制不了这个。
    """

    def __init__(self, pieces):
        self.pieces = [f"<|s{i}|>" for i in range(16)] + list(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def id(self, piece: str) -> int:
        return self.pieces.index(piece)

    def encode(self, text: str) -> list[int]:
        ids = []
        while text:
            best = max((i for i in range(16, len(self.pieces)) if text.startswith(self.pieces[i])), key=lambda i: len(self.pieces[i]))
            ids.append(best)
            text = text[len(self.pieces[best]) :]
        return ids

    def decode(self, ids) -> str:
        return "".join(self.pieces[i] for i in ids)


def ranked_logits(script: list[list[int]], step: int, vocab_size: int) -> torch.Tensor:
    """第 step 个新位置的打分：script[step] 里越靠前的 token 分越高，其余都是 -1；脚本用完后重复最后一项。"""
    ranked = script[min(step, len(script) - 1)]
    logits = torch.full((vocab_size,), -1.0)
    for rank, t in enumerate(ranked):
        logits[t] = float(len(ranked) - rank)
    return logits


class RankedModel(nn.Module):
    """像黑米那样说话的假模型：forward 返回 (logits, None)，generate 每步取 argmax。

    新位置的序号 = 收到的序列长度 − 第一次收到的长度，打分见 ranked_logits；seen 记下每次 forward 收到的序列。
    """

    def __init__(self, script: list[list[int]], vocab_size: int, seq_len: int = 64):
        super().__init__()
        self.script = script
        self.cfg = SimpleNamespace(seq_len=seq_len, vocab_size=vocab_size)
        self.start: int | None = None
        self.seen: list[list[int]] = []
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        if self.start is None:
            self.start = idx.shape[1]
        self.seen.append(idx[0].tolist())
        logits = torch.full((1, idx.shape[1], self.cfg.vocab_size), -1.0)
        logits[0, -1] = ranked_logits(self.script, idx.shape[1] - self.start, self.cfg.vocab_size)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            idx = torch.cat([idx, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
        return idx


def write_code_data(d: Path) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    n = 3
    pq.write_table(
        pa.table({
            "repo_name": ["o/r"] * n, "file_path": [f"m{i}.py" for i in range(n)], "import_statement": ["import os"] * n,
            "cropped_code": ["def f():\n    x = 1\n"] * n, "next_line": ["    return x"] * n, "level": ["2k"] * n,
            "all_code": ["unused"] * n,
        }),
        d / REPOBENCH_LOCAL,
    )
    rows = [
        {"task_id": f"t{i}", "entry_point": "f", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n", "test": "unused"}
        for i in range(n)
    ]
    (d / INFILL_LOCAL).write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return d


def official_record(checkpoint, tokens_total, limit=None, code_limit=500, code_method="healed", **metrics):
    from minids.official.results import OfficialRecord

    return OfficialRecord(
        checkpoint=checkpoint, stage=checkpoint.split("_")[0], global_step=tokens_total // 100, tokens_total=tokens_total,
        params_non_embedding=1000, limit=limit, code_limit=code_limit, metrics=metrics or {"blimp_acc": 0.5},
        harness_version="0.4.13", time="2026-11-09T12:00:00-05:00", code_method=code_method,
    )


def legacy_record(*args, **kwargs) -> dict:
    """2026-11-21 第一次评测留下的那种记录：没有 code_method 这一项。"""
    from dataclasses import asdict

    rec = asdict(official_record(*args, **kwargs))
    del rec["code_method"]
    return rec
