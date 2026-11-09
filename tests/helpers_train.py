"""训练与评测测试共用的小工具：两个桩模型、合成数据、迷你阶梯配置。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import LongTensor, nn

from minids.config import ModelConfig
from minids.data.loader import MixConfig
from minids.data.prepare import write_index
from minids.data.tokenizer import EOT, HeimiTokenizer
from minids.train.config import DataPaths, EvalConfig, LadderConfig, Prompt, StageConfig
from tests.helpers_data import tiny_tokenizer, write_shard

# 每篇 ≥ 3 个非空行，给 FIM 单行评测抽样用
CODE_DOCS: list[str] = [
    "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i)\n",
    "import os\n\nfiles = os.listdir('.')\nfor f in files:\n    print(f)\n",
    "x = 1\ny = 2\nz = x + y\nprint(z)\n",
    "#!/bin/sh\nset -e\necho start\nls -la\necho done\n",
    "const a = 1;\nconst b = 2;\nconsole.log(a + b);\n",
    "class Cat:\n    def __init__(self, name):\n        self.name = name\n\n    def meow(self):\n        return 'meow'\n",
]


def write_tiny_data(root: Path, tok: HeimiTokenizer, seed: int = 0, train_tokens: int = 20_000) -> Path:
    """root/tokens/{web,code}/train_0000.bin 为随机 token，val_0000.bin 为 CODE_DOCS 编码后以 0 分隔；返回 index.json 路径。"""
    rng = np.random.default_rng(seed)
    vocab = tok.vocab_size
    val_ids: list[int] = []
    while len(val_ids) < 3000:
        for d in CODE_DOCS:
            val_ids += tok.encode(d) + [EOT]
    per_source = {}
    for src in ("web", "code"):
        d = Path(root) / "tokens" / src
        train = rng.integers(16, vocab, size=train_tokens)
        write_shard(d / "train_0000.bin", train.tolist())
        write_shard(d / "val_0000.bin", val_ids)
        per_source[src] = {
            "docs": len(CODE_DOCS), "skipped": 0, "tokens_train": train_tokens, "tokens_val": len(val_ids), "val_rows": 0,
            "train": [{"file": "train_0000.bin", "tokens": train_tokens}],
            "val": [{"file": "val_0000.bin", "tokens": len(val_ids)}],
        }
    return write_index(Path(root) / "tokens", per_source, vocab)


def tiny_ladder(
    root: Path,
    stage_steps: tuple[int, ...] = (3, 2, 2),
    ckpt_interval: int = 2,
    log_interval: int = 1,
    eval_interval: int = 2,
) -> LadderConfig:
    """迷你阶梯：seq_len 32、微批 4、每步 256 token（累积 2 微步）；stage_steps 给 1 到 4 个数，依次为
    smoke0 none / smoke1 depth / smoke2 width / smoke_task none（mix 0.2/0.8，fim 0.5）。"""
    root = Path(root)
    tok = tiny_tokenizer(root)
    index = write_tiny_data(root, tok)
    base = ModelConfig(
        vocab_size=tok.vocab_size, seq_len=32, n_layer=2, d_model=64, n_head=2,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=32, kv_lora_rank=32, ffn_hidden=128,
    )
    wide = base.grown_depth().grown_width()
    models = [base, base.grown_depth(), wide, wide]
    names = ["smoke0", "smoke1", "smoke2", "smoke_task"]
    growths = ["none", "depth", "width", "none"]
    mixes = [MixConfig(0.5, 0.5, 0.0)] * 3 + [MixConfig(0.2, 0.8, 0.5)]
    stages = tuple(
        StageConfig(names[i], growths[i], models[i], steps * 256, 1e-3, 0.2, 4, mixes[i])
        for i, steps in enumerate(stage_steps)
    )
    return LadderConfig(
        seed=0, seq_len=32, tokens_per_step=256, stages=stages,
        prompts=(Prompt("en", "The cat"), Prompt("fim", "def f():\n    ", "\n")),
        data=DataPaths(index=str(index), tokenizer=str(root / "tok300.json"), hellaswag=str(root / "hellaswag_val.parquet")),
        eval=EvalConfig(interval=eval_interval, val_windows=4, hellaswag=False, fim_em_samples=3,
                        sample_interval=eval_interval, sample_tokens=4),
        log_interval=log_interval, ckpt_interval=ckpt_interval, keep_checkpoints=2,
    )


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
