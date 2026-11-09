"""合成 token 分片，给 configs/tiny_smoke.yaml 用（spec 13.2）。

    python scripts/make_smoke_data.py [--out data/smoke] [--vocab 16384] [--tokens 400000] [--seed 0]

web 与 code 的 train/val 各一片随机 token（范围 [16, vocab)），每 300 个 token 放一个 <|endoftext|>，
并写出与 prepare.py 同格式的 index.json。随机 token 训不出语言，只用来检查流水线能跑、损失会降、增长与恢复正确。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minids.data.prepare import write_index  # noqa: E402


def write_smoke_data(out: Path, vocab: int, tokens: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    per_source = {}
    for src in ("web", "code"):
        d = Path(out) / src
        d.mkdir(parents=True, exist_ok=True)
        entries = {}
        for split in ("train", "val"):
            arr = rng.integers(16, vocab, size=tokens, dtype=np.int64)
            arr[299::300] = 0  # 文档结束符
            name = f"{split}_0000.bin"
            arr.astype(np.uint16).tofile(d / name)
            entries[split] = [{"file": name, "tokens": int(tokens)}]
        per_source[src] = {
            "docs": tokens // 300, "skipped": 0, "tokens_train": tokens, "tokens_val": tokens, "val_rows": 0,
            "train": entries["train"], "val": entries["val"],
        }
    return write_index(Path(out), per_source, vocab)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成烟雾测试用的合成 token 分片")
    p.add_argument("--out", default="data/smoke")
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--tokens", type=int, default=400_000, help="每来源每 split 的 token 数")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    idx = write_smoke_data(Path(args.out), args.vocab, args.tokens, args.seed)
    print(f"index 写入 {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
