"""命令行 FIM 单行补全（spec 11.3）。

    echo def add(a, b): | python complete.py --model models/heimi-105m
    python complete.py --model models/heimi-105m --prefix-file p.txt --suffix-file s.txt --max-tokens 32
    python complete.py --ckpt runs/heimi/ckpt/task_final.pt --tokenizer data/tokenizer/heimi-16k.json --prefix-file p.txt

没有 --prefix-file 时从标准输入读前缀；没有 --suffix-file 时后缀为空。默认 CPU，--device cuda 可切 GPU。
把补出的那一行写到标准输出。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from minids.completion import complete_middle
from minids.data.tokenizer import HeimiTokenizer
from minids.export import load_exported
from minids.train.checkpoint import load_model


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="黑米：命令行 FIM 单行补全")
    p.add_argument("--model", default=None, help="export.py 导出的目录")
    p.add_argument("--ckpt", default=None, help="训练检查点（需配合 --tokenizer）")
    p.add_argument("--tokenizer", default="data/tokenizer/heimi-16k.json")
    p.add_argument("--prefix-file", default=None)
    p.add_argument("--suffix-file", default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if bool(args.model) == bool(args.ckpt):
        p.error("--model 与 --ckpt 必须且只能给一个")

    if args.model:
        model, tok = load_exported(Path(args.model), args.device)
    else:
        model, _ = load_model(Path(args.ckpt), args.device)
        tok = HeimiTokenizer.load(Path(args.tokenizer))

    prefix = Path(args.prefix_file).read_text(encoding="utf-8") if args.prefix_file else sys.stdin.read()
    suffix = Path(args.suffix_file).read_text(encoding="utf-8") if args.suffix_file else ""
    seq_len = model.cfg.seq_len
    max_prefix = min(512, seq_len // 2)
    max_suffix = min(256, seq_len // 4)
    max_new = min(args.max_tokens, seq_len - max_prefix - max_suffix - 3)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") else contextlib.nullcontext()
    with ctx:
        out = complete_middle(model, tok, prefix, suffix, max_new, args.device, max_prefix, max_suffix)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
