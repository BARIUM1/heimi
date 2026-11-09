"""文本生成。

    python sample.py --ckpt runs/heimi/ckpt/stage0_final.pt --prompt "The history of" [--max-tokens 64]
                     [--temperature 0.8] [--top-k 50] [--seed 0] [--device cuda] [--tokenizer data/tokenizer/heimi-16k.json]

temperature 0 为贪心。打印提示加续写。
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Sequence
from pathlib import Path

import torch

from minids.data.tokenizer import HeimiTokenizer
from minids.train.checkpoint import load_model


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="黑米：用检查点续写文本")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/heimi-16k.json")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0 表示不截断")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    model, _ = load_model(Path(args.ckpt), args.device)
    tok = HeimiTokenizer.load(Path(args.tokenizer))
    torch.manual_seed(args.seed)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") else contextlib.nullcontext()
    with ctx:
        out = model.generate(ids, args.max_tokens, temperature=args.temperature, top_k=args.top_k or None)
    print(args.prompt + tok.decode(out[0, ids.shape[1] :].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
