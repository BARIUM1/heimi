"""导出模型目录：python export.py --ckpt runs/heimi/ckpt/task_final.pt --out models/heimi-105m [--tokenizer data/tokenizer/heimi-16k.json]"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from minids.export import export_model

log = logging.getLogger("export")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="黑米：把检查点导出成 safetensors 模型目录")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/heimi-16k.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = export_model(args.ckpt, args.tokenizer, args.out)
    log.info("已导出到 %s", out)  # 走 stderr，标准输出留给管道里的下一个程序
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
