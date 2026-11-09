"""对任意检查点跑全部评测（spec 10）。

    python evaluate.py --ckpt runs/heimi/ckpt/stage0_final.pt [--config configs/ladder.yaml] [--device cuda]
                       [--no-hellaswag] [--fim-samples N] [--out runs/heimi/eval.jsonl]

数据路径与评测设置取自 --config；stage、global_step、tokens_total 取自检查点里的训练状态。
结果追加到该 run 的 eval.jsonl（默认 ckpt 所在 run 目录）并把 JSON 打印到标准输出。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch

from minids.data.tokenizer import HeimiTokenizer
from minids.eval.hellaswag import download_hellaswag
from minids.eval.suite import EvalSuite
from minids.logging_io import EVAL_FILE, EvalRecord, append_jsonl, now_iso
from minids.train.checkpoint import load_model
from minids.train.config import load_ladder

log = logging.getLogger("evaluate")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="黑米：对检查点跑验证损失、HellaSwag、FIM 精确匹配")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/ladder.yaml", help="提供数据路径与评测设置")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-hellaswag", action="store_true")
    p.add_argument("--fim-samples", type=int, default=None, help="覆盖配置里的 eval.fim_em_samples")
    p.add_argument("--out", default=None, help="追加到哪个 eval.jsonl，默认 ckpt 所在 run 目录")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_ladder(args.config)
    eval_cfg = cfg.eval
    if args.fim_samples is not None:
        eval_cfg = replace(eval_cfg, fim_em_samples=args.fim_samples)
    if args.no_hellaswag:
        eval_cfg = replace(eval_cfg, hellaswag=False)
    cfg = replace(cfg, eval=eval_cfg)
    if cfg.eval.hellaswag and not Path(cfg.data.hellaswag).exists():
        download_hellaswag(Path(cfg.data.hellaswag))

    ckpt = Path(args.ckpt)
    model, meta = load_model(ckpt, args.device)
    state = meta["state"]
    tok = HeimiTokenizer.load(Path(cfg.data.tokenizer))
    suite = EvalSuite(cfg, tok, args.device)
    names = [s.name for s in cfg.stages]
    micro_batch = cfg.stage(state["stage"]).micro_batch if state["stage"] in names else 8
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") else contextlib.nullcontext()
    with ctx:
        res = suite.run(model, micro_batch=micro_batch, hellaswag=cfg.eval.hellaswag, fim=True)
    rec = EvalRecord(
        state["stage"], state["global_step"], state["tokens_total"],
        res["val_loss_web"], res["val_loss_code"], res["hellaswag_acc"], res["fim_em"], now_iso(),
    )
    out = Path(args.out) if args.out else ckpt.parent.parent / EVAL_FILE
    append_jsonl(out, rec)
    log.info("已追加到 %s", out)
    print(json.dumps(asdict(rec), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
