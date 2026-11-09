"""黑米训练入口。

    python train.py --config configs/ladder.yaml --run heimi            # 按顺序跑完全部阶段，有检查点自动续训
    python train.py --config configs/ladder.yaml --run heimi --stage stage0
    python train.py --config configs/baseline105m.yaml --run baseline
    python train.py --config configs/tiny_smoke.yaml --run smoke --max-steps 12   # 试跑 12 步后保存检查点退出

日志与检查点写到 <runs-dir>/<run>/。训练期间阻止系统待机。中止（NaN、显存不足）返回码 3。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from minids.eval.hellaswag import download_hellaswag
from minids.keepawake import keep_awake
from minids.train.config import load_ladder
from minids.train.loop import TrainingAborted, run_ladder

log = logging.getLogger("train")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="黑米：按阶梯配置逐级训练")
    p.add_argument("--config", required=True, help="阶梯配置 yaml")
    p.add_argument("--run", required=True, help="run 名，日志与检查点写到 <runs-dir>/<run>/")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--stage", default=None, help="只跑这一级（要求前一级的最终检查点存在）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps", type=int, default=None, help="本次进程最多执行的优化步数，到达后保存检查点退出")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_ladder(args.config)
    if cfg.eval.hellaswag and not Path(cfg.data.hellaswag).exists():
        log.info("下载 HellaSwag 验证集到 %s", cfg.data.hellaswag)
        download_hellaswag(Path(cfg.data.hellaswag))

    run_dir = Path(args.runs_dir) / args.run
    try:
        with keep_awake():
            run_ladder(cfg, run_dir, args.device, only_stage=args.stage, max_steps=args.max_steps)
    except TrainingAborted as e:
        log.error("训练中止：%s", e)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
