"""官方基准评测命令行入口：训练出一个阶段后运行一次，把结果记进该 run 的 official.jsonl。

    python official_eval.py --run heimi                      # 评这个 run 里还没评过的最终检查点
    python official_eval.py --run heimi --ckpt stage1_final.pt --limit 50   # 指定检查点、小样本试跑
    python official_eval.py --reference gpt2                 # 评 GPT-2 small 作参照，写 docs/official/reference.json

退出码：0 全部成功；1 至少一个检查点评测失败；2 用法或前提不满足。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from minids.gpu import gpu_status
from minids.official.codedata import ensure_code_data
from minids.official.reference import evaluate_reference
from minids.official.results import (
    append_record,
    completeness,
    final_checkpoints,
    load_reference,
    pending_checkpoints,
    save_reference,
)
from minids.official.run import CODE_LIMIT, evaluate_checkpoint
from minids.train.checkpoint import CheckpointError

log = logging.getLogger("official_eval")

REPO_ROOT = Path(__file__).resolve().parent
BUSY_GPU_FRAC = 0.5  # 显存已用超过一半，就当作训练正在占着显卡


def harness_available() -> bool:
    return importlib.util.find_spec("lm_eval") is not None


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _positive_int(text: str) -> int:
    """argparse 的类型：必须是 ≥ 1 的整数，其余当作用法错误。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"需要 ≥ 1 的整数，收到 {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"需要 ≥ 1 的整数，收到 {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="官方基准评测（lm-evaluation-harness + 代码评测）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run 名字（runs/ 下的目录名）")
    g.add_argument("--reference", choices=["gpt2"], help="评测参照模型并写入参照文件")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--ckpt", action="append", help="只评这个检查点（文件名，可重复）")
    p.add_argument("--device", choices=["cuda", "cpu"], default=_default_device())
    p.add_argument("--limit", type=_positive_int, default=None, help="每个框架任务最多评多少条（默认全量）")
    p.add_argument("--code-limit", type=_positive_int, default=CODE_LIMIT, help="代码评测抽样条数")
    p.add_argument("--code-all", action="store_true", help="代码评测用全量")
    p.add_argument("--skip-code", action="store_true", help="不跑代码评测")
    p.add_argument("--force", action="store_true", help="已评过的也重评；评参照模型时，允许用不够完整的一次评测覆盖参照文件里更完整的那条")
    p.add_argument("--allow-busy-gpu", action="store_true", help="显卡正被占用也照样评")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "eval")
    p.add_argument("--reference-out", type=Path, default=REPO_ROOT / "docs" / "official" / "reference.json")
    return p


def _busy_gpu_message() -> str | None:
    s = gpu_status()
    if s and s["mem_used_gb"] > BUSY_GPU_FRAC * s["mem_total_gb"]:
        return (
            f"显卡显存已经用了 {s['mem_used_gb']:.1f} / {s['mem_total_gb']:.1f} GB，多半是训练正在跑。"
            "Windows 上显存不够会悄悄挤到系统内存，两边都会慢得像爬。三条出路：\n"
            "  1. 等训练停下来再评测；\n"
            "  2. 加 --device cpu，用处理器评（慢，但不影响训练）；\n"
            "  3. 确认显存够用的话，加 --allow-busy-gpu 照样评。"
        )
    return None


def _targets(args, run_dir: Path, code_limit: int | None) -> list[Path] | int:
    """返回要评的检查点；用法错误返回退出码 2。"""
    if not (run_dir / "ckpt").is_dir():
        _err(f"找不到 run 目录里的检查点：{run_dir / 'ckpt'}")
        return 2
    if args.ckpt:
        out = []
        for name in args.ckpt:
            path = run_dir / "ckpt" / name
            if name != Path(name).name or not path.is_file():
                _err(f"找不到检查点：{name}（只接受 {run_dir / 'ckpt'} 下的文件名）")
                return 2
            out.append(path)
        return out
    if args.force:
        return final_checkpoints(run_dir)
    return pending_checkpoints(run_dir, args.limit, code_limit)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(message)s")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    code_limit = 0 if args.skip_code else (None if args.code_all else args.code_limit)

    if not harness_available():
        _err('没有安装官方评测依赖，先运行：uv pip install -e ".[official]"')
        return 2
    if args.device == "cuda" and not args.allow_busy_gpu:
        msg = _busy_gpu_message()
        if msg:
            _err(msg)
            return 2

    if args.reference:
        # 抽样或不带代码评测的一次，不能悄悄盖掉参照文件里更完整的那条；评测和下载开始之前就拦下
        existing = load_reference(args.reference_out).get(args.reference)
        if (
            isinstance(existing, dict)
            and not args.force
            and completeness({"limit": args.limit, "code_limit": code_limit}) < completeness(existing)
        ):
            _err(
                f"参照文件 {args.reference_out} 里已有更完整的 {args.reference} 评测结果，这一次（抽样或不带代码评测）会把它盖掉。"
                "两条出路：加 --force 确认覆盖；或加 --reference-out <别的路径> 写到别处。"
            )
            return 2
        # 代码评测数据先备齐，免得慢的框架任务跑完才发现缺文件
        if code_limit != 0:
            ensure_code_data(args.data_dir)
        entry = evaluate_reference(args.device, args.data_dir, limit=args.limit, code_limit=code_limit)
        save_reference(args.reference_out, args.reference, entry)
        print(json.dumps(entry, ensure_ascii=False))
        return 0

    run_dir = args.runs / args.run
    targets = _targets(args, run_dir, code_limit)
    if isinstance(targets, int):
        return targets
    if not targets:
        _err("没有待评测的检查点（都评过了；加 --force 重评）")
        return 0
    if code_limit != 0:
        ensure_code_data(args.data_dir)
    failed = 0
    for ckpt in targets:
        log.info("评测 %s（设备 %s）", ckpt.name, args.device)
        try:
            rec = evaluate_checkpoint(ckpt, args.device, args.data_dir, limit=args.limit, code_limit=code_limit)
        except (CheckpointError, FileNotFoundError, RuntimeError, ValueError) as e:
            _err(f"检查点 {ckpt.name} 评测失败：{e}")
            failed += 1
            continue
        append_record(run_dir, rec)
        print(json.dumps(asdict(rec), ensure_ascii=False), flush=True)
    if failed:
        _err(f"{failed} 个检查点评测失败，其余已记录")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
