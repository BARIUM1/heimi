"""GPU 吞吐基准。

对阶梯某一级的结构，用随机 token 做完整训练步（前向、反向、优化器），
丢弃前几步预热，输出 tokens/s、峰值显存和每步毫秒，并把结果追加到
docs/benchmarks/throughput.jsonl。用于修正 spec 8.1 的时间预算和选微批次。

用法：
  python scripts/bench_throughput.py --stage 2 --micro-batch 16 [--compile] [--steps 30]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minids.config import ModelConfig  # noqa: E402
from minids.model.transformer import Transformer  # noqa: E402

STAGES = {
    0: ModelConfig(),
    1: ModelConfig(n_layer=16),
    2: ModelConfig(n_layer=16, d_model=768, n_head=12, kv_lora_rank=256, ffn_hidden=2048),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=sorted(STAGES), required=True)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30, help="计时的步数（不含预热）")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--max-mem-frac",
        type=float,
        default=0.9,
        help="峰值显存超过物理显存的这个比例就判定为溢出到系统内存（Windows 不会报 OOM，只会变得极慢）",
    )
    p.add_argument("--out", default="docs/benchmarks/throughput.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("需要 CUDA GPU，当前不可用。", file=sys.stderr)
        return 1

    cfg = STAGES[args.stage]
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    n_params = model.count_params()

    compiled = False
    if args.compile:
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as e:  # noqa: BLE001
            print(f"警告：torch.compile 失败，改用普通模式：{type(e).__name__}: {e}", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": args.stage,
        "params_non_embedding": n_params,
        "micro_batch": args.micro_batch,
        "seq_len": args.seq_len,
        "compiled": compiled,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    def step() -> None:
        batch = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len + 1), device=device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    record["gpu_total_mem_gb"] = round(total_mem_gb, 1)

    def check_spill() -> bool:
        peak = torch.cuda.max_memory_allocated() / 1e9
        if peak > args.max_mem_frac * total_mem_gb:
            record.update({"status": "spill", "peak_mem_gb": round(peak, 2)})
            print(json.dumps(record, ensure_ascii=False))
            _append(args.out, record)
            return True
        return False

    try:
        try:
            step()  # 第一步先探显存，溢出就立刻停，不白等
            torch.cuda.synchronize()
            if check_spill():
                return 2
            for _ in range(args.warmup - 1):
                step()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            if not compiled:
                raise
            # Windows 上 torch.compile 常在首次前向时才暴露工具链缺失，此时回退普通模式
            print(f"警告：编译模式首次运行失败，改用普通模式：{type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            compiled = False
            record["compiled"] = False
            torch.manual_seed(0)
            model = Transformer(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
            for _ in range(args.warmup):
                step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        record.update({"status": "oom"})
        print(json.dumps(record, ensure_ascii=False))
        _append(args.out, record)
        return 2

    tokens = args.steps * args.micro_batch * args.seq_len
    record.update(
        {
            "status": "ok",
            "tokens_per_s": round(tokens / elapsed),
            "ms_per_step": round(1000 * elapsed / args.steps, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
    )
    print(json.dumps(record, ensure_ascii=False))
    _append(args.out, record)
    return 0


def _append(path: str, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
