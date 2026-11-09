"""只读地把 runs/<run>/ 下的日志与检查点整理成窗口要的数据。

窗口从不写 runs/。这里的函数不经 HTTP 就能测：列 run、列检查点、成长总览序列。
run 名与检查点名来自 URL 和请求体，一律先过 safe_run_dir / safe_checkpoint：
名字不合规、带 ..、解析后跑出目录的，都当作不存在。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from minids.logging_io import EVAL_FILE, EVENTS_FILE, METRICS_FILE, SAMPLES_FILE, read_jsonl

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CKPT_RE = re.compile(r"^(?P<stage>[A-Za-z0-9][A-Za-z0-9_.-]*?)_(?:step(?P<step>\d+)|final)\.pt$")
LOG_FILES = (METRICS_FILE, EVAL_FILE, EVENTS_FILE, SAMPLES_FILE)
_EVAL_KEYS = ("stage", "global_step", "tokens_total", "val_loss_web", "val_loss_code", "hellaswag_acc", "fim_em")
_GROWTH_WORD = {"depth": "加深", "width": "加宽"}


class RunNotFound(LookupError):
    """run 或检查点不存在，或名字不合规。接口层统一映射成 404。"""


def safe_run_dir(runs_dir: Path, name: str) -> Path:
    if not isinstance(name, str) or not RUN_NAME_RE.match(name) or ".." in name:
        raise RunNotFound(f"没有名为 {name!r} 的 run")
    base = Path(runs_dir).resolve()
    target = (base / name).resolve()
    if target.parent != base or not target.is_dir():
        raise RunNotFound(f"没有名为 {name!r} 的 run")
    return Path(runs_dir) / name


def list_checkpoints(run_dir: Path) -> list[dict]:
    """ckpt/ 下的 <stage>_final.pt 与 <stage>_step<N>.pt；final 在前，各自按修改时间新到旧。"""
    d = Path(run_dir) / "ckpt"
    if not d.is_dir():
        return []
    found = []
    for p in d.iterdir():
        m = _CKPT_RE.match(p.name)
        if not m or not p.is_file():
            continue  # .tmp、.corrupt 与其他文件
        st = p.stat()
        step = m.group("step")
        found.append((
            step is not None, -st.st_mtime_ns, p.name,
            {
                "name": p.name,
                "stage": m.group("stage"),
                "kind": "step" if step is not None else "final",
                "step": int(step) if step is not None else None,
                "size_mb": round(st.st_size / 1e6, 1),
                "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
            },
        ))
    found.sort(key=lambda t: t[:3])
    return [t[3] for t in found]


def safe_checkpoint(run_dir: Path, name: str) -> Path:
    if name not in {c["name"] for c in list_checkpoints(run_dir)}:
        raise RunNotFound(f"没有名为 {name!r} 的检查点")
    return Path(run_dir) / "ckpt" / name


def format_params(n: int) -> str:
    """12_934_528 → "13M"，75_136 → "75K"。"""
    return f"{round(n / 1e6)}M" if n >= 1_000_000 else f"{round(n / 1e3)}K"


def _parse_time(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def last_update(run_dir: Path) -> datetime | None:
    """四个日志文件各自最后一条记录的 time 里最新的一个。"""
    times = []
    for name in LOG_FILES:
        records = read_jsonl(Path(run_dir) / name)
        if records:
            t = _parse_time(records[-1].get("time"))
            if t is not None:
                times.append(t)
    return max(times) if times else None


def _latest_progress(metrics: list[dict], events: list[dict]) -> dict | None:
    """最新进度所在的记录：比 global_step，相同时取事件（新一级的 stage_start 与上一级最后一条指标同步数）。"""
    m = metrics[-1] if metrics else None
    e = events[-1] if events else None
    if m is None or e is None:
        return m or e
    return e if e["global_step"] >= m["global_step"] else m


def run_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    metrics = read_jsonl(run_dir / METRICS_FILE)
    events = read_jsonl(run_dir / EVENTS_FILE)
    ended = {e["stage"] for e in events if e["type"] == "stage_end"}
    stages: dict[str, dict] = {}
    for e in events:
        if e["type"] == "stage_start":  # 同一级重来过就以最后一次为准
            stages[e["stage"]] = {
                "name": e["stage"],
                "params": e["data"].get("params_non_embedding"),
                "steps": e["data"].get("steps"),
                "growth": e["data"].get("growth"),
                "done": e["stage"] in ended,
            }
    latest = _latest_progress(metrics, events)
    updated = last_update(run_dir)
    return {
        "name": run_dir.name,
        "stages": list(stages.values()),
        "stage": latest["stage"] if latest else None,
        "global_step": latest["global_step"] if latest else 0,
        "tokens_total": latest["tokens_total"] if latest else 0,
        "last_update": updated.isoformat(timespec="seconds") if updated else None,
        "checkpoints": list_checkpoints(run_dir),
    }


def list_runs(runs_dir: Path) -> list[dict]:
    """名字合规且至少有一个日志文件的子目录，按最近更新排在前。"""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out = [
        run_summary(d)
        for d in sorted(runs_dir.iterdir())
        if d.is_dir() and RUN_NAME_RE.match(d.name) and any((d / f).exists() for f in LOG_FILES)
    ]

    def key(r: dict) -> float:
        t = _parse_time(r["last_update"])
        return t.timestamp() if t else float("-inf")

    out.sort(key=key, reverse=True)
    return out


def overview(run_dir: Path) -> dict:
    """成长总览的三组序列：去重后的评测点、带标注的增长事件、参数量阶梯。"""
    run_dir = Path(run_dir)
    evals = read_jsonl(run_dir / EVAL_FILE)
    events = read_jsonl(run_dir / EVENTS_FILE)
    metrics = read_jsonl(run_dir / METRICS_FILE)

    # 级末评测与 evaluate.py 重算会在同一步留下两条记录：留最后一条
    latest_eval: dict[tuple, dict] = {}
    for e in evals:
        latest_eval[(e["stage"], e["global_step"])] = e
    eval_out = sorted(
        ({k: e.get(k) for k in _EVAL_KEYS} for e in latest_eval.values()),
        key=lambda e: (e["tokens_total"], e["global_step"]),
    )

    growth = []
    for e in events:
        if e["type"] != "growth":
            continue
        d = e["data"]
        growth.append({
            "tokens_total": e["tokens_total"],
            "kind": d["kind"],
            "params_before": d["params_before"],
            "params_after": d["params_after"],
            "label": f"{format_params(d['params_before'])} → {format_params(d['params_after'])} {_GROWTH_WORD.get(d['kind'], d['kind'])}",
        })

    params = [
        {"tokens_total": e["tokens_total"], "params": e["data"]["params_non_embedding"]}
        for e in events
        if e["type"] == "stage_start" and "params_non_embedding" in e["data"]
    ]
    if params:  # 把阶梯延伸到当前进度，最后一级才不是一个孤点
        current = max((r["tokens_total"] for r in (*metrics, *evals, *events)), default=0)
        if current > params[-1]["tokens_total"]:
            params.append({"tokens_total": current, "params": params[-1]["params"]})
    return {"eval": eval_out, "growth": growth, "params": params}
