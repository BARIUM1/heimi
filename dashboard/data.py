"""只读地把 runs/<run>/ 下的日志与检查点整理成窗口要的数据。

窗口从不写 runs/。这里的函数不经 HTTP 就能测：列 run、列检查点、成长总览序列、训练状态与预计剩余。
run 名与检查点名来自 URL 和请求体，一律先过 safe_run_dir / safe_checkpoint：
名字不合规、带 ..、解析后跑出目录的，都当作不存在。

训练状态分三档：最后一条记录在 RUNNING_WITHIN_S 秒内是"训练中"；超过但在 QUIET_WITHIN_S 内是
"评测中或暂停"（级末评测最长约 20 分钟，期间不写指标）；再久就是"已停止"。

训练进程只往日志末尾追加，从不回头改，所以日志里会有三种"重复"，都在这里收拾：
- 中断后从更早的检查点恢复，同一段步数的指标写了两遍 → supersede_rewinds 只留后写的；
- 同一级进了两次（头 500 步内被打断，没有检查点），增长事件与 stage_start 各有两条 → 按级去重；
- 同一步有两条评测（级末评测加事后的 evaluate.py）→ 合并，新值覆盖旧值，空值不抹掉已有的。
某个 run 的日志中间坏了一行（断电）只影响它自己：LogUnreadable 被就地接住，别的 run 照常。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from statistics import median

from minids.gpu import gpu_status
from minids.logging_io import EVAL_FILE, EVENTS_FILE, METRICS_FILE, SAMPLES_FILE, read_jsonl
from minids.official.results import OFFICIAL_FILE, latest_per_checkpoint, load_reference

RUNNING_WITHIN_S = 180.0
QUIET_WITHIN_S = 1800.0
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CKPT_RE = re.compile(r"^(?P<stage>[A-Za-z0-9][A-Za-z0-9_.-]*?)_(?:step(?P<step>\d+)|final)\.pt$")
LOG_FILES = (METRICS_FILE, EVAL_FILE, EVENTS_FILE, SAMPLES_FILE)
_EVAL_KEYS = ("stage", "global_step", "tokens_total", "val_loss_web", "val_loss_code", "hellaswag_acc", "fim_em")
_GROWTH_WORD = {"depth": "加深", "width": "加宽"}


class RunNotFound(LookupError):
    """run 或检查点不存在，或名字不合规。接口层统一映射成 404。"""


class LogUnreadable(RuntimeError):
    """某个 run 的日志文件读不出来（中间有损坏的行）。只影响这一个 run。"""


def read_log(run_dir: Path, name: str) -> list[dict]:
    """读一个 run 的一份日志；坏文件抛 LogUnreadable（消息里带 run 名与文件名）。"""
    run_dir = Path(run_dir)
    try:
        return read_jsonl(run_dir / name)
    except (ValueError, OSError) as e:  # JSONDecodeError 与 UnicodeDecodeError 都是 ValueError
        raise LogUnreadable(f"{run_dir.name}/{name} 读不出来，中间有损坏的行（{type(e).__name__}）") from e


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
        t = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo is not None else t.astimezone()  # 没带时区的按本地时间算


def last_update(run_dir: Path) -> datetime | None:
    """四个日志文件各自最后一条记录的 time 里最新的一个。"""
    times = []
    for name in LOG_FILES:
        records = read_log(run_dir, name)
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


def _summarize(run_dir: Path) -> dict:
    metrics = read_log(run_dir, METRICS_FILE)
    events = read_log(run_dir, EVENTS_FILE)
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


def run_summary(run_dir: Path) -> dict:
    """一个 run 的概况。日志读不出来时仍返回一条带 error 的概况，页面据此说明原因而不是整页空白。"""
    run_dir = Path(run_dir)
    try:
        return _summarize(run_dir)
    except LogUnreadable as e:
        return {
            "name": run_dir.name, "stages": [], "stage": None, "global_step": 0, "tokens_total": 0,
            "last_update": None, "checkpoints": list_checkpoints(run_dir), "error": str(e),
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


def supersede_rewinds(records: list[dict]) -> list[dict]:
    """去掉被"回退重写"作废的记录：后写的一条 global_step 不比前面大，说明训练从更早的检查点接着跑了，
    前面那些步数不小于它的记录属于被放弃的那一段。结果按 global_step 严格递增。"""
    out: list[dict] = []
    for r in records:
        gs = r.get("global_step", 0)
        while out and out[-1].get("global_step", 0) >= gs:
            out.pop()
        out.append(r)
    return out


def metrics_since(run_dir: Path, since: int | None = None) -> list[dict]:
    """训练指标，去掉回退作废的段；since 给出时只要 global_step 更大的。"""
    rows = supersede_rewinds(read_log(run_dir, METRICS_FILE))
    return rows if since is None else [r for r in rows if r.get("global_step", 0) > since]


def merged_evals(evals: list[dict]) -> list[dict]:
    """同一级同一步的评测合并成一条（后写的非空值覆盖先写的，空值不抹掉已有的），按进度排序。"""
    merged: dict[tuple, dict] = {}
    for e in evals:
        key = (e["stage"], e["global_step"])
        if key in merged:
            merged[key] = {**merged[key], **{k: v for k, v in e.items() if v is not None}}
        else:
            merged[key] = dict(e)
    return sorted(merged.values(), key=lambda e: (e["tokens_total"], e["global_step"]))


def overview(run_dir: Path) -> dict:
    """成长总览的三组序列：合并后的评测点、带标注的增长事件、参数量阶梯。"""
    run_dir = Path(run_dir)
    evals = read_log(run_dir, EVAL_FILE)
    events = read_log(run_dir, EVENTS_FILE)
    metrics = read_log(run_dir, METRICS_FILE)

    eval_out = [{k: e.get(k) for k in _EVAL_KEYS} for e in merged_evals(evals)]

    # 同一级进了两次时增长事件与 stage_start 各有两条：按级留最后一条
    growth_by_stage: dict[str, dict] = {}
    start_by_stage: dict[str, dict] = {}
    for e in events:
        if e["type"] == "growth":
            d = e["data"]
            growth_by_stage[e["stage"]] = {
                "tokens_total": e["tokens_total"],
                "kind": d["kind"],
                "params_before": d["params_before"],
                "params_after": d["params_after"],
                "label": f"{format_params(d['params_before'])} → {format_params(d['params_after'])} {_GROWTH_WORD.get(d['kind'], d['kind'])}",
            }
        elif e["type"] == "stage_start" and "params_non_embedding" in e["data"]:
            start_by_stage[e["stage"]] = {"tokens_total": e["tokens_total"], "params": e["data"]["params_non_embedding"]}
    growth = list(growth_by_stage.values())
    params = list(start_by_stage.values())
    if params:  # 把阶梯延伸到当前进度，最后一级才不是一个孤点
        current = max((r["tokens_total"] for r in (*metrics, *evals, *events)), default=0)
        if current > params[-1]["tokens_total"]:
            params.append({"tokens_total": current, "params": params[-1]["params"]})
    return {"eval": eval_out, "growth": growth, "params": params}


_REFERENCE_KEYS = ("label", "params_non_embedding", "limit", "code_limit", "metrics")


def official(run_dir: Path, reference_path: Path | None) -> dict:
    """官方评测表：每个检查点一行（latest_per_checkpoint 挑出的那条）加参照行。

    comparable：没有参照时为真；否则当且仅当这一行的 limit 与 code_limit 都等于第一条参照的，
    即样本设置相同，才能和参照并排比。official.jsonl 不在 LOG_FILES 里：官方评测不是训练活动，
    不能让它推迟"训练最后一次更新"的时间。窗口只读，这里不创建任何文件。
    """
    reference = []
    if reference_path is not None:
        for name, entry in load_reference(reference_path).items():
            if isinstance(entry, dict) and isinstance(entry.get("metrics"), dict):
                reference.append({"name": name, **{k: entry.get(k) for k in _REFERENCE_KEYS}})
    first = reference[0] if reference else None
    rows = []
    for r in latest_per_checkpoint(read_log(run_dir, OFFICIAL_FILE)):
        same = first is None or (r.get("limit") == first["limit"] and r.get("code_limit") == first["code_limit"])
        rows.append({**r, "comparable": same})
    return {"rows": rows, "reference": reference}


def training_state(last: datetime | None, now: datetime) -> str:
    if last is None:
        return "stopped"
    age = (now - last).total_seconds()
    if age <= RUNNING_WITHIN_S:
        return "running"
    return "quiet" if age <= QUIET_WITHIN_S else "stopped"


def _stage_steps(events: list[dict], stage: str) -> int | None:
    steps = None
    for e in events:
        if e.get("type") == "stage_start" and e.get("stage") == stage:
            steps = e.get("data", {}).get("steps")
    return steps


def eta_seconds(metrics: list[dict], events: list[dict]) -> float | None:
    """本级剩余 token ÷ 该级最近 10 条指标（即最近 100 步）吞吐的中位数。评测后那一条偏低，中位数不受影响。"""
    if not metrics:
        return None
    last = metrics[-1]
    stage, step = last["stage"], last["step"]
    steps = _stage_steps(events, stage)
    if not steps or not step:
        return None
    if step >= steps:
        return 0.0
    recent = [m["tok_per_s"] for m in metrics if m["stage"] == stage and m["tok_per_s"] > 0][-10:]
    if not recent:
        return None
    tokens_per_step = last["tokens_stage"] / step
    return (steps - step) * tokens_per_step / median(recent)


def _newest_run(runs_dir: Path) -> tuple[datetime, Path] | None:
    """最近写过日志的那个 run：它就是"现在在训练的"候选。读不出来的 run 跳过。"""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    best: tuple[datetime, Path] | None = None
    for d in runs_dir.iterdir():
        if d.is_dir() and RUN_NAME_RE.match(d.name):
            try:
                t = last_update(d)
            except LogUnreadable:
                continue
            if t is not None and (best is None or t.timestamp() > best[0].timestamp()):
                best = (t, d)
    return best


def status(runs_dir: Path, now: datetime | None = None, run: str | None = None) -> dict:
    """某个 run 的当前状态：给了 run 就报它（页面正在看的那个），否则报最近更新的。

    无论报哪个，active_run / active_state 都指最近更新的那个，页面据此提示"现在在训练的是谁"。
    没有任何 run 时各数值为 None；所选 run 的日志读不出来时带一个 error 说明。
    """
    now = now or datetime.now().astimezone()
    out: dict = {
        "run": None, "state": "stopped", "running": False,
        "stage": None, "step": None, "steps": None, "global_step": None, "tokens_total": None,
        "loss": None, "lr": None, "tok_per_s": None, "mem_gb": None, "gpu_temp_c": None,
        "eta_seconds": None, "last_update": None, "seconds_since_update": None, "last_eval": None,
        "active_run": None, "active_state": "stopped",
        "gpu": gpu_status(),
    }
    newest = _newest_run(runs_dir)
    if newest is not None:
        out["active_run"] = newest[1].name
        out["active_state"] = training_state(newest[0], now)
    if run is not None:
        run_dir = safe_run_dir(Path(runs_dir), run)
    elif newest is not None:
        run_dir = newest[1]
    else:
        return out
    out["run"] = run_dir.name

    try:
        updated = last_update(run_dir)
        metrics = read_log(run_dir, METRICS_FILE)
        events = read_log(run_dir, EVENTS_FILE)
        evals = merged_evals(read_log(run_dir, EVAL_FILE))
    except LogUnreadable as e:
        out["error"] = str(e)
        return out

    latest = _latest_progress(metrics, events)
    state = training_state(updated, now)
    out.update({
        "state": state, "running": state == "running",
        "last_update": updated.isoformat(timespec="seconds") if updated else None,
        "seconds_since_update": (now - updated).total_seconds() if updated else None,
        "last_eval": evals[-1] if evals else None,  # 按进度最靠后的那次，不是最后写进文件的那条
        "eta_seconds": eta_seconds(metrics, events),
    })
    if latest is not None:
        stage = latest["stage"]
        out.update({"stage": stage, "steps": _stage_steps(events, stage),
                    "global_step": latest["global_step"], "tokens_total": latest["tokens_total"], "step": 0})
    if metrics:
        m = metrics[-1]
        if latest is None or m["stage"] == latest["stage"]:
            out["step"] = m["step"]
        for k in ("loss", "lr", "tok_per_s", "mem_gb", "gpu_temp_c"):
            out[k] = m.get(k)
    return out
