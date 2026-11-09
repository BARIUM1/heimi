"""训练运行目录下四个 jsonl 的记录格式与读写。

字段名与 spec 9.2 完全一致，训练端（计划三）写、窗口端（计划四）读，定义只在这里。
每行一个 JSON 对象，ensure_ascii=False 保持中文可读，写一行刷一次。
读取时容忍最后一行不完整：窗口可能恰好在训练进程写到一半时读文件；
中间的坏行说明文件真的损坏了，照常抛错。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path

METRICS_FILE = "metrics.jsonl"
EVAL_FILE = "eval.jsonl"
EVENTS_FILE = "events.jsonl"
SAMPLES_FILE = "samples.jsonl"
EVENT_TYPES = ("stage_start", "stage_end", "growth", "checkpoint", "nan_abort", "oom", "compile_fallback", "resume")


def now_iso() -> str:
    """本地时间，秒精度，带时区偏移。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class MetricsRecord:
    stage: str
    step: int  # 本级内
    global_step: int
    tokens_stage: int
    tokens_total: int  # 跨级累计，窗口的横轴
    loss: float
    lr: float
    grad_norm: float
    tok_per_s: float
    mem_gb: float
    gpu_temp_c: float | None
    time: str


@dataclass(frozen=True)
class EvalRecord:
    stage: str
    global_step: int
    tokens_total: int
    val_loss_web: float
    val_loss_code: float
    hellaswag_acc: float | None  # 只在每级结束算
    fim_em: float | None  # 任务阶段每次算，其他级只在级末算
    time: str


@dataclass(frozen=True)
class EventRecord:
    type: str
    stage: str
    global_step: int
    tokens_total: int
    time: str
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"未知事件 type={self.type!r}，可选：{EVENT_TYPES}")


@dataclass(frozen=True)
class SampleRecord:
    stage: str
    global_step: int
    tokens_total: int
    prompt_id: str
    prompt: str
    completion: str
    time: str


def append_jsonl(path: Path, record) -> None:
    d = asdict(record) if is_dataclass(record) else dict(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path, since_global_step: int | None = None) -> list[dict]:
    """文件不存在返回 []；末行不是完整 JSON 就跳过；since 给出时只返回 global_step 更大的记录。"""
    path = Path(path)
    if not path.exists():
        return []
    # 按字节切行再逐行解码：末行可能恰好断在一个多字节字符中间，整体解码会直接 UnicodeDecodeError
    lines = path.read_bytes().split(b"\n")
    while lines and not lines[-1].strip():
        lines.pop()
    out: list[dict] = []
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if i == len(lines) - 1:
                break  # 写了一半的末行
            raise
        if since_global_step is None or rec.get("global_step", 0) > since_global_step:
            out.append(rec)
    return out


class RunLogger:
    """一个 run 目录的四个 jsonl 写入口。"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        if name not in (METRICS_FILE, EVAL_FILE, EVENTS_FILE, SAMPLES_FILE):
            raise ValueError(f"未知日志文件 {name}")
        return self.run_dir / name

    def metrics(self, rec: MetricsRecord) -> None:
        append_jsonl(self.path(METRICS_FILE), rec)

    def eval(self, rec: EvalRecord) -> None:
        append_jsonl(self.path(EVAL_FILE), rec)

    def event(self, type: str, stage: str, global_step: int, tokens_total: int, **data) -> None:
        append_jsonl(self.path(EVENTS_FILE), EventRecord(type, stage, global_step, tokens_total, now_iso(), dict(data)))

    def sample(self, rec: SampleRecord) -> None:
        append_jsonl(self.path(SAMPLES_FILE), rec)
