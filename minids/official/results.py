"""官方评测的结果记录与参照文件。不 import torch：看板进程也要用它。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from minids.logging_io import append_jsonl, read_jsonl

OFFICIAL_FILE = "official.jsonl"
# 代码评测的出题方式。"healed"：提示最后那个空白 token 交给模型自己重写（minids/official/healing.py），
# 填空题待填那一行行末的换行算在后缀开头。"plain"：2026-11-21 第一次评测用的旧方式（提示原样喂进去），
# 那时的记录和参照条目里没有 code_method 这一项。两种方式的代码成绩不能放在一起比。
CODE_METHOD = "healed"
LEGACY_CODE_METHOD = "plain"


@dataclass
class OfficialRecord:
    checkpoint: str
    stage: str
    global_step: int
    tokens_total: int
    params_non_embedding: int
    limit: int | None  # None 表示框架任务全量
    code_limit: int | None  # None 全量；0 没跑代码评测；正数抽样条数
    metrics: dict[str, float]
    harness_version: str
    time: str
    code_method: str  # 这条记录的代码评测用的出题方式，见 CODE_METHOD


def code_method_of(rec: dict) -> str | None:
    """记录（或参照条目）的代码评测出题方式；没跑代码评测的返回 None，旧记录没有这一项的算 "plain"。"""
    if rec.get("code_limit") == 0:
        return None
    return rec.get("code_method", LEGACY_CODE_METHOD)


def append_record(run_dir: Path, rec: OfficialRecord) -> None:
    append_jsonl(Path(run_dir) / OFFICIAL_FILE, rec)


def read_records(run_dir: Path) -> list[dict]:
    # read_jsonl 对不存在的文件（含目录不存在）返回 []
    return read_jsonl(Path(run_dir) / OFFICIAL_FILE)


def completeness(rec: dict) -> tuple[bool, bool]:
    """记录（或参照条目）的完整程度，越大越完整：框架任务全量 > 抽样，其次带代码评测 > 不带。"""
    return (rec.get("limit") is None, rec.get("code_limit") != 0)


def latest_per_checkpoint(records: list[dict]) -> list[dict]:
    """每个检查点只留一条：框架任务全量 > 抽样，其次带代码评测 > 不带，再次代码评测用现在的出题方式 > 旧方式，最后取后写入的。
    这样小样本试跑不会盖掉全量结果，旧代码后来补写的记录也不会盖掉现在这种出题方式的结果。
    结果按 (tokens_total, global_step) 升序。"""
    best: dict[str, tuple[tuple, dict]] = {}
    for i, r in enumerate(records):
        key = (*completeness(r), code_method_of(r) in (None, CODE_METHOD), i)
        cur = best.get(r["checkpoint"])
        if cur is None or key > cur[0]:
            best[r["checkpoint"]] = (key, r)
    return sorted((r for _, r in best.values()), key=lambda r: (r["tokens_total"], r["global_step"]))


def final_checkpoints(run_dir: Path) -> list[Path]:
    """run_dir/ckpt 下所有 *_final.pt，按修改时间升序，同一时刻按名字；目录不存在返回 []。"""
    ck = Path(run_dir) / "ckpt"
    if not ck.is_dir():
        return []
    return sorted((p for p in ck.glob("*_final.pt") if p.is_file()), key=lambda p: (p.stat().st_mtime_ns, p.name))


def pending_checkpoints(run_dir: Path, limit: int | None, code_limit: int | None) -> list[Path]:
    """还没有「同一 limit、同一 code_limit、代码评测用现在这种出题方式」记录的最终检查点。

    旧出题方式留下的记录不算评过：带代码评测的检查点会用现在的方式重评一遍。
    """
    done = {(r["checkpoint"], r.get("limit"), r.get("code_limit"), code_method_of(r)) for r in read_records(run_dir)}
    method = code_method_of({"code_limit": code_limit, "code_method": CODE_METHOD})
    return [p for p in final_checkpoints(run_dir) if (p.name, limit, code_limit, method) not in done]


def load_reference(path: Path) -> dict:
    """文件不存在、不是合法 JSON、或顶层不是对象，都当作没有参照。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_reference(path: Path, name: str, entry: dict) -> None:
    """读出已有内容、更新一项，经同目录临时文件 os.replace 写回。"""
    path = Path(path)
    data = load_reference(path)
    data[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
