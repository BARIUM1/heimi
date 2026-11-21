"""官方代码评测的数据：RepoBench-C（下一行补全）与 HumanEval 单行填空，下载与读取。

只下载、只读取，不执行数据里的任何代码。下载做法与 minids/eval/hellaswag.py 一致。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPOBENCH_REPO = "tianyang/repobench_python_v1.1"
REPOBENCH_FILE = "data/in_file-00000-of-00001-9721c68f6224a587.parquet"
REPOBENCH_LOCAL = "repobench_python_in_file.parquet"

INFILL_REPO = "loubnabnl/humaneval_infilling"
INFILL_FILE = "data/HumanEval-SingleLineInfilling.jsonl"
INFILL_LOCAL = "humaneval_single_line.jsonl"

_REPOBENCH_FIELDS = ["repo_name", "file_path", "import_statement", "cropped_code", "next_line", "level"]
_INFILL_FIELDS = ["task_id", "prompt", "suffix", "canonical_solution"]


def download_dataset_file(repo: str, filename: str, dest: Path) -> Path:
    """dest 已存在直接返回；否则下载到临时目录再移到 dest（失败时半个文件和临时目录一起消失）。"""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        got = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=tmp)
        shutil.move(got, dest)
    return dest


def ensure_code_data(data_dir: Path) -> None:
    """两份文件缺哪份下哪份。"""
    data_dir = Path(data_dir)
    if not (data_dir / REPOBENCH_LOCAL).exists():
        download_dataset_file(REPOBENCH_REPO, REPOBENCH_FILE, data_dir / REPOBENCH_LOCAL)
    if not (data_dir / INFILL_LOCAL).exists():
        download_dataset_file(INFILL_REPO, INFILL_FILE, data_dir / INFILL_LOCAL)


def load_repobench(path: Path) -> list[dict]:
    table = pq.read_table(path, columns=_REPOBENCH_FIELDS)
    cols = [table.column(c).to_pylist() for c in _REPOBENCH_FIELDS]
    return [dict(zip(_REPOBENCH_FIELDS, vals)) for vals in zip(*cols)]


def load_infill(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({k: row[k] for k in _INFILL_FIELDS})
    return rows
