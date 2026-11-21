"""官方评测测试共用：记录提示长度的脚本模型、迷你代码评测数据、造记录的小函数。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from minids.official.codedata import INFILL_LOCAL, REPOBENCH_LOCAL
from tests.helpers_train import ScriptedModel


class SpyScripted(ScriptedModel):
    """记下每次 generate 收到的序列长度，和第一次收到的整段提示。"""

    def __init__(self, script, seq_len=64):
        super().__init__(script, seq_len)
        self.lengths: list[int] = []
        self.first: list[int] | None = None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.lengths.append(idx.shape[1])
        if self.first is None:
            self.first = idx[0].tolist()
        return super().generate(idx, max_new_tokens, temperature, top_k)


def write_code_data(d: Path) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    n = 3
    pq.write_table(
        pa.table({
            "repo_name": ["o/r"] * n, "file_path": [f"m{i}.py" for i in range(n)], "import_statement": ["import os"] * n,
            "cropped_code": ["def f():\n    x = 1\n"] * n, "next_line": ["    return x"] * n, "level": ["2k"] * n,
            "all_code": ["unused"] * n,
        }),
        d / REPOBENCH_LOCAL,
    )
    rows = [
        {"task_id": f"t{i}", "entry_point": "f", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n", "test": "unused"}
        for i in range(n)
    ]
    (d / INFILL_LOCAL).write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return d
