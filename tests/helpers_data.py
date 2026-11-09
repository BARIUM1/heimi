"""数据管线测试共用的小工具：写迷你 parquet、训迷你分词器、写迷你分片。"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from minids.data.tokenizer import HeimiTokenizer, train_tokenizer

_CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "def add(a, b):\n    return a + b\n",
    "for i in range(10):\n    print(i)\n",
    "Hello, world! 中文 ünïcode 🙂",
    "SELECT * FROM users WHERE id = 1;",
    "const x = () => { return 42; };",
] * 20


def write_parquet(path: Path, texts: list, column: str = "text", extra: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {column: pa.array(texts, type=pa.string())}
    for name, values in (extra or {}).items():
        cols[name] = pa.array(values)
    pq.write_table(pa.table(cols), path)
    return path


def tiny_tokenizer(tmp_path: Path, vocab_size: int = 300) -> HeimiTokenizer:
    return train_tokenizer(_CORPUS, tmp_path / f"tok{vocab_size}.json", vocab_size=vocab_size)


def write_shard(path: Path, tokens: Sequence[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype=np.uint16).tofile(path)
    return path
