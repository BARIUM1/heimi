"""流式读取 parquet 里的文本列。

文本列名按候选顺序自动识别（默认 text、content），都没有就报错并列出全部列名。
按 batch_rows 行一批地读，内存与文件大小无关。
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from minids.data.sources import TEXT_COLUMNS


def detect_text_column(column_names: Sequence[str], candidates: Sequence[str] = TEXT_COLUMNS) -> str:
    for c in candidates:
        if c in column_names:
            return c
    raise ValueError(f"找不到文本列：候选 {list(candidates)}，文件里的列是 {list(column_names)}")


def count_rows(parquet_path: Path) -> int:
    return pq.ParquetFile(parquet_path).metadata.num_rows


def iter_documents(
    parquet_path: Path,
    text_columns: Sequence[str] = TEXT_COLUMNS,
    batch_rows: int = 1024,
) -> Iterator[tuple[int, str | None]]:
    """产出 (行号, 文本)；null 行的文本为 None，空串原样产出，由调用方决定跳过。"""
    pf = pq.ParquetFile(parquet_path)
    col = detect_text_column(pf.schema_arrow.names, text_columns)
    row = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=[col]):
        for value in batch.column(0).to_pylist():
            yield row, value
            row += 1
