"""流式读取 parquet 里的文本列。

文本列名按候选顺序自动识别（默认 text、content），都没有就报错并列出全部列名。
可选地按另一列的取值过滤行（例如只要 language 为 Python 的行），被过滤掉的行不产出，
但行号保持原文件的行号，这样验证集按"前 N 行"切分时两边口径一致。
按 batch_rows 行一批地读，内存与文件大小无关。
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from minids.data.sources import TEXT_COLUMNS, Source


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
    filter_column: str | None = None,
    filter_values: Sequence[str] = (),
) -> Iterator[tuple[int, str | None]]:
    """产出 (行号, 文本)；null 行的文本为 None，空串原样产出，由调用方决定跳过。"""
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    col = detect_text_column(names, text_columns)
    columns = [col]
    if filter_column is not None:
        if filter_column not in names:
            raise ValueError(f"找不到过滤列 {filter_column}，文件里的列是 {list(names)}")
        columns.append(filter_column)
    keep = set(filter_values)

    row = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        texts = batch.column(0).to_pylist()
        if filter_column is None:
            for value in texts:
                yield row, value
                row += 1
        else:
            for value, tag in zip(texts, batch.column(1).to_pylist()):
                if tag in keep:
                    yield row, value
                row += 1


def iter_source_documents(parquet_path: Path, source: Source, batch_rows: int = 1024) -> Iterator[tuple[int, str | None]]:
    """按数据源注册的文本列与过滤规则读取。"""
    return iter_documents(
        parquet_path,
        text_columns=source.text_columns,
        batch_rows=batch_rows,
        filter_column=source.filter_column,
        filter_values=source.filter_values,
    )
