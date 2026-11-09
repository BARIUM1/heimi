"""把原始 parquet 流式分词写成 uint16 token 分片，并生成 index.json。

分片格式：小端 uint16 原始数组，无文件头，每片 tokens_per_shard 个 token，
文档之间插入 <|endoftext|>。每个来源第一个文件的前 val_rows 行进验证集。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minids.data.parquet_io import iter_documents
from minids.data.sources import SOURCES, TEXT_COLUMNS, Source, local_files
from minids.data.tokenizer import EOT, HeimiTokenizer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareConfig:
    tokens_per_shard: int = 100_000_000
    batch_rows: int = 1024
    val_target_bytes: int = 80_000_000  # 约 2000 万 token × 4 字节


def estimate_val_rows(
    parquet_path: Path,
    target_bytes: int,
    text_columns: Sequence[str] = TEXT_COLUMNS,
    filter_column: str | None = None,
    filter_values: Sequence[str] = (),
) -> int:
    """顺序读文件，累计非空文本的 UTF-8 字节数首次 ≥ target_bytes 时的行数（含该行）。"""
    used = 0
    rows = 0
    for row, text in iter_documents(parquet_path, text_columns, filter_column=filter_column, filter_values=filter_values):
        rows = row + 1
        if text:
            used += len(text.encode("utf-8"))
        if used >= target_bytes:
            return rows
    return rows


class ShardWriter:
    """把 token 流切成定长分片写盘；一篇文档跨片时切开写入相邻两片。"""

    def __init__(self, out_dir: Path, prefix: str, tokens_per_shard: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.cap = tokens_per_shard
        self._buf = np.empty(tokens_per_shard, dtype=np.uint16)
        self._fill = 0
        self._index = 0
        self.shards: list[dict] = []

    def append(self, tokens: Sequence[int]) -> None:
        arr = np.asarray(tokens, dtype=np.uint16)
        pos = 0
        while pos < len(arr):
            n = min(self.cap - self._fill, len(arr) - pos)
            self._buf[self._fill : self._fill + n] = arr[pos : pos + n]
            self._fill += n
            pos += n
            if self._fill == self.cap:
                self._flush()

    def _flush(self) -> None:
        if self._fill == 0:
            return
        name = f"{self.prefix}_{self._index:04d}.bin"
        self._buf[: self._fill].tofile(self.out_dir / name)
        self.shards.append({"file": name, "tokens": int(self._fill)})
        self._index += 1
        self._fill = 0

    def close(self) -> list[dict]:
        self._flush()
        return self.shards


def prepare_source(
    source_name: str,
    parquet_paths: Sequence[Path],
    tokenizer: HeimiTokenizer,
    out_root: Path,
    val_rows: int,
    cfg: PrepareConfig = PrepareConfig(),
    text_columns: Sequence[str] = TEXT_COLUMNS,
    filter_column: str | None = None,
    filter_values: Sequence[str] = (),
) -> dict:
    out_dir = Path(out_root) / source_name
    writers = {True: ShardWriter(out_dir, "val", cfg.tokens_per_shard), False: ShardWriter(out_dir, "train", cfg.tokens_per_shard)}
    counts = {"docs": 0, "skipped": 0, "tokens_val": 0, "tokens_train": 0}
    t0 = time.perf_counter()
    last_log = t0

    def flush(texts: list[str], is_val: list[bool]) -> None:
        for ids, v in zip(tokenizer.encode_batch(texts), is_val):
            ids.append(EOT)
            writers[v].append(ids)
            counts["tokens_val" if v else "tokens_train"] += len(ids)

    for file_i, path in enumerate(parquet_paths):
        texts: list[str] = []
        flags: list[bool] = []
        for row, text in iter_documents(
            path, text_columns, batch_rows=cfg.batch_rows, filter_column=filter_column, filter_values=filter_values
        ):
            if not text:
                counts["skipped"] += 1
                continue
            counts["docs"] += 1
            texts.append(text)
            flags.append(file_i == 0 and row < val_rows)
            if len(texts) >= cfg.batch_rows:
                flush(texts, flags)
                texts, flags = [], []
                now = time.perf_counter()
                if now - last_log > 30:
                    done = counts["tokens_val"] + counts["tokens_train"]
                    log.info("%s：%s 第 %d 行，累计 %.2fB token，%.0f token/s", source_name, path.name, row, done / 1e9, done / (now - t0))
                    last_log = now
        if texts:
            flush(texts, flags)
        log.info("%s：%s 完成", source_name, path.name)

    val_shards = writers[True].close()
    train_shards = writers[False].close()
    return {
        "docs": counts["docs"],
        "skipped": counts["skipped"],
        "tokens_train": counts["tokens_train"],
        "tokens_val": counts["tokens_val"],
        "val_rows": val_rows,
        "train": train_shards,
        "val": val_shards,
    }


def write_index(out_root: Path, per_source: dict[str, dict], vocab_size: int) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "index.json"
    index = {"vocab_size": vocab_size, "dtype": "uint16", "eot_id": EOT, "sources": per_source}
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="黑米数据预处理")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("val-rows", help="估算每个来源进入验证集的行数")
    v.add_argument("--root", default="data")
    v.add_argument("--out", default="data/tokens/val_rows.json")
    v.add_argument("--target-bytes", type=int, default=PrepareConfig.val_target_bytes)

    r = sub.add_parser("run", help="分词并写分片")
    r.add_argument("--root", default="data")
    r.add_argument("--tokenizer", default="data/tokenizer/heimi-16k.json")
    r.add_argument("--val-rows", default="data/tokens/val_rows.json")
    r.add_argument("--source", choices=["web", "code", "all"], default="all")
    r.add_argument("--tokens-per-shard", type=int, default=PrepareConfig.tokens_per_shard)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(args.root)

    if args.cmd == "val-rows":
        rows = {}
        for name, source in SOURCES.items():
            first = local_files(root, source)[0]
            rows[name] = estimate_val_rows(
                first, args.target_bytes, source.text_columns, source.filter_column, source.filter_values
            )
            log.info("%s：%s 的前 %d 行进验证集（≈ %.0f MB 文本）", name, first.name, rows[name], args.target_bytes / 1e6)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    tok = HeimiTokenizer.load(Path(args.tokenizer))
    val_rows = json.loads(Path(args.val_rows).read_text(encoding="utf-8"))
    out_root = root / "tokens"
    index_path = out_root / "index.json"
    per_source = json.loads(index_path.read_text(encoding="utf-8"))["sources"] if index_path.exists() else {}
    cfg = PrepareConfig(tokens_per_shard=args.tokens_per_shard)
    for name in (["web", "code"] if args.source == "all" else [args.source]):
        t0 = time.perf_counter()
        src: Source = SOURCES[name]
        stats = prepare_source(
            name, local_files(root, src), tok, out_root, val_rows[name], cfg,
            text_columns=src.text_columns, filter_column=src.filter_column, filter_values=src.filter_values,
        )
        stats["seconds"] = round(time.perf_counter() - t0)
        per_source[name] = stats
        log.info(
            "%s 完成：%d 篇（跳过 %d），训练 %.3fB token / %d 片，验证 %.3fB token / %d 片，%d 秒",
            name, stats["docs"], stats["skipped"], stats["tokens_train"] / 1e9, len(stats["train"]),
            stats["tokens_val"] / 1e9, len(stats["val"]), stats["seconds"],
        )
        write_index(out_root, per_source, tok.vocab_size)  # 每个来源完成就落盘一次
    log.info("index 写入 %s", index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
