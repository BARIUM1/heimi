import json

import numpy as np

from minids.data.prepare import PrepareConfig, ShardWriter, estimate_val_rows, prepare_source, write_index
from tests.helpers_data import tiny_tokenizer, write_parquet


def test_shard_writer_splits_across_shards_without_loss(tmp_path):
    w = ShardWriter(tmp_path, "train", tokens_per_shard=10)
    w.append(list(range(7)))
    w.append(list(range(100, 118)))  # 第二篇跨两片
    shards = w.close()
    assert [s["tokens"] for s in shards] == [10, 10, 5]
    got = np.concatenate([np.fromfile(tmp_path / s["file"], dtype=np.uint16) for s in shards])
    assert got.tolist() == list(range(7)) + list(range(100, 118))


def test_estimate_val_rows_counts_bytes_of_nonempty_rows(tmp_path):
    p = write_parquet(tmp_path / "a.parquet", ["ab", None, "", "cdef", "gh"])
    assert estimate_val_rows(p, target_bytes=5) == 4  # ab(2) + cdef(4) 在第 4 行达到 ≥ 5


def test_prepare_source_splits_val_rows_then_train_and_counts_skips(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    texts = ["hello world"] * 20 + [None, ""] + ["code = 1"] * 30
    p = write_parquet(tmp_path / "a.parquet", texts)
    stats = prepare_source("web", [p], tok, tmp_path / "tokens", val_rows=22, cfg=PrepareConfig(tokens_per_shard=64))
    assert stats["skipped"] == 2 and stats["docs"] == 50 and stats["val_rows"] == 22
    assert stats["tokens_val"] == sum(len(tok.encode("hello world")) + 1 for _ in range(20))
    assert stats["tokens_train"] == sum(len(tok.encode("code = 1")) + 1 for _ in range(30))
    val = np.concatenate([np.fromfile(tmp_path / "tokens" / "web" / s["file"], dtype=np.uint16) for s in stats["val"]])
    assert val.max() < tok.vocab_size and (val == 0).sum() == 20  # 每篇文档一个分隔符


def test_index_json_shape(tmp_path):
    idx = write_index(
        tmp_path,
        {"web": {"train": [], "val": [], "docs": 0, "skipped": 0, "tokens_train": 0, "tokens_val": 0, "val_rows": 0}},
        16384,
    )
    d = json.loads(idx.read_text(encoding="utf-8"))
    assert d["vocab_size"] == 16384 and d["dtype"] == "uint16" and d["eot_id"] == 0 and "web" in d["sources"]
