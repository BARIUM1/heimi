import json

import pytest
import torch

from minids.data.loader import MixConfig, MixedLoader, ShardSet
from tests.helpers_data import write_shard


def _two_sets(tmp_path, min_tokens=9):
    a = write_shard(tmp_path / "web_0.bin", list(range(10, 60)))  # 值在 10..59，避开特殊 id 0..3
    b = write_shard(tmp_path / "code_0.bin", list(range(100, 170)))  # 值 ≥ 100
    return {"web": ShardSet([a], [50], min_tokens), "code": ShardSet([b], [70], min_tokens)}


def test_loader_requires_min_tokens_to_cover_window(tmp_path):
    with pytest.raises(ValueError):
        MixedLoader(_two_sets(tmp_path, 9), MixConfig(), batch_size=4, seq_len=16, seed=1, vocab_size=200)


def test_batch_shapes_and_shift(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path), MixConfig(), batch_size=4, seq_len=8, seed=1, vocab_size=200)
    x, y = ld.get_batch(0)
    assert x.shape == y.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_deterministic_per_step(tmp_path):
    def mk():
        return MixedLoader(_two_sets(tmp_path), MixConfig(), 4, 8, seed=7, vocab_size=200)

    assert torch.equal(mk().get_batch(3)[0], mk().get_batch(3)[0])
    assert not torch.equal(mk().get_batch(3)[0], mk().get_batch(4)[0])


def test_windows_never_cross_shards(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path), MixConfig(web=0.5, code=0.5), 32, 8, seed=0, vocab_size=200)
    for step in range(20):
        x, _ = ld.get_batch(step)
        assert all(((row < 100).all() or (row >= 100).all()).item() for row in x)


def test_mix_ratio_converges(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path), MixConfig(web=0.85, code=0.15), 16, 8, seed=0, vocab_size=200)
    n_code = 0
    for s in range(200):
        ld.get_batch(s)
        n_code += ld.sources_in_last_batch().count("code")
    assert 0.12 <= n_code / (16 * 200) <= 0.18


def test_fim_applied_only_to_code(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path, 17), MixConfig(web=0.0, code=1.0, fim_rate=1.0), 8, 16, seed=0, vocab_size=200)
    x, y = ld.get_batch(0)
    full = torch.cat([x[:, :1], y], dim=1)  # 还原 seq_len + 1 的序列
    assert (full[:, 0] == 1).all() and (full[:, -1] == 0).all()
    ld2 = MixedLoader(_two_sets(tmp_path, 17), MixConfig(web=1.0, code=0.0, fim_rate=1.0), 8, 16, seed=0, vocab_size=200)
    x2, _ = ld2.get_batch(0)
    assert (x2[:, 0] != 1).all()


def test_fim_rate_converges(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path, 17), MixConfig(web=0.0, code=1.0, fim_rate=0.5), 16, 16, seed=0, vocab_size=200)
    n_fim = sum(int((ld.get_batch(s)[0][:, 0] == 1).sum()) for s in range(100))
    assert 0.42 <= n_fim / (16 * 100) <= 0.58


def test_short_shard_excluded_with_warning(tmp_path):
    short = write_shard(tmp_path / "s.bin", list(range(5)))
    ok = write_shard(tmp_path / "o.bin", list(range(50)))
    with pytest.warns(UserWarning, match="s.bin"):
        ss = ShardSet([short, ok], [5, 50], min_tokens=9)
    assert ss.total_tokens == 50
    with pytest.raises(ValueError):
        ShardSet([short], [5], min_tokens=9)


def test_corrupt_token_value_raises_with_shard_name(tmp_path):
    bad = write_shard(tmp_path / "bad.bin", [1, 2, 40000] + list(range(20)))
    sets = {"web": ShardSet([bad], [23], min_tokens=9), "code": ShardSet([bad], [23], min_tokens=9)}
    ld = MixedLoader(sets, MixConfig(), 8, 8, seed=0, vocab_size=16384)
    with pytest.raises(ValueError, match="bad.bin"):
        for s in range(50):
            ld.get_batch(s)


def test_from_index(tmp_path):
    write_shard(tmp_path / "web" / "train_0000.bin", list(range(30)))
    idx = tmp_path / "index.json"
    idx.write_text(
        json.dumps({"vocab_size": 16384, "dtype": "uint16", "eot_id": 0, "sources": {
            "web": {"train": [{"file": "train_0000.bin", "tokens": 30}], "val": []}}}),
        encoding="utf-8",
    )
    assert ShardSet.from_index(idx, "web", "train", min_tokens=9).total_tokens == 30
