"""从 uint16 分片随机取窗口、按配比混合两个来源的加载器。

随机性完全由 (seed, step) 决定：同一 step 永远取到同一 batch，恢复训练不会错位。
窗口不跨分片。每个窗口独立以概率 mix.code 选 code 来源；
被选中做 FIM 的 code 窗口取 seq_len - 3 个 token，改写后恰好 seq_len + 1。
组 batch 时检查所有 token < vocab_size，损坏的分片在这里被拦成 ValueError，
不会流到 GPU 上变成 device-side assert。
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from minids.data.fim import FimIds, apply_fim_psm


class ShardSet:
    def __init__(self, files: Sequence[Path], token_counts: Sequence[int], min_tokens: int):
        if len(files) != len(token_counts):
            raise ValueError("files 与 token_counts 长度不一致")
        self._files: list[Path] = []
        self._counts: list[int] = []
        self._maps: list[np.memmap] = []
        for f, n in zip(files, token_counts):
            f = Path(f)
            if n < min_tokens:
                warnings.warn(f"分片 {f.name} 只有 {n} 个 token，少于 {min_tokens}，已排除", UserWarning, stacklevel=2)
                continue
            self._maps.append(np.memmap(f, dtype=np.uint16, mode="r", shape=(int(n),)))
            self._files.append(f)
            self._counts.append(int(n))
        if not self._maps:
            raise ValueError(f"没有任何分片的 token 数 ≥ {min_tokens}")
        self._min_tokens = min_tokens
        self._counts_arr = np.asarray(self._counts, dtype=np.int64)

    @classmethod
    def from_index(cls, index_path: Path, source: str, split: str, min_tokens: int) -> "ShardSet":
        index_path = Path(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index["sources"][source][split]
        files = [index_path.parent / source / e["file"] for e in entries]
        return cls(files, [e["tokens"] for e in entries], min_tokens)

    @property
    def total_tokens(self) -> int:
        return int(self._counts_arr.sum())

    @property
    def min_tokens(self) -> int:
        return self._min_tokens

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    def sample_window(self, rng: np.random.Generator, length: int) -> tuple[np.ndarray, str]:
        """先按可取起点数加权选片，再在片内均匀选起点；返回 (int64 数组, 分片文件名)。"""
        starts = self._counts_arr - length + 1
        if (starts <= 0).all():
            raise ValueError(f"窗口长度 {length} 超过所有分片的 token 数")
        starts = np.clip(starts, 0, None)
        i = int(rng.choice(len(starts), p=starts / starts.sum()))
        start = int(rng.integers(0, starts[i]))
        return np.asarray(self._maps[i][start : start + length], dtype=np.int64), self._files[i].name


@dataclass(frozen=True)
class MixConfig:
    web: float = 0.85
    code: float = 0.15
    fim_rate: float = 0.0  # 只作用于 code 窗口


class MixedLoader:
    def __init__(
        self,
        sets: dict[str, ShardSet],
        mix: MixConfig,
        batch_size: int,
        seq_len: int,
        seed: int,
        vocab_size: int,
        fim_ids: FimIds = FimIds(),
    ):
        for name, s in sets.items():
            if s.min_tokens < seq_len + 1:
                raise ValueError(f"来源 {name} 的 ShardSet.min_tokens={s.min_tokens} 小于 seq_len + 1 = {seq_len + 1}")
        total = mix.web + mix.code
        if total <= 0:
            raise ValueError("web 与 code 的配比之和必须大于 0")
        for name, weight in (("web", mix.web), ("code", mix.code)):
            if weight > 0 and name not in sets:
                raise ValueError(f"配比里 {name} > 0，但没有提供它的 ShardSet")
        self.sets = sets
        self.p_code = mix.code / total
        self.fim_rate = mix.fim_rate
        self.fim_ids = fim_ids
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.vocab_size = vocab_size
        self._last_sources: list[str] = []

    def get_batch(self, step: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        rng = np.random.default_rng([self.seed, step])
        window = self.seq_len + 1
        rows = np.empty((self.batch_size, window), dtype=np.int64)
        sources: list[str] = []
        for i in range(self.batch_size):
            use_code = rng.random() < self.p_code
            name = "code" if use_code else "web"
            if use_code and self.fim_rate > 0 and rng.random() < self.fim_rate:
                win, shard = self.sets["code"].sample_window(rng, window - 4)
                win = apply_fim_psm(win, rng, self.fim_ids)
            else:
                win, shard = self.sets[name].sample_window(rng, window)
            peak = int(win.max())
            if peak >= self.vocab_size:
                raise ValueError(f"分片 {shard} 含越界 token {peak} ≥ vocab_size {self.vocab_size}，文件可能损坏")
            rows[i] = win
            sources.append(name)
        self._last_sources = sources
        x = torch.from_numpy(np.ascontiguousarray(rows[:, :-1]))
        y = torch.from_numpy(np.ascontiguousarray(rows[:, 1:]))
        return x, y

    def sources_in_last_batch(self) -> list[str]:
        return list(self._last_sources)
