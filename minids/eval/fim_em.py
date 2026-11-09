"""单行 FIM 精确匹配率（spec 10）。

从代码验证集抽文档，每篇随机挖掉一整个非空行做 middle，模型贪心补到行尾，
去首尾空白后与原行完全相同算命中。样例由固定种子决定，每次评测比的是同一批行。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minids.completion import complete_middle
from minids.data.loader import ShardSet
from minids.data.tokenizer import EOT, HeimiTokenizer


@dataclass(frozen=True)
class FimCase:
    prefix: str
    middle: str
    suffix: str  # prefix + middle + suffix == 原文


def split_documents(tokens: np.ndarray, eot: int = 0) -> list[np.ndarray]:
    """按 eot 切成文档，丢弃空段。"""
    t = np.asarray(tokens)
    docs: list[np.ndarray] = []
    start = 0
    for c in np.flatnonzero(t == eot):
        if c > start:
            docs.append(t[start:c])
        start = int(c) + 1
    if start < len(t):
        docs.append(t[start:])
    return docs


def _nonblank_line_spans(doc: str) -> list[tuple[int, int]]:
    """每个非空行的 (起点, 终点) 字符偏移，终点不含换行。"""
    spans = []
    pos = 0
    for line in doc.split("\n"):
        end = pos + len(line)
        if line.strip():
            spans.append((pos, end))
        pos = end + 1
    return spans


def make_fim_cases(docs: Sequence[str], n: int, rng: np.random.Generator, min_lines: int = 3) -> list[FimCase]:
    """只用 ≥ min_lines 个非空行的文档；不放回抽 n 篇（不足全取）；每篇均匀随机挑一个非空行做 middle。"""
    eligible: list[tuple[str, list[tuple[int, int]]]] = []
    for d in docs:
        spans = _nonblank_line_spans(d)
        if len(spans) >= min_lines:
            eligible.append((d, spans))
    if not eligible:
        return []
    chosen = rng.choice(len(eligible), size=min(n, len(eligible)), replace=False)
    cases = []
    for i in chosen:
        d, spans = eligible[int(i)]
        a, b = spans[int(rng.integers(0, len(spans)))]
        cases.append(FimCase(d[:a], d[a:b], d[b:]))
    return cases


def cases_from_index(index_path: Path, tok: HeimiTokenizer, n: int, seed: int, max_tokens: int = 4_000_000) -> list[FimCase]:
    """从代码验证分片的前 max_tokens 个 token 里抽 n 个样例（实测前 400 万 token 约 1800 篇文档）。"""
    tokens = ShardSet.from_index(Path(index_path), "code", "val", 1).read_prefix(max_tokens)
    docs = [tok.decode(d.tolist()) for d in split_documents(tokens, eot=EOT)]
    return make_fim_cases(docs, n, np.random.default_rng(seed))


def fim_exact_match(
    model,
    tok: HeimiTokenizer,
    cases: Sequence[FimCase],
    device: str,
    max_new_tokens: int = 64,
    max_prefix_tokens: int = 512,
    max_suffix_tokens: int = 256,
) -> float:
    if not cases:
        raise ValueError("没有 FIM 样例可评测")
    hits = 0
    for c in cases:
        out = complete_middle(model, tok, c.prefix, c.suffix, max_new_tokens, device, max_prefix_tokens, max_suffix_tokens)
        hits += out.strip() == c.middle.strip()
    return hits / len(cases)
