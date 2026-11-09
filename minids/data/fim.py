"""Fill-in-the-Middle 的 PSM 改写与还原（DeepSeek-Coder 的格式）。

    <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle <|endoftext|>

两个切分点 a ≤ b 在 [0, n] 上均匀随机，prefix = t[:a]，middle = t[a:b]，suffix = t[b:]。
输出比输入长 4 个 token。分词器从不会从文本里产出 1、2、3 这几个 id，
所以还原时可以安全地按标记位置拆分。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FimIds:
    prefix: int = 1
    middle: int = 2
    suffix: int = 3
    eot: int = 0


def apply_fim_psm(tokens: np.ndarray, rng: np.random.Generator, ids: FimIds = FimIds()) -> np.ndarray:
    t = np.asarray(tokens, dtype=np.int64)
    n = len(t)
    a, b = sorted(int(v) for v in rng.integers(0, n + 1, size=2))
    return np.concatenate(
        [
            np.array([ids.prefix], dtype=np.int64),
            t[:a],
            np.array([ids.suffix], dtype=np.int64),
            t[b:],
            np.array([ids.middle], dtype=np.int64),
            t[a:b],
            np.array([ids.eot], dtype=np.int64),
        ]
    )


def undo_fim_psm(tokens: np.ndarray, ids: FimIds = FimIds()) -> np.ndarray:
    t = np.asarray(tokens, dtype=np.int64)
    if len(t) < 4 or t[0] != ids.prefix or t[-1] != ids.eot:
        raise ValueError("不是合法的 PSM 序列：开头应为 fim_prefix，结尾应为 endoftext")
    body = t[1:-1]
    s = np.flatnonzero(body == ids.suffix)
    m = np.flatnonzero(body == ids.middle)
    if len(s) != 1 or len(m) != 1 or s[0] > m[0]:
        raise ValueError("不是合法的 PSM 序列：fim_suffix 与 fim_middle 标记应各出现一次且顺序正确")
    prefix = body[: s[0]]
    suffix = body[s[0] + 1 : m[0]]
    middle = body[m[0] + 1 :]
    return np.concatenate([prefix, middle, suffix])
