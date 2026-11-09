"""FIM 补全：构造 PSM 提示，贪心生成到行尾。

被三处共用：单行 FIM 精确匹配评测、固定提示采样里的填空提示、complete.py 命令行工具。
提示格式与训练时 fim.py 的改写一致：<|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>，
模型接着生成 middle。分词器把换行编成独立 token，所以"到行尾"靠解码文本里出现 "\\n" 判断。
"""

from __future__ import annotations

import torch

from minids.data.fim import FimIds
from minids.data.tokenizer import HeimiTokenizer

FIM_HOLE = "<FILL>"  # samples.jsonl 里展示 FIM 提示时用它标出空位


def fim_prompt_ids(
    tok: HeimiTokenizer,
    prefix: str,
    suffix: str,
    max_prefix_tokens: int,
    max_suffix_tokens: int,
    ids: FimIds = FimIds(),
) -> list[int]:
    """[prefix 标记] + 前缀的最后 max_prefix_tokens 个 token + [suffix 标记] + 后缀的前 max_suffix_tokens 个 + [middle 标记]。"""
    p = tok.encode(prefix)
    s = tok.encode(suffix)
    p = p[max(0, len(p) - max_prefix_tokens) :]
    s = s[:max_suffix_tokens]
    return [ids.prefix] + p + [ids.suffix] + s + [ids.middle]


@torch.no_grad()
def greedy_line(
    model, tok: HeimiTokenizer, prompt_ids: list[int], max_new_tokens: int, device: str, eot: int = 0, n_special: int = 16
) -> str:
    """每步贪心一个 token；生成 eot 或任何特殊 id（< n_special）即停，特殊 token 的字面串从不属于补全内容；
    解码文本出现换行即截到第一个换行并停。调用方保证提示 + 预算 ≤ seq_len。"""
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        idx = model.generate(idx, 1, temperature=0.0)
        t = int(idx[0, -1].item())
        if t == eot or t < n_special:
            break
        generated.append(t)
        text = tok.decode(generated)
        if "\n" in text:
            return text.split("\n", 1)[0]
    return tok.decode(generated)


def complete_middle(
    model,
    tok: HeimiTokenizer,
    prefix: str,
    suffix: str,
    max_new_tokens: int,
    device: str,
    max_prefix_tokens: int = 512,
    max_suffix_tokens: int = 256,
) -> str:
    """默认预算配 seq_len 1024：512 + 256 + 3 个标记 + 最多 253 个新 token。"""
    ids = fim_prompt_ids(tok, prefix, suffix, max_prefix_tokens, max_suffix_tokens)
    return greedy_line(model, tok, ids, max_new_tokens, device)
