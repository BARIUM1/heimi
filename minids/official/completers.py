"""黑米的两个补全函数：下一行补全、单行填空，交给 codebench 的计分器打分。

计分器按原文打分，所以这里保证返回的文本里没有任何特殊 token 的字面串：
第一个 token 就是特殊 token（早期检查点、没训过 FIM 的检查点常见）时返回空串，照常记为答错。
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from minids.completion import complete_middle, greedy_line
from minids.data.tokenizer import HeimiTokenizer
from minids.official.codebench import first_code_line

CODE_MAX_NEW_TOKENS = 64


@torch.no_grad()
def greedy_text(
    model, tok: HeimiTokenizer, prompt_ids: list[int], max_new_tokens: int, device: str, done: Callable[[str], bool], n_special: int = 16
) -> str:
    """贪心生成，直到 done(已生成文本) 为真、遇到特殊 id（< n_special，不计入结果）或用完预算；返回解码文本。

    不直接用 minids.completion.greedy_line：它遇到第一个换行就停，而下一行补全要跳过开头的注释和空行、
    写完第一行代码才停，停止条件不同。改 greedy_line 会动到自有评测的路径，本计划不碰它，所以另写一份。
    """
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    generated: list[int] = []
    text = ""
    for _ in range(max_new_tokens):
        idx = model.generate(idx, 1, temperature=0.0)
        t = int(idx[0, -1].item())
        if t < n_special:
            break
        generated.append(t)
        text = tok.decode(generated)
        if done(text):
            break
    return text


def _budget(model, max_new_tokens: int) -> int:
    # 新 token 预算最多占上下文的一半，免得提示被截成空的
    return max(1, min(max_new_tokens, model.cfg.seq_len // 2))


def heimi_next_line(model, tok: HeimiTokenizer, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str], str]:
    budget = _budget(model, max_new_tokens)
    seq_len = model.cfg.seq_len

    def complete(prompt: str) -> str:
        ids = tok.encode(prompt)[-(seq_len - budget) :]
        return greedy_text(model, tok, ids, budget, device, done=lambda t: first_code_line(t)[1])

    return complete


def heimi_infill(model, tok: HeimiTokenizer, device: str, mode: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str, str], str]:
    if mode not in ("fim", "prefix"):
        raise ValueError(f"mode 只能是 'fim' 或 'prefix'，收到 {mode!r}")
    budget = _budget(model, max_new_tokens)
    seq_len = model.cfg.seq_len

    if mode == "fim":
        room = seq_len - budget - 3
        max_prefix = room * 2 // 3
        max_suffix = room - max_prefix

        def complete(prefix: str, suffix: str) -> str:
            return complete_middle(model, tok, prefix, suffix, budget, device, max_prefix, max_suffix)

    else:

        def complete(prefix: str, suffix: str) -> str:
            ids = tok.encode(prefix)[-(seq_len - budget) :]
            return greedy_line(model, tok, ids, budget, device)

    return complete
