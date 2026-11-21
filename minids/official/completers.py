"""黑米的两个补全函数：下一行补全、单行填空，交给 codebench 的计分器打分。

提示结尾的空白不放进提示，由模型自己写出来（规则和原因见 healing.py）；返回的是接在原提示后面的文本。

计分器按原文打分，所以这里保证返回的文本里没有任何特殊 token 的字面串：
写完结尾空白后第一个 token 就是特殊 token（早期检查点、没训过 FIM 的检查点常见）时返回空串，照常记为答错。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from minids.completion import fim_prompt_ids
from minids.data.tokenizer import EOT, HeimiTokenizer
from minids.official.codebench import first_code_line
from minids.official.healing import split_tail, whitespace_tokens, write_tail

CODE_MAX_NEW_TOKENS = 64


@torch.no_grad()
def greedy_text(
    model, tok: HeimiTokenizer, prompt_ids: list[int], max_new_tokens: int, device: str, done: Callable[[str], bool],
    n_special: int = 16, start: Sequence[int] = (),
) -> str:
    """贪心生成，直到 done(已生成文本) 为真、遇到特殊 id（< n_special，不计入结果）或用完预算；返回解码文本。

    start 是已经定下来的开头几个 token（写提示结尾空白的那几个）：接在提示后面，算进预算，也算进返回的文本。

    不用 minids.completion.greedy_line：它遇到第一个换行就停，而这里的停止条件由调用方给，
    开头还可能带着 start。改 greedy_line 会动到自有评测的路径，所以另写一份。
    """
    idx = torch.tensor([[*prompt_ids, *start]], dtype=torch.long, device=device)
    generated = list(start)
    text = tok.decode(generated) if generated else ""
    while len(generated) < max_new_tokens and not (generated and done(text)):
        idx = model.generate(idx, 1, temperature=0.0)
        t = int(idx[0, -1].item())
        if t < n_special:
            break
        generated.append(t)
        text = tok.decode(generated)
    return text


def _budget(model, max_new_tokens: int) -> int:
    # 新 token 预算最多占上下文的一半，免得提示被截成空的
    return max(1, min(max_new_tokens, model.cfg.seq_len // 2))


def _continuer(model, tok: HeimiTokenizer, device: str, budget: int) -> Callable[[list[int], str, Callable[[str], bool]], str]:
    """返回 continue_(提示 token, tail, done)：让模型先把 tail 写出来再往下写，返回 tail 之后的文本。

    写 tail 的 token 也算在 budget 里；budget 内写不完 tail 时返回空串。done 收到的是 tail 之后的文本。
    """
    ws_tokens = whitespace_tokens(tok.decode([i]) for i in range(tok.vocab_size))

    @torch.no_grad()
    def continue_(prompt_ids: list[int], tail: str, done: Callable[[str], bool]) -> str:
        def pick(written: list[int], allowed: list[int]) -> int:
            logits, _ = model(torch.tensor([[*prompt_ids, *written]], dtype=torch.long, device=device))
            return allowed[int(logits[0, -1].float()[allowed].argmax())]

        start = write_tail(tail, ws_tokens, pick, budget)
        if start is None:
            return ""
        text = greedy_text(model, tok, prompt_ids, budget, device, done=lambda t: done(t[len(tail) :]), start=start)
        return text[len(tail) :]

    return continue_


def heimi_next_line(model, tok: HeimiTokenizer, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str], str]:
    budget = _budget(model, max_new_tokens)
    seq_len = model.cfg.seq_len
    continue_ = _continuer(model, tok, device, budget)

    def complete(prompt: str) -> str:
        core, tail = split_tail(prompt)
        # 提示只有空白时，从文档开头标记写起
        ids = tok.encode(core)[-(seq_len - budget) :] or [EOT]
        return continue_(ids, tail, lambda t: first_code_line(t)[1])

    return complete


def heimi_infill(model, tok: HeimiTokenizer, device: str, mode: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str, str], str]:
    """complete(前缀, 后缀) 返回接在前缀后面的第一行（到第一个换行为止，不含换行）。"""
    if mode not in ("fim", "prefix"):
        raise ValueError(f"mode 只能是 'fim' 或 'prefix'，收到 {mode!r}")
    budget = _budget(model, max_new_tokens)
    seq_len = model.cfg.seq_len
    continue_ = _continuer(model, tok, device, budget)
    room = seq_len - budget - 3
    max_prefix = room * 2 // 3
    max_suffix = room - max_prefix

    def complete(prefix: str, suffix: str) -> str:
        core, tail = split_tail(prefix)
        if mode == "fim":
            ids = fim_prompt_ids(tok, core, suffix, max_prefix, max_suffix)
        else:
            ids = tok.encode(core)[-(seq_len - budget) :] or [EOT]
        return continue_(ids, tail, lambda t: "\n" in t).split("\n", 1)[0]

    return complete
