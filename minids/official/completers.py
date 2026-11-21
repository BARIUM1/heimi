"""黑米的两个补全函数：下一行补全、单行填空，交给 codebench 的计分器打分。

提示的最后一个 token 如果只是空白，由模型自己重写（规则和原因见 healing.py）；返回的是接在原提示后面的文本。

计分器按原文打分，所以这里保证返回的文本里没有任何特殊 token 的字面串：一遇到特殊 token 就停，它不计入结果
（早期检查点、没训过 FIM 的检查点常常一上来就写特殊 token）。这时返回的是在它之前写出的东西：
通常是空串；如果重写的那个 token 连缩进一起写了，就是那几个空格。两种情况计分器都记为答错。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from minids.completion import fim_prompt_ids
from minids.data.tokenizer import EOT, FIM_SUFFIX, HeimiTokenizer
from minids.official.codebench import first_code_line
from minids.official.healing import Healer

CODE_MAX_NEW_TOKENS = 64


@torch.no_grad()
def greedy_text(
    model, tok: HeimiTokenizer, prompt_ids: list[int], max_new_tokens: int, device: str, done: Callable[[str], bool],
    n_special: int = 16, start: Sequence[int] = (),
) -> str:
    """贪心生成，直到 done(已生成文本) 为真、遇到特殊 id（< n_special，不计入结果）或用完预算；返回解码文本。

    start 是已经定下来的开头 token（重写提示最后那个空白 token 的那一个）：接在提示后面，算进 max_new_tokens，也算进返回的文本。

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


def _continuer(model, tok: HeimiTokenizer, device: str, budget: int) -> Callable[[list[int], list[int], Callable[[str], bool]], str]:
    """返回 continue_(head, rest, done)：提示是 head + rest，返回接在原提示文本后面的文字。

    head 的最后一个 token 只是空白时，它由模型重写（healing.py）：不放进提示，第一个新 token 在文字以它开头的 token 里
    取打分最高的，这个 token 不占 budget。rest 是排在它后面、原样保留的 token（FIM 的后缀和标记；别的用法是空表）。
    done 收到的是原提示之后的文字。
    """
    healer = Healer(tok.decode([i]) for i in range(tok.vocab_size))

    @torch.no_grad()
    def continue_(head: list[int], rest: list[int], done: Callable[[str], bool]) -> str:
        kept, healed, allowed = healer.heal(head)
        # 提示只有一个空白 token（或干脆是空的）时，从文档开头标记写起
        prompt = (kept + rest) or [EOT]
        start = []
        if healed:
            logits, _ = model(torch.tensor([prompt], dtype=torch.long, device=device))
            start = [allowed[int(logits[0, -1].float()[allowed].argmax())]]
        text = greedy_text(model, tok, prompt, budget + len(start), device, done=lambda t: done(t[len(healed) :]), start=start)
        return text[len(healed) :]

    return continue_


def heimi_next_line(model, tok: HeimiTokenizer, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str], str]:
    budget = _budget(model, max_new_tokens)
    seq_len = model.cfg.seq_len
    continue_ = _continuer(model, tok, device, budget)

    def complete(prompt: str) -> str:
        return continue_(tok.encode(prompt)[-(seq_len - budget) :], [], lambda t: first_code_line(t)[1])

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
        if mode == "fim":
            ids = fim_prompt_ids(tok, prefix, suffix, max_prefix, max_suffix)
            # 分词器从不会从文本里产出标记的 id，第一个 <|fim_suffix|> 就是标记：它前面是前缀段，可能要重写的是前缀段的最后一个 token
            cut = ids.index(FIM_SUFFIX)
            head, rest = ids[:cut], ids[cut:]
        else:
            head, rest = tok.encode(prefix)[-(seq_len - budget) :], []
        return continue_(head, rest, lambda t: "\n" in t).split("\n", 1)[0]

    return complete
