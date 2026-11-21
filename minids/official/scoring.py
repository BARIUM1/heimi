"""官方评测的打分核心：给定上下文，算续写的对数似然；以及滚动窗口对数似然。

不依赖评测框架，只依赖模型的前向。float32、不开 autocast，结果可与框架的打分逐项对照。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from minids.data.tokenizer import EOT, PAD

Pair = tuple[list[int], list[int]]


def encode_pair(tok, context: str, continuation: str, prefix_token: int = EOT) -> Pair:
    """把（上下文, 续写）文本编成 token。

    空上下文用 prefix_token 代替。上下文末尾的空白挪到续写开头，
    这样空格和后面的词一起编码，与训练时的切分一致。
    """
    if context == "":
        return [prefix_token], tok.encode(continuation)
    stripped = len(context.rstrip())
    context, continuation = context[:stripped], context[stripped:] + continuation
    whole = tok.encode(context + continuation)
    ctx = tok.encode(context)
    return ctx, whole[len(ctx) :]


@torch.no_grad()
def continuation_logprobs(
    model,
    pairs: Sequence[Pair],
    device: str,
    max_len: int,
    batch_size: int = 64,
    pad_id: int = PAD,
) -> list[tuple[float, bool]]:
    """每对 (ctx, cont) 返回 (续写的对数似然之和, 续写是否每个 token 都是模型的贪心选择)。

    上下文过长时从左边截断，续写完整保留；结果与 pairs 同序。
    不改变模型的 train/eval 状态，调用方保证是 eval。
    """
    results: list[tuple[float, bool]] = [(0.0, True)] * len(pairs)
    todo: list[tuple[int, list[int], list[int]]] = []  # (原序号, 输入, 续写)
    for i, (ctx, cont) in enumerate(pairs):
        if not cont:
            continue
        if not ctx:
            # 没有上下文就没有位置可以用来预测续写的第一个 token；encode_pair / rolling_windows 都会补前缀 token
            raise ValueError("上下文为空，无法给续写打分（空上下文应先用前缀 token 代替，见 encode_pair）")
        if len(cont) > max_len:
            raise ValueError(f"续写有 {len(cont)} 个 token，超过窗口 {max_len}；这种文本应走 rolling_logprob")
        inp = (list(ctx) + list(cont))[-(max_len + 1) :][:-1]
        todo.append((i, inp, list(cont)))
    todo.sort(key=lambda t: len(t[1]), reverse=True)  # 长的在前，批内填充最少

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        width = len(batch[0][1])
        x = torch.full((len(batch), width), pad_id, dtype=torch.long)
        for row, (_, inp, _) in enumerate(batch):
            x[row, : len(inp)] = torch.tensor(inp)  # 右填充：因果注意力下不影响前面的位置
        logp = F.log_softmax(model(x.to(device))[0].float(), dim=-1)
        for row, (i, inp, cont) in enumerate(batch):
            rows = logp[row, len(inp) - len(cont) : len(inp)]
            want = torch.tensor(cont, device=rows.device)
            lp = rows[torch.arange(len(cont), device=rows.device), want].double().sum().item()
            results[i] = (lp, bool((rows.argmax(-1) == want).all()))
    return results


def rolling_windows(tokens: Sequence[int], prefix_token: int, max_len: int) -> list[Pair]:
    """把长文本切成互不重叠的续写窗口，每窗尽量带满上下文（与框架的滚动窗口逐窗相同）。"""
    tokens = list(tokens)
    if not tokens:
        return []
    first = min(max_len, len(tokens))
    out = [([prefix_token], tokens[:first])]
    done = first
    while done < len(tokens):
        n = min(len(tokens) - done, max_len)
        end = done + n
        out.append((tokens[end - max_len - 1 : end - n], tokens[end - n : end]))
        done = end
    return out


def rolling_logprob(
    model,
    tokens: Sequence[int],
    device: str,
    max_len: int,
    prefix_token: int = EOT,
    batch_size: int = 8,
) -> float:
    """整段文本的对数似然：各窗口之和；空序列为 0.0。"""
    windows = rolling_windows(tokens, prefix_token, max_len)
    scores = continuation_logprobs(model, windows, device, max_len, batch_size=batch_size)
    return sum(lp for lp, _ in scores)
