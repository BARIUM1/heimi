"""固定提示采样（spec 9.2 的 samples.jsonl）：同一批提示在训练各阶段的续写，窗口的"成长样本墙"用。

普通提示用温度 0.8、top-k 50 采样，随机数在 fork_rng 里按 seed 固定，不打扰训练的 RNG；
FIM 提示走 complete_middle 贪心，展示文本用 FIM_HOLE 标出空位。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from minids.completion import FIM_HOLE, complete_middle
from minids.data.tokenizer import HeimiTokenizer
from minids.train.config import Prompt


def generate_samples(
    model,
    tok: HeimiTokenizer,
    prompts: Sequence[Prompt],
    max_new_tokens: int,
    device: str,
    seed: int,
    max_prefix_tokens: int = 512,
    max_suffix_tokens: int = 256,
) -> list[tuple[str, str, str]]:
    """每条提示返回 (prompt_id, 展示文本, completion)。"""
    is_cuda = str(device).startswith("cuda")
    devices = [torch.device(device).index or 0] if is_cuda else []
    out: list[tuple[str, str, str]] = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for p in prompts:
            if p.suffix is not None:
                completion = complete_middle(
                    model, tok, p.prefix, p.suffix, max_new_tokens, device, max_prefix_tokens, max_suffix_tokens
                )
                shown = p.prefix + FIM_HOLE + p.suffix
            else:
                ids = torch.tensor([tok.encode(p.prefix)], dtype=torch.long, device=device)
                gen = model.generate(ids, max_new_tokens, temperature=0.8, top_k=50)
                completion = tok.decode(gen[0, ids.shape[1] :].tolist())
                shown = p.prefix
            out.append((p.id, shown, completion))
    return out
