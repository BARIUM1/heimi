"""参照模型 GPT-2 small：走同一套框架任务和代码评测，进同样的指标名。

transformers 只在 load_gpt2 函数体内 import；GPT-2 没有 FIM 训练，所以只有前缀补全的填空数。
提示结尾的空白和黑米一样交给模型自己写（规则见 healing.py）。
文本一律用 skip_special_tokens=True 解码，返回值里不会有 <|endoftext|>；
第一个新 token 就是 EOS 或没生成任何东西时返回空串，计分器照常记为答错。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from minids.logging_io import now_iso
from minids.official.completers import CODE_MAX_NEW_TOKENS
from minids.official.healing import split_tail, whitespace_tokens, write_tail
from minids.official.results import CODE_METHOD
from minids.official.run import CODE_LIMIT, code_metrics

GPT2_ID = "openai-community/gpt2"
GPT2_LABEL = "GPT-2 small"


def load_gpt2(device: str) -> tuple[object, object, int]:
    """返回 (模型, 分词器, 非嵌入参数量)；float32。输出层与 wte 共享权重，只减一次 wte，再减 wpe。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(GPT2_ID, dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(GPT2_ID)
    total = sum(p.numel() for p in model.parameters())
    embed = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    return model, tok, total - embed


def _hf_continuer(hf_model, hf_tok, device: str, max_new_tokens: int) -> Callable[[str], str]:
    """返回 continue_(提示)：贪心生成，返回接在提示后面的文本（不含特殊 token）。

    取提示正文的末尾，新 token 预算最多占上下文一半；提示结尾的空白（tail）由模型在限定的 token 里自己写出来，
    写 tail 的 token 也算在预算里，预算内写不完时返回空串。解码一律不做空格清理，文本要和 token 一一对得上。
    """
    n_positions = hf_model.config.n_positions
    budget = max(1, min(max_new_tokens, n_positions // 2))
    ws_tokens = whitespace_tokens(hf_tok.decode([i], clean_up_tokenization_spaces=False) for i in range(len(hf_tok)))

    @torch.no_grad()
    def continue_(prompt: str) -> str:
        core, tail = split_tail(prompt)
        # 提示只有空白时，从文档开头标记写起
        ids = hf_tok.encode(core)[-(n_positions - budget) :] or [hf_tok.eos_token_id]

        def pick(written: list[int], allowed: list[int]) -> int:
            logits = hf_model(torch.tensor([ids + written], device=device)).logits[0, -1]
            return allowed[int(logits.float()[allowed].argmax())]

        new = write_tail(tail, ws_tokens, pick, budget)
        if new is None:
            return ""
        if len(new) < budget:
            out = hf_model.generate(
                torch.tensor([ids + new], device=device), max_new_tokens=budget - len(new), do_sample=False, pad_token_id=hf_tok.eos_token_id
            )
            new = out[0, len(ids) :].tolist()
        return hf_tok.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[len(tail) :]

    return continue_


def hf_next_line(hf_model, hf_tok, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str], str]:
    return _hf_continuer(hf_model, hf_tok, device, max_new_tokens)


def hf_infill_prefix(hf_model, hf_tok, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str, str], str]:
    """只看前缀，忽略后缀，取第一个换行之前的部分。"""
    continue_ = _hf_continuer(hf_model, hf_tok, device, max_new_tokens)
    return lambda prefix, suffix: continue_(prefix).split("\n", 1)[0]


def evaluate_reference(
    device: str, data_dir: Path, limit: int | None = None, code_limit: int | None = CODE_LIMIT,
    harness: Callable | None = None, loader: Callable | None = None,
) -> dict:
    from minids.official.harness import harness_version

    if harness is None:
        from minids.official.harness import run_harness as harness
    hf_model, hf_tok, params = (loader or load_gpt2)(device)
    # model_args 明确 float32，官方打分不用半精度
    metrics = dict(harness("hf", limit=limit, model_args=f"pretrained={GPT2_ID},dtype=float32", device=device, batch_size=64))
    if code_limit != 0:
        metrics.update(code_metrics(hf_next_line(hf_model, hf_tok, device), {"prefix": hf_infill_prefix(hf_model, hf_tok, device)}, data_dir, code_limit))
    return {
        "label": GPT2_LABEL, "model_id": GPT2_ID, "params_non_embedding": params, "limit": limit, "code_limit": code_limit,
        "metrics": metrics, "harness_version": harness_version(), "time": now_iso(), "code_method": CODE_METHOD,
    }
