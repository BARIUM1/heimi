"""提示边界的处理：提示结尾的那串空白不放进提示，交给模型自己写出来（常说的 token healing）。

为什么要这样：代码评测的提示都停在换行后面，而黑米的分词器把"换行 + 后面的缩进"并成一个 token。
"提示以单独一个换行 token 结尾、下一行却带缩进"的 token 序列在它的训练数据里几乎不出现，
模型于是接着写顶格的内容，评测只取第一行，答案就成了空的。GPT-2 的分词器没有这个问题。
实测数字见 docs/official-eval.md。

做法：把提示结尾的空白（tail）拿掉；生成时先只许模型选"文本和 tail 对得上"的 token，
直到 tail 写完（最后一个 token 可以多带一截，比如把缩进一起写了），之后放开。
模型拿到的文本信息和原提示一样（tail 是什么已经定死），只是 token 在哪里切由它自己的分词习惯决定。
黑米和 GPT-2 走同一套规则。

本模块只有纯 Python 的规则，不 import torch；模型给每个 token 打多少分，由调用方传进来的 pick 去问。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

WHITESPACE = " \t\r\n"


def split_tail(text: str) -> tuple[str, str]:
    """(正文, tail)：tail 是结尾的那一整串空格、制表符、回车、换行，可以是空串。"""
    core = text.rstrip(WHITESPACE)
    return core, text[len(core) :]


def whitespace_tokens(token_texts: Iterable[str]) -> dict[int, str]:
    """id → 文本，只留文本以 WHITESPACE 里的字符开头的 token；只有它们可能用来写 tail。"""
    return {i: t for i, t in enumerate(token_texts) if t[:1] and t[0] in WHITESPACE}


def allowed_ids(remaining: str, ws_tokens: dict[int, str]) -> list[int]:
    """还剩 remaining 没写时可以选的 token：文本是 remaining 的开头一段，或者以 remaining 开头（写完还多带一截）。"""
    return [i for i, t in ws_tokens.items() if t.startswith(remaining) or remaining.startswith(t)]


def write_tail(tail: str, ws_tokens: dict[int, str], pick: Callable[[list[int], list[int]], int], max_tokens: int) -> list[int] | None:
    """让模型把 tail 写出来，返回它选的 token。

    pick(已经写下的 token, 这一步允许的 id) 返回选中的 id（调用方在允许的 id 里取模型打分最高的）。
    写不出来时返回 None：词表里没有对得上的 token，或者用了 max_tokens 个还没写完。
    """
    written: list[int] = []
    remaining = tail
    while remaining:
        allowed = allowed_ids(remaining, ws_tokens)
        if not allowed or len(written) >= max_tokens:
            return None
        written.append(pick(written, allowed))
        remaining = remaining[len(ws_tokens[written[-1]]) :]
    return written
