"""提示边界的处理：提示的最后一个 token 如果只是空白，就让模型自己重写它（常说的 token healing）。

为什么要这样：代码评测的提示都停在换行后面，而黑米的分词器把"换行 + 后面的缩进"并成一个 token。
"提示以单独一个换行 token 结尾、下一行却带缩进"的 token 序列在它的训练数据里几乎不出现，
模型于是接着写顶格的内容，评测只取第一行，答案就成了空的。GPT-2 的分词器不合并换行和缩进，没有这个问题。
实测数字见 docs/official-eval.md 的"代码题的出题方式"一节。

做法：提示照常分词、照常截取；最后一个 token 如果只由空格、制表符、回车、换行组成，就不放进提示。
生成的第一个 token 只许在"文字以它开头"的 token 里挑（取模型打分最高的）：
可以原样写回同一个 token，这时后面的生成和不做任何处理时一模一样
（下一行补全和只看前缀的填空是这样；FIM 模式下这个 token 从前缀段的末尾挪到了中间段的开头，喂给模型的序列本来就和旧方式不同）；
也可以写一个更长的，比如把换行和后面的缩进一起写出来。之后照常生成。
模型拿到的文本信息和原提示一样（那段空白是什么已经定死），只是这一个 token 在哪里结束由它自己的分词习惯决定。
重写的这一个 token 不占新 token 的预算。黑米和 GPT-2 走同一套规则。

本模块只有纯 Python 的规则，不 import torch；模型给每个 token 打多少分，由调用方去问。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

WHITESPACE = " \t\r\n"


def whitespace_tokens(token_texts: Iterable[str]) -> dict[int, str]:
    """id → 文本，只留文本以 WHITESPACE 里的字符开头的 token；只有它们可能用来重写一个空白 token。"""
    return {i: t for i, t in enumerate(token_texts) if t[:1] and t[0] in WHITESPACE}


class Healer:
    """按一份词表（每个 id 单独解码出来的文本）做上面的处理。"""

    def __init__(self, token_texts: Iterable[str]):
        self._ws = whitespace_tokens(token_texts)
        self._extensions: dict[int, list[int]] = {}

    def heal(self, ids: Sequence[int]) -> tuple[list[int], str, list[int]]:
        """返回 (留在提示里的 token, 拿掉的那个 token 的文字, 第一个新 token 允许的 id)。

        最后一个 token 只由空白组成时拿掉它；允许的 id 是文字以它开头的所有 token，它自己一定在里面。
        否则原样返回，文字是空串、允许的 id 是空表：不做任何处理。只看最后一个 token，前面的空白 token 不动。
        """
        ids = list(ids)
        text = self._ws.get(ids[-1], "") if ids else ""
        if not text or text.strip(WHITESPACE):
            return ids, "", []
        if ids[-1] not in self._extensions:
            self._extensions[ids[-1]] = [i for i, t in self._ws.items() if t.startswith(text)]
        return ids[:-1], text, self._extensions[ids[-1]]
