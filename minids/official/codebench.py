"""代码补全评测的计分：RepoBench 下一行补全、HumanEval 单行填空。

只做计分：不跑模型、不联网、不执行模型写出来的代码。模型怎么生成由调用方传进来的 complete 决定。
"""
import difflib
import random
import re
from typing import Callable, Sequence

SAMPLE_SEED = 1337


def first_code_line(text: str) -> tuple[str, bool]:
    """官方取行规则：跳过空行、注释和三引号块，取第一行真正的代码。

    返回 (这一行, 这一行是否已经写完，即后面还跟着换行)。找不到代码行时退回第一行，并算没写完。
    """
    lines = text.lstrip("\n").split("\n")
    in_block = False
    for index, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        opens = s.startswith(('"""', "'''"))
        closes = s.endswith(('"""', "'''"))
        if not in_block and opens:
            in_block = True
            continue
        if in_block and closes:
            in_block = False
            continue
        if in_block or s.startswith("#"):
            continue
        return line, index < len(lines) - 1
    return lines[0], False


def edit_similarity(pred: str, gt: str) -> int:
    """0-100 的编辑相似度，等同 fuzzywuzzy.fuzz.ratio 的纯 Python 路径（先 gt 后 pred）。"""
    if pred == gt:
        return 100
    if not pred or not gt:
        return 0
    return int(round(100 * difflib.SequenceMatcher(None, gt, pred).ratio()))


def lines_match(pred: str, gt: str) -> bool:
    """只忽略空白差异的整行相等。"""
    return pred.split() == gt.split()


def repobench_prompt(row: dict) -> str:
    """官方的同文件部分：路径注释、导入、裁剪后的代码；连续空行压成一个。"""
    text = f"# Path: {row['file_path']}\n{row['import_statement']}\n{row['cropped_code'].rstrip()}\n"
    return re.sub(r"\n{4,}", "\n\n", text)


def sample_rows(rows: Sequence[dict], limit: int | None, seed: int = SAMPLE_SEED) -> list[dict]:
    """固定种子抽样并保持原顺序；limit 为 None 或不小于总数时返回全部。"""
    if limit is None or limit >= len(rows):
        return list(rows)
    picked = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in picked]


def repobench_scores(complete: Callable[[str], str], rows: Sequence[dict]) -> dict:
    """complete 收到提示，返回模型生成的原始文本；返回 EM、ES（0-100）和条数。"""
    if not rows:
        raise ValueError("没有可计分的 RepoBench 题目")
    hits = 0
    similarity = 0
    for row in rows:
        pred = first_code_line(complete(repobench_prompt(row)))[0]
        hits += lines_match(pred, row["next_line"])
        similarity += edit_similarity(pred, row["next_line"])
    n = len(rows)
    return {"em": round(hits / n, 5), "es": round(similarity / n, 5), "n": n}


def infill_suffix(row: dict) -> str:
    """交给模型的后缀：待填那一行行末的换行算在后缀开头。

    原数据把这个换行算在答案末尾，后缀从下一行的缩进开始。挪过来以后，后缀是在一串空白的起点切开的，
    和模型训练时见到的切法一致（原因见 healing.py），整篇文本一个字符都没变。
    这里不看答案：每道题的答案都是以换行结尾的一整行，由 codedata.load_infill 在读数据时检查。
    """
    return "\n" + row["suffix"]


def infill_scores(complete: Callable[[str, str], str], rows: Sequence[dict]) -> dict:
    """complete 收到 (prompt, infill_suffix)，返回生成的文本；比较第一行与参考答案，返回 EM 和条数。"""
    if not rows:
        raise ValueError("没有可计分的单行填空题目")
    hits = 0
    for row in rows:
        generated = complete(row["prompt"], infill_suffix(row))
        hits += generated.split("\n", 1)[0].strip() == row["canonical_solution"].strip()
    n = len(rows)
    return {"em": round(hits / n, 5), "n": n}
