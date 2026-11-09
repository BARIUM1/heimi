"""HellaSwag 准确率（spec 10）：10,042 条验证集，按续写的归一化对数似然选项。

数据来自 Hugging Face 数据集 Rowan/hellaswag 的 validation parquet（原 GitHub 上的 jsonl 链接已失效）；
label 列是字符串，读入时转 int。小模型上接近随机（0.25），主要在级 2 与任务阶段有区分度。
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from minids.data.tokenizer import PAD, HeimiTokenizer

HELLASWAG_REPO = "Rowan/hellaswag"
HELLASWAG_FILE = "data/validation-00000-of-00001.parquet"  # 6.3 MB，10,042 行


def download_hellaswag(dest: Path) -> Path:
    """dest 已存在直接返回；否则从 Hugging Face 下载到临时目录再移到 dest。"""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        got = hf_hub_download(repo_id=HELLASWAG_REPO, filename=HELLASWAG_FILE, repo_type="dataset", local_dir=tmp)
        shutil.move(got, dest)
    return dest


def load_hellaswag(path: Path) -> list[dict]:
    table = pq.read_table(path, columns=["ctx", "endings", "label"])
    cols = [table.column(c).to_pylist() for c in ("ctx", "endings", "label")]
    return [{"ctx": ctx, "endings": list(endings), "label": int(label)} for ctx, endings, label in zip(*cols)]


@torch.no_grad()
def ending_scores(model, ctx_ids: list[int], endings_ids: list[list[int]], device: str, max_len: int, pad_id: int = PAD) -> list[float]:
    """每个候选 = ctx + ending（超长从左截断），右填充成一批前向一次；返回每个候选 ending 各 token log_softmax 的均值。"""
    seqs: list[list[int]] = []
    ending_lens: list[int] = []
    for e in endings_ids:
        s = list(ctx_ids) + list(e)
        if len(s) > max_len:
            s = s[-max_len:]
        seqs.append(s)
        ending_lens.append(min(len(e), len(s) - 1))  # 至少留一个 token 在前面做条件
    width = max(len(s) for s in seqs)
    x = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    x = x.to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
    with ctx:
        logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    scores: list[float] = []
    for i, s in enumerate(seqs):
        n, k = len(s), ending_lens[i]
        if k <= 0:
            scores.append(float("-inf"))
            continue
        targets = torch.tensor(s[n - k : n], device=device)
        positions = torch.arange(n - k - 1, n - 1, device=device)  # 预测 ending 各 token 的位置
        scores.append(logp[i, positions, targets].double().mean().item())
    return scores


def choose_ending(model, ctx_ids: list[int], endings_ids: list[list[int]], device: str, max_len: int, pad_id: int = PAD) -> int:
    """归一化对数似然最大的候选下标；并列时取第一个。"""
    scores = ending_scores(model, ctx_ids, endings_ids, device, max_len, pad_id)
    return max(range(len(scores)), key=lambda i: scores[i])


def hellaswag_accuracy(model, tok: HeimiTokenizer, rows: Sequence[dict], device: str, max_len: int, limit: int | None = None) -> float:
    """ending 前补一个空格再编码（续写接在 ctx 后面通常以空格开头）。结束后恢复训练模式。"""
    rows = list(rows)[:limit] if limit else list(rows)
    if not rows:
        raise ValueError("没有 HellaSwag 样例可评测")
    was_training = model.training
    model.eval()
    correct = 0
    for r in rows:
        ctx_ids = tok.encode(r["ctx"])
        endings_ids = [tok.encode(" " + e) for e in r["endings"]]
        correct += choose_ending(model, ctx_ids, endings_ids, device, max_len) == r["label"]
    model.train(was_training)
    return correct / len(rows)
