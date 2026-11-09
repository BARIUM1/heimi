"""评测套件：把验证窗口、FIM 样例、HellaSwag 行预加载一次，训练循环每次评测直接用，evaluate.py 也走这里。

验证窗口固定、FIM 样例由种子固定、HellaSwag 全量：各阶段的评测比的是同一批数据。
"""

from __future__ import annotations

from pathlib import Path

import torch

from minids.data.tokenizer import HeimiTokenizer
from minids.eval.fim_em import cases_from_index, fim_exact_match
from minids.eval.hellaswag import hellaswag_accuracy, load_hellaswag
from minids.eval.val_loss import fixed_val_windows, val_loss
from minids.train.config import LadderConfig


class EvalSuite:
    def __init__(self, cfg: LadderConfig, tok: HeimiTokenizer, device: str):
        self.cfg = cfg
        self.tok = tok
        self.device = device
        seq_len = cfg.seq_len
        index = Path(cfg.data.index)
        self.windows = {src: fixed_val_windows(index, src, cfg.eval.val_windows, seq_len) for src in ("web", "code")}
        self.fim_cases = cases_from_index(index, tok, cfg.eval.fim_em_samples, cfg.seed)
        self.hellaswag_rows: list[dict] | None = None
        if cfg.eval.hellaswag:
            p = Path(cfg.data.hellaswag)
            if not p.exists():
                raise FileNotFoundError(
                    f"HellaSwag 文件 {p} 不存在：train.py 会自动下载，或调用 minids.eval.hellaswag.download_hellaswag(path)"
                )
            self.hellaswag_rows = load_hellaswag(p)
        # FIM 生成预算：前缀、后缀、3 个标记、新 token 之和不超过 seq_len
        self.max_prefix_tokens = min(512, seq_len // 2)
        self.max_suffix_tokens = min(256, seq_len // 4)
        self.max_new_tokens = min(64, seq_len - self.max_prefix_tokens - self.max_suffix_tokens - 3)

    @property
    def growth_tokens(self) -> torch.LongTensor:
        """增长验证用的一批真实 token：web 固定窗口的前 4 个（去掉最后一列）。"""
        return self.windows["web"][:4, :-1]

    def run(self, model, *, micro_batch: int, hellaswag: bool, fim: bool) -> dict:
        out = {
            "val_loss_web": val_loss(model, self.windows["web"], self.device, micro_batch),
            "val_loss_code": val_loss(model, self.windows["code"], self.device, micro_batch),
            "hellaswag_acc": None,
            "fim_em": None,
        }
        was_training = model.training
        model.eval()
        if hellaswag and self.hellaswag_rows is not None:
            out["hellaswag_acc"] = hellaswag_accuracy(model, self.tok, self.hellaswag_rows, self.device, max_len=self.cfg.seq_len)
        if fim:
            out["fim_em"] = fim_exact_match(
                model, self.tok, self.fim_cases, self.device,
                self.max_new_tokens, self.max_prefix_tokens, self.max_suffix_tokens,
            )
        model.train(was_training)
        return out
