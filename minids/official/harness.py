"""把黑米的检查点接到 lm-evaluation-harness：打分走 scoring.py，框架负责任务与数据。

本文件是唯一在模块顶层 import lm_eval 的地方；没装 official 额外依赖时别的模块不受影响。
"""

from __future__ import annotations

import logging
from typing import Sequence

import lm_eval
from lm_eval.api.model import LM

from minids.official.scoring import continuation_logprobs, encode_pair, rolling_logprob

log = logging.getLogger(__name__)

HARNESS_TASKS: tuple[str, ...] = ("blimp", "hellaswag", "wikitext")


class HeimiLM(LM):
    """只实现打分两个接口；生成类任务不经框架。"""

    def __init__(self, model, tok, device: str, batch_size: int = 64):
        super().__init__()
        self.model = model
        self.tok = tok
        self._device = device  # 框架基类的 device 是只读属性，取值自 _device
        self.batch_size = batch_size
        self.max_len = model.cfg.seq_len

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        log.info("loglikelihood：%d 条请求", len(requests))  # 全量 BLiMP 有 13.4 万条，要让人看到在动
        pairs = [encode_pair(self.tok, *req.args) for req in requests]
        scored = continuation_logprobs(self.model, pairs, self.device, self.max_len, self.batch_size)
        return [(float(lp), bool(greedy)) for lp, greedy in scored]

    def loglikelihood_rolling(self, requests) -> list[float]:
        log.info("loglikelihood_rolling：%d 条请求", len(requests))
        return [float(rolling_logprob(self.model, self.tok.encode(req.args[0]), self.device, self.max_len)) for req in requests]

    def generate_until(self, requests):
        raise NotImplementedError("黑米不经评测框架做生成；代码补全走 minids.official 自己的评测")


def harness_version() -> str:
    return lm_eval.__version__


def flatten_results(results: dict, tasks: Sequence[str]) -> dict[str, float]:
    """从框架结果里取出所列任务的数值指标，记成 任务_指标 的扁平字典。"""
    out: dict[str, float] = {}
    for task in tasks:
        entry = results["results"].get(task)
        if entry is None:
            raise ValueError(f"评测框架没有返回任务 {task}")
        for key, value in entry.items():
            if not key.endswith(",none") or "_stderr" in key:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{task}_{key.split(',')[0]}"] = value
    return out


def run_harness(model, tasks: Sequence[str] = HARNESS_TASKS, limit: int | None = None, **kwargs) -> dict[str, float]:
    """model 是 LM 实例或框架的模型名（如 "hf"）；kwargs 原样交给 simple_evaluate。需要联网取数据。"""
    results = lm_eval.simple_evaluate(model=model, tasks=list(tasks), limit=limit, bootstrap_iters=0, log_samples=False, **kwargs)
    return flatten_results(results, tasks)
