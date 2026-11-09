"""增长验证：在一批真实 token 上比较增长前后模型的 logits。

两个模型都以 fp32 拷贝、eval 模式、no_grad 前向，不改动原模型。
相对误差超过阈值就抛异常，训练循环据此中止，不进入下一级。
"""

import copy
from dataclasses import asdict, dataclass

import torch
from torch import LongTensor

from minids.model.transformer import Transformer


class GrowthVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GrowthReport:
    params_before: int
    params_after: int
    max_abs_diff: float
    max_rel_err: float  # max|Δlogit| / max|logit_before|
    exact: bool  # torch.equal 为 True

    def to_dict(self) -> dict:
        return asdict(self)


def verify_growth(
    old: Transformer,
    new: Transformer,
    tokens: LongTensor,
    rel_tol: float = 1e-2,
) -> GrowthReport:
    device = tokens.device
    a = copy.deepcopy(old).float().to(device).eval()
    b = copy.deepcopy(new).float().to(device).eval()
    with torch.no_grad():
        logits_a = a(tokens)[0]
        logits_b = b(tokens)[0]

    max_abs_diff = (logits_a - logits_b).abs().max().item()
    scale = logits_a.abs().max().item()
    if scale > 0:
        max_rel_err = max_abs_diff / scale
    else:
        max_rel_err = 0.0 if max_abs_diff == 0.0 else float("inf")

    report = GrowthReport(
        params_before=old.count_params(),
        params_after=new.count_params(),
        max_abs_diff=max_abs_diff,
        max_rel_err=max_rel_err,
        exact=torch.equal(logits_a, logits_b),
    )
    if max_rel_err > rel_tol:
        raise GrowthVerificationError(
            f"增长验证失败：max_rel_err={max_rel_err:.3e} 超过 rel_tol={rel_tol:.3e}"
            f"（max_abs_diff={max_abs_diff:.3e}，参数 {report.params_before} → {report.params_after}）"
        )
    return report
