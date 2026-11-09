"""进入新一级：按级配置做增长、验证、对称破缺（spec 7.2、7.3）。

depth：块扩展后 fp32 logits 必须逐位相等。
width：先以 noise_std_rel=0 复制并要求相对误差 ≤ 1e-4，再 break_symmetry，再要求 ≤ 该级 rel_tol。
none：结构必须与前一级相同，原样返回同一个模型对象。
验证失败由 verify_growth 抛 GrowthVerificationError，训练循环据此中止，不进入下一级。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from minids.growth.depth import grow_depth
from minids.growth.verify import GrowthVerificationError, verify_growth
from minids.growth.width import break_symmetry, grow_width
from minids.model.transformer import Transformer
from minids.train.config import StageConfig

WIDTH_EXACT_REL_TOL = 1e-4  # spec 7.3：加宽不加噪时允许的浮点求和顺序误差


@dataclass(frozen=True)
class GrowthOutcome:
    kind: str
    params_before: int
    params_after: int
    exact_report: dict | None  # none 时为 None
    noisy_report: dict | None  # 只有 width 有
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def grow_for_stage(
    prev: Transformer,
    stage: StageConfig,
    verify_tokens: torch.LongTensor,
    seed: int,
) -> tuple[Transformer, GrowthOutcome]:
    t0 = time.perf_counter()
    tokens = verify_tokens.to(next(prev.parameters()).device)
    before = prev.count_params()

    if stage.growth == "none":
        if prev.cfg != stage.model:
            raise ValueError(f"级 {stage.name} growth=none 但结构与前一级不同：{prev.cfg.to_dict()} ≠ {stage.model.to_dict()}")
        return prev, GrowthOutcome("none", before, before, None, None, time.perf_counter() - t0)

    if stage.growth == "depth":
        new = grow_depth(prev)
        report = verify_growth(prev, new, tokens, rel_tol=0.0)
        if not report.exact:
            raise GrowthVerificationError(f"加深后 logits 不是逐位相等：max_abs_diff={report.max_abs_diff:.3e}")
        exact, noisy = report.to_dict(), None
    elif stage.growth == "width":
        new = grow_width(prev, noise_std_rel=0.0)
        exact = verify_growth(prev, new, tokens, rel_tol=WIDTH_EXACT_REL_TOL).to_dict()
        break_symmetry(new, stage.noise_std_rel, torch.Generator().manual_seed(seed))
        noisy = verify_growth(prev, new, tokens, rel_tol=stage.rel_tol).to_dict()
    else:
        raise ValueError(f"未知 growth={stage.growth!r}")

    if new.cfg != stage.model:
        raise ValueError(f"增长后结构 {new.cfg.to_dict()} 与级 {stage.name} 的配置 {stage.model.to_dict()} 不一致")
    return new, GrowthOutcome(stage.growth, before, new.count_params(), exact, noisy, time.perf_counter() - t0)
