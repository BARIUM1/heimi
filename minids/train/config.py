"""训练配置：整条阶梯、各级结构与超参数、数据路径、评测节奏。

yaml 结构见 configs/ladder.yaml：顶层是全局项，`model` 是各级共用的结构字段，
`stages` 里每级的 `model` 覆盖在它之上再构造 ModelConfig。

加载时做两类校验：
- 增长链：depth 级的结构必须等于前一级 grown_depth()，width 级等于 grown_width()，
  none 级（非首级）与前一级相同。配置和算子说的不是一回事时在这里就报错。
- 字段类型：PyYAML 把 2e-3 读成字符串（2.0e-3 才是浮点数），字符串学习率要在这里拦下，
  不能等到优化器里才炸。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from minids.config import ModelConfig
from minids.data.loader import MixConfig

GROWTH_KINDS = ("none", "depth", "width")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _require_int(owner: str, name: str, value: object, minimum: int = 1) -> None:
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{owner} 的 {name} 必须是 ≥ {minimum} 的整数，得到 {value!r}")


def _require_number(owner: str, name: str, value: object) -> None:
    if not _is_number(value):
        raise ValueError(f"{owner} 的 {name} 必须是数字，得到 {value!r}（yaml 里科学计数法要写成 2.0e-3，2e-3 会被读成字符串）")


@dataclass(frozen=True)
class DataPaths:
    index: str = "data/tokens/index.json"
    tokenizer: str = "data/tokenizer/heimi-16k.json"
    hellaswag: str = "data/eval/hellaswag_val.parquet"


@dataclass(frozen=True)
class EvalConfig:
    interval: int = 500  # 每多少步评测
    val_windows: int = 200  # 每来源固定窗口数
    hellaswag: bool = True  # 级末是否跑 HellaSwag
    fim_em_samples: int = 1000
    sample_interval: int = 500
    sample_tokens: int = 64

    def __post_init__(self) -> None:
        for name in ("interval", "val_windows", "fim_em_samples", "sample_interval", "sample_tokens"):
            _require_int("eval", name, getattr(self, name))


@dataclass(frozen=True)
class Prompt:
    id: str
    prefix: str
    suffix: str | None = None  # 非 None 即 FIM 填空提示


@dataclass(frozen=True)
class StageConfig:
    name: str
    growth: str  # GROWTH_KINDS 之一
    model: ModelConfig
    tokens: int
    lr: float
    warmup_frac: float
    micro_batch: int
    mix: MixConfig
    noise_std_rel: float = 1e-3
    rel_tol: float = 1e-2

    def __post_init__(self) -> None:
        owner = f"级 {self.name}"
        if self.growth not in GROWTH_KINDS:
            raise ValueError(f"{owner} 的 growth={self.growth!r} 不在 {GROWTH_KINDS} 中")
        _require_int(owner, "tokens", self.tokens)
        _require_int(owner, "micro_batch", self.micro_batch)
        for name in ("lr", "warmup_frac", "noise_std_rel", "rel_tol"):
            _require_number(owner, name, getattr(self, name))
        if self.lr <= 0:
            raise ValueError(f"{owner} 的 lr 必须大于 0，得到 {self.lr}")
        if not 0 <= self.warmup_frac < 1:
            raise ValueError(f"{owner} 的 warmup_frac 必须在 [0, 1) 内，得到 {self.warmup_frac}")
        for name in ("web", "code", "fim_rate"):
            v = getattr(self.mix, name)
            if not _is_number(v) or not 0 <= v <= 1:
                raise ValueError(f"{owner} 的 mix.{name} 必须在 [0, 1] 内，得到 {v!r}")


@dataclass(frozen=True)
class LadderConfig:
    seed: int
    seq_len: int
    tokens_per_step: int
    stages: tuple[StageConfig, ...]
    prompts: tuple[Prompt, ...]
    data: DataPaths = field(default_factory=DataPaths)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log_interval: int = 10
    ckpt_interval: int = 500
    keep_checkpoints: int = 2
    compile: bool = False
    max_mem_frac: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "prompts", tuple(self.prompts))
        self.validate()

    def validate(self) -> None:
        _require_int("配置", "seed", self.seed, minimum=0)
        for name in ("seq_len", "tokens_per_step", "log_interval", "ckpt_interval", "keep_checkpoints"):
            _require_int("配置", name, getattr(self, name))
        _require_number("配置", "max_mem_frac", self.max_mem_frac)
        if not 0 < self.max_mem_frac <= 1:
            raise ValueError(f"max_mem_frac 必须在 (0, 1] 内，得到 {self.max_mem_frac}")
        if not self.stages:
            raise ValueError("stages 不能为空")

        seen: set[str] = set()
        for i, s in enumerate(self.stages):
            if s.name in seen:
                raise ValueError(f"级名重复：{s.name}")
            seen.add(s.name)
            if i == 0 and s.growth != "none":
                raise ValueError(f"第一级 {s.name} 必须是 growth: none（从零初始化），得到 {s.growth}")
            if i > 0:
                prev = self.stages[i - 1].model
                expected = {"none": prev, "depth": prev.grown_depth(), "width": prev.grown_width()}[s.growth]
                if s.model != expected:
                    raise ValueError(
                        f"级 {s.name} 的结构与 growth={s.growth} 不符：按前一级 {self.stages[i - 1].name} 应为 "
                        f"{expected.to_dict()}，实际 {s.model.to_dict()}"
                    )
            if s.model.seq_len != self.seq_len:
                raise ValueError(f"级 {s.name} 的 model.seq_len={s.model.seq_len} 与全局 seq_len={self.seq_len} 不一致")
            if s.model.vocab_size != self.stages[0].model.vocab_size:
                raise ValueError(f"级 {s.name} 的 vocab_size={s.model.vocab_size} 与第一级不一致")
            micro_tokens = s.micro_batch * self.seq_len
            if self.tokens_per_step % micro_tokens != 0:
                raise ValueError(
                    f"tokens_per_step={self.tokens_per_step} 不能被级 {s.name} 的 micro_batch × seq_len = {micro_tokens} 整除"
                )
            if s.tokens < self.tokens_per_step:
                raise ValueError(f"级 {s.name} 的 tokens={s.tokens} 不足一个优化步（tokens_per_step={self.tokens_per_step}）")

    def stage(self, name: str) -> StageConfig:
        return self.stages[self.stage_index(name)]

    def stage_index(self, name: str) -> int:
        for i, s in enumerate(self.stages):
            if s.name == name:
                return i
        raise KeyError(f"没有名为 {name} 的级，可选：{[s.name for s in self.stages]}")

    def steps_for(self, stage: StageConfig) -> int:
        return stage.tokens // self.tokens_per_step

    def accum_steps(self, stage: StageConfig) -> int:
        return self.tokens_per_step // (stage.micro_batch * self.seq_len)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "seq_len": self.seq_len,
            "tokens_per_step": self.tokens_per_step,
            "log_interval": self.log_interval,
            "ckpt_interval": self.ckpt_interval,
            "keep_checkpoints": self.keep_checkpoints,
            "compile": self.compile,
            "max_mem_frac": self.max_mem_frac,
            "data": asdict(self.data),
            "eval": asdict(self.eval),
            "prompts": [asdict(p) for p in self.prompts],
            "stages": [
                {
                    "name": s.name,
                    "growth": s.growth,
                    "model": s.model.to_dict(),
                    "tokens": s.tokens,
                    "lr": s.lr,
                    "warmup_frac": s.warmup_frac,
                    "micro_batch": s.micro_batch,
                    "mix": asdict(s.mix),
                    "noise_std_rel": s.noise_std_rel,
                    "rel_tol": s.rel_tol,
                }
                for s in self.stages
            ],
        }


def ladder_from_dict(d: dict) -> LadderConfig:
    d = dict(d)
    base_model = dict(d.pop("model", None) or {})
    try:
        stages = []
        for sd in d.pop("stages", None) or []:
            sd = dict(sd)
            model = ModelConfig.from_dict({**base_model, **(sd.pop("model", None) or {})})
            mix = MixConfig(**(sd.pop("mix", None) or {}))
            stages.append(StageConfig(model=model, mix=mix, **sd))
        prompts = tuple(Prompt(**p) for p in d.pop("prompts", None) or [])
        data = DataPaths(**(d.pop("data", None) or {}))
        eval_cfg = EvalConfig(**(d.pop("eval", None) or {}))
        return LadderConfig(stages=tuple(stages), prompts=prompts, data=data, eval=eval_cfg, **d)
    except TypeError as e:  # 多余或缺失的字段
        raise ValueError(f"配置字段错误：{e}") from e


def load_ladder(path: str | Path) -> LadderConfig:
    return ladder_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
