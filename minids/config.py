"""模型配置。

字段默认值就是 spec 5.3 的级 0 结构。配置是不可变的数据类：
增长不修改原配置，而是通过 grown_depth / grown_width 派生出新配置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

_POSITIVE_INT_FIELDS = (
    "vocab_size",
    "seq_len",
    "n_layer",
    "d_model",
    "n_head",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "kv_lora_rank",
    "ffn_hidden",
)
_POSITIVE_FLOAT_FIELDS = ("rope_base", "norm_eps", "init_std")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16384
    seq_len: int = 1024
    n_layer: int = 8
    d_model: int = 384
    n_head: int = 6
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64
    kv_lora_rank: int = 128
    ffn_hidden: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def __post_init__(self) -> None:
        self.validate()

    @property
    def qk_head_dim(self) -> int:
        """每个头的 Q/K 总维度：不带位置的部分加带 RoPE 的部分。"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def validate(self) -> None:
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数，得到 {value!r}")
        for name in _POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} 必须是正数，得到 {value!r}")
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError(f"qk_rope_head_dim 必须是偶数（RoPE 按两两一对旋转），得到 {self.qk_rope_head_dim}")
        if self.vocab_size < 16:
            raise ValueError(f"vocab_size 至少为 16（前 16 个位置留给特殊 token），得到 {self.vocab_size}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def grown_depth(self) -> "ModelConfig":
        """加深：层数翻倍，其余不变。"""
        return replace(self, n_layer=self.n_layer * 2)

    def grown_width(self) -> "ModelConfig":
        """加宽：模型维度、头数、潜向量维度、前馈隐层翻倍；每头内部维度不变。"""
        return replace(
            self,
            d_model=self.d_model * 2,
            n_head=self.n_head * 2,
            kv_lora_rank=self.kv_lora_rank * 2,
            ffn_hidden=self.ffn_hidden * 2,
        )
