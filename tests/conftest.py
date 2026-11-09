"""共用夹具：迷你配置用于 CPU 单测，级 0 / 级 2 配置用于参数量等精确断言。"""

import pytest

from minids.config import ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        seq_len=64,
        n_layer=2,
        d_model=64,
        n_head=2,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=32,
        kv_lora_rank=32,
        ffn_hidden=128,
    )


@pytest.fixture
def stage0_cfg() -> ModelConfig:
    # 默认值即 spec 5.3 的级 0
    return ModelConfig()


@pytest.fixture
def stage2_cfg() -> ModelConfig:
    return ModelConfig(n_layer=16, d_model=768, n_head=12, kv_lora_rank=256, ffn_hidden=2048)
