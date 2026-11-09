"""共用夹具：迷你配置用于 CPU 单测，级 0 / 级 2 配置用于参数量等精确断言；dash_runs 是窗口测试共用的一个真实迷你 run。"""

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


@pytest.fixture(scope="session")
def dash_runs(tmp_path_factory):
    """一个真实的迷你 run：runs_dir/heimi，两级（smoke0 两步、smoke1 加深后两步），日志与检查点齐全。

    返回 (runs_dir, cfg)。整个测试会话只训练一次，测试不得修改它。
    """
    from minids.train.loop import run_ladder
    from tests.helpers_train import tiny_ladder

    root = tmp_path_factory.mktemp("dash")
    cfg = tiny_ladder(root, stage_steps=(2, 2))
    run_ladder(cfg, root / "runs" / "heimi", "cpu")
    return root / "runs", cfg
