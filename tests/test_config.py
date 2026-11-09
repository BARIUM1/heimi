import pytest

from minids.config import ModelConfig


def test_defaults_match_spec_stage0(stage0_cfg):
    assert (stage0_cfg.n_layer, stage0_cfg.d_model, stage0_cfg.n_head) == (8, 384, 6)
    assert stage0_cfg.qk_head_dim == 64


def test_grown_depth_doubles_layers(stage0_cfg):
    g = stage0_cfg.grown_depth()
    assert g.n_layer == 16 and g.d_model == 384


def test_grown_width_doubles_width_only(stage0_cfg):
    g = stage0_cfg.grown_width()
    assert (g.d_model, g.n_head, g.kv_lora_rank, g.ffn_hidden) == (768, 12, 256, 2048)
    assert (g.n_layer, g.qk_nope_head_dim, g.qk_rope_head_dim, g.v_head_dim) == (8, 48, 16, 64)


def test_yaml_round_trip(tmp_path, stage2_cfg):
    p = tmp_path / "c.yaml"
    stage2_cfg.to_yaml(p)
    assert ModelConfig.from_yaml(p) == stage2_cfg


def test_invalid_rope_dim_raises():
    with pytest.raises(ValueError):
        ModelConfig(qk_rope_head_dim=15)
