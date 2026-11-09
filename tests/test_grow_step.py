from dataclasses import replace

import pytest
import torch

from minids.data.loader import MixConfig
from minids.model.transformer import Transformer
from minids.train.config import StageConfig
from minids.train.grow import grow_for_stage


def _stage(model_cfg, growth):
    return StageConfig("s", growth, model_cfg, 256, 1e-3, 0.1, 4, MixConfig(), noise_std_rel=1e-3, rel_tol=1e-2)


def test_depth_is_exact(tiny_cfg):
    prev = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (2, 16))
    new, out = grow_for_stage(prev, _stage(tiny_cfg.grown_depth(), "depth"), tokens, seed=0)
    assert new.cfg == tiny_cfg.grown_depth() and out.kind == "depth"
    assert out.exact_report["exact"] is True and out.noisy_report is None
    assert out.params_after == Transformer(tiny_cfg.grown_depth()).count_params()


def test_width_verifies_twice_and_breaks_symmetry(tiny_cfg):
    prev = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (2, 16))
    new, out = grow_for_stage(prev, _stage(tiny_cfg.grown_width(), "width"), tokens, seed=0)
    assert out.exact_report["max_rel_err"] <= 1e-4 and out.noisy_report["max_rel_err"] <= 1e-2
    half = new.blocks[0].mlp.w_up.weight.shape[0] // 2
    w = new.blocks[0].mlp.w_up.weight
    assert not torch.equal(w[:half], w[half:])  # 对称已打破


def test_none_returns_same_object_or_rejects_mismatch(tiny_cfg):
    prev = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (1, 8))
    same, out = grow_for_stage(prev, _stage(tiny_cfg, "none"), tokens, seed=0)
    assert same is prev and out.kind == "none" and out.exact_report is None
    with pytest.raises(ValueError):
        grow_for_stage(prev, _stage(replace(tiny_cfg, n_layer=3), "none"), tokens, seed=0)
