import pytest
import torch

from minids.growth.depth import grow_depth
from minids.growth.verify import GrowthVerificationError, verify_growth
from minids.growth.width import grow_width
from minids.model.transformer import Transformer


def test_depth_growth_reports_exact(tiny_cfg):
    old = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (2, 32))
    r = verify_growth(old, grow_depth(old), tokens)
    assert r.exact and r.max_abs_diff == 0.0 and r.params_after > r.params_before


def test_width_growth_with_noise_passes(tiny_cfg):
    old = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (2, 32))
    r = verify_growth(old, grow_width(old, generator=torch.Generator().manual_seed(1)), tokens)
    assert r.max_rel_err <= 1e-2


def test_broken_growth_raises(tiny_cfg):
    old = Transformer(tiny_cfg)
    bad = grow_depth(old)
    bad.blocks[1].attn.wo.weight.data.normal_()  # 破坏恒等性
    with pytest.raises(GrowthVerificationError):
        verify_growth(old, bad, torch.randint(0, 256, (2, 32)))


def test_report_to_dict_keys(tiny_cfg):
    old = Transformer(tiny_cfg)
    d = verify_growth(old, grow_depth(old), torch.randint(0, 256, (1, 8))).to_dict()
    assert set(d) == {"params_before", "params_after", "max_abs_diff", "max_rel_err", "exact"}
