import torch

from minids.model.mla import MLA
from minids.model.rope import precompute_rope


def _run(cfg, x):
    cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.seq_len)
    return MLA(cfg)(x, cos[: x.shape[1]], sin[: x.shape[1]])


def test_output_shape(tiny_cfg):
    x = torch.randn(2, 16, tiny_cfg.d_model)
    assert _run(tiny_cfg, x).shape == (2, 16, tiny_cfg.d_model)


def test_causal(tiny_cfg):
    torch.manual_seed(0)
    m = MLA(tiny_cfg)
    cos, sin = precompute_rope(tiny_cfg.qk_rope_head_dim, tiny_cfg.seq_len)
    x = torch.randn(1, 12, tiny_cfg.d_model)
    x2 = x.clone()
    x2[:, 8:] += 5.0
    y, y2 = m(x, cos[:12], sin[:12]), m(x2, cos[:12], sin[:12])
    # 改动位置 8 之后的输入，不能影响位置 8 之前的输出
    assert torch.allclose(y[:, :8], y2[:, :8], atol=1e-5)
    assert not torch.allclose(y[:, 8:], y2[:, 8:], atol=1e-3)


def test_parameter_shapes(stage0_cfg):
    m = MLA(stage0_cfg)
    assert m.wq.weight.shape == (384, 384)
    assert m.wdkv.weight.shape == (128, 384)
    assert m.wukv.weight.shape == (672, 128)
    assert m.wkpe.weight.shape == (16, 384)
    assert m.wo.weight.shape == (384, 384)
    assert all(mod.bias is None for mod in [m.wq, m.wdkv, m.wukv, m.wkpe, m.wo])


def test_backward_flows(tiny_cfg):
    x = torch.randn(2, 8, tiny_cfg.d_model, requires_grad=True)
    _run(tiny_cfg, x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
