import torch

from minids.model.block import Block
from minids.model.rope import precompute_rope


def test_shape(tiny_cfg):
    cos, sin = precompute_rope(tiny_cfg.qk_rope_head_dim, 16)
    x = torch.randn(2, 16, tiny_cfg.d_model)
    assert Block(tiny_cfg)(x, cos, sin).shape == x.shape


def test_identity_when_residual_projections_are_zero(tiny_cfg):
    # 加深算子依赖这个性质：输出投影置零的块就是恒等映射
    b = Block(tiny_cfg)
    b.attn.wo.weight.data.zero_()
    b.mlp.w_down.weight.data.zero_()
    cos, sin = precompute_rope(tiny_cfg.qk_rope_head_dim, 16)
    x = torch.randn(2, 16, tiny_cfg.d_model)
    assert torch.equal(b(x, cos, sin), x)
