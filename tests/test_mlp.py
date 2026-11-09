import torch

from minids.model.mlp import SwiGLU


def test_shape():
    assert SwiGLU(64, 128)(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_zero_down_proj_gives_zero_output():
    m = SwiGLU(64, 128)
    m.w_down.weight.data.zero_()
    assert torch.equal(m(torch.randn(2, 5, 64)), torch.zeros(2, 5, 64))


def test_no_bias():
    m = SwiGLU(8, 16)
    assert m.w_gate.bias is None and m.w_up.bias is None and m.w_down.bias is None
