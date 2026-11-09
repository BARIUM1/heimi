import torch

from minids.model.norm import RMSNorm


def test_output_rms_is_one():
    norm = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 3
    y = norm(x)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4, 10), atol=1e-4)


def test_weight_scales_output():
    norm = RMSNorm(8)
    norm.weight.data.fill_(2.0)
    x = torch.randn(2, 8)
    assert torch.allclose(norm(x), 2.0 * RMSNorm(8)(x))


def test_zero_input_is_finite():
    assert torch.isfinite(RMSNorm(8)(torch.zeros(2, 8))).all()


def test_preserves_dtype():
    assert RMSNorm(8)(torch.randn(2, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16
