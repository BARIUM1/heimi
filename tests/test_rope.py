import pytest
import torch

from minids.model.rope import apply_rope, precompute_rope


def test_shapes():
    cos, sin = precompute_rope(16, 32)
    assert cos.shape == sin.shape == (32, 16)


def test_relative_position_property():
    dim, T = 16, 40
    cos, sin = precompute_rope(dim, T)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    def score(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    # 只依赖相对位置 m - n
    assert torch.allclose(score(3, 1), score(13, 11), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)


def test_position_zero_is_identity():
    cos, sin = precompute_rope(8, 4)
    x = torch.randn(1, 2, 1, 8)
    assert torch.allclose(apply_rope(x, cos[0:1], sin[0:1]), x)


def test_odd_dim_raises():
    with pytest.raises(ValueError):
        precompute_rope(7, 4)
