import pytest
import torch

from minids.model.transformer import Transformer


def test_forward_shapes_and_loss(tiny_cfg):
    m = Transformer(tiny_cfg)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, 256) and loss.ndim == 0 and torch.isfinite(loss)


def test_param_counts_match_spec(stage0_cfg, stage2_cfg):
    # 精确到个位：这是对结构是否与 spec 5.3 一致的硬检查
    assert Transformer(stage0_cfg).count_params() == 12_934_528
    assert Transformer(stage0_cfg).count_params(non_embedding=False) == 12_934_528 + 2 * 16384 * 384
    assert Transformer(stage2_cfg).count_params() == 103_247_872 + 768


def test_residual_proj_init_is_scaled(stage0_cfg):
    torch.manual_seed(0)
    m = Transformer(stage0_cfg)
    assert abs(m.blocks[0].mlp.w_up.weight.std().item() - 0.02) < 0.002
    # 残差输出投影按 1/sqrt(2 * n_layer) 缩放，sqrt(2*8) = 4
    assert abs(m.blocks[0].mlp.w_down.weight.std().item() - 0.02 / 4) < 0.001


def test_too_long_sequence_raises(tiny_cfg):
    with pytest.raises(ValueError, match="exceeds"):
        Transformer(tiny_cfg)(torch.zeros(1, 65, dtype=torch.long))


def test_out_of_range_token_raises(tiny_cfg):
    with pytest.raises(ValueError):
        Transformer(tiny_cfg)(torch.tensor([[0, 256]]))


def test_generate_length_and_greedy(tiny_cfg):
    m = Transformer(tiny_cfg)
    idx = torch.randint(0, 256, (1, 5))
    out = m.generate(idx, 7, temperature=0.0)
    assert out.shape == (1, 12) and torch.equal(out[:, :5], idx)
    # 贪心解码可复现
    assert torch.equal(out, m.generate(idx, 7, temperature=0.0))


def test_generate_crops_long_prompt(tiny_cfg):
    m = Transformer(tiny_cfg)
    idx = torch.randint(0, 256, (1, 100))
    assert m.generate(idx, 3, temperature=0.0).shape == (1, 103)
