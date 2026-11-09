import torch

from minids.growth.width import break_symmetry, grow_width
from minids.model.transformer import Transformer


def test_break_symmetry_matches_fused_path(tiny_cfg):
    # 计划三要按 spec 顺序"先精确验证再加噪"，所以加噪必须能单独调用，且与一步到位的结果一致
    torch.manual_seed(0)
    old = Transformer(tiny_cfg)
    staged = grow_width(old, noise_std_rel=0.0)
    break_symmetry(staged, 1e-3, torch.Generator().manual_seed(7))
    fused = grow_width(old, noise_std_rel=1e-3, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(a, b) for a, b in zip(staged.state_dict().values(), fused.state_dict().values()))


def test_shapes_doubled(tiny_cfg):
    new = grow_width(Transformer(tiny_cfg), noise_std_rel=0.0)
    c = new.cfg
    assert (c.d_model, c.n_head, c.kv_lora_rank, c.ffn_hidden) == (128, 4, 64, 256)
    assert new.blocks[0].attn.wkpe.weight.shape == (8, 128)
    assert new.lm_head.weight.shape == (256, 128)


def test_function_preserved_without_noise(tiny_cfg):
    torch.manual_seed(0)
    old = Transformer(tiny_cfg).eval()
    new = grow_width(old, noise_std_rel=0.0).eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        a, b = old(idx)[0], new(idx)[0]
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_noise_keeps_relative_error_within_tolerance(tiny_cfg):
    torch.manual_seed(0)
    old = Transformer(tiny_cfg).eval()
    g = torch.Generator().manual_seed(1)
    new = grow_width(old, noise_std_rel=1e-3, generator=g).eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        a, b = old(idx)[0], new(idx)[0]
    assert ((a - b).abs().max() / a.abs().max()).item() <= 1e-2


def test_noise_breaks_symmetry(tiny_cfg):
    g = torch.Generator().manual_seed(1)
    new = grow_width(Transformer(tiny_cfg), noise_std_rel=1e-3, generator=g)
    W = new.blocks[0].mlp.w_gate.weight
    h = W.shape[0] // 2
    assert not torch.equal(W[:h], W[h:])
    E = new.tok_emb.weight
    d = E.shape[1] // 2
    assert not torch.equal(E[:, :d], E[:, d:])


def test_old_model_untouched(tiny_cfg):
    old = Transformer(tiny_cfg)
    before = {k: v.clone() for k, v in old.state_dict().items()}
    grow_width(old)
    assert all(torch.equal(before[k], v) for k, v in old.state_dict().items())
