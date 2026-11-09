import torch

from minids.growth.depth import grow_depth
from minids.model.transformer import Transformer


def test_doubles_layers_and_params(tiny_cfg):
    old = Transformer(tiny_cfg)
    new = grow_depth(old)
    assert new.cfg.n_layer == 4 and len(new.blocks) == 4
    per_layer = sum(p.numel() for p in old.blocks[0].parameters())
    assert new.count_params() == old.count_params() + 2 * per_layer


def test_function_preserved_exactly(tiny_cfg):
    torch.manual_seed(0)
    old = Transformer(tiny_cfg).eval()
    new = grow_depth(old).eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        assert torch.equal(old(idx)[0], new(idx)[0])


def test_new_blocks_have_zero_residual_projections(tiny_cfg):
    new = grow_depth(Transformer(tiny_cfg))
    for j in (1, 3):
        assert not new.blocks[j].attn.wo.weight.any()
        assert not new.blocks[j].mlp.w_down.weight.any()
    assert new.blocks[0].attn.wo.weight.any()


def test_old_model_untouched(tiny_cfg):
    old = Transformer(tiny_cfg)
    before = {k: v.clone() for k, v in old.state_dict().items()}
    grow_depth(old)
    assert all(torch.equal(before[k], v) for k, v in old.state_dict().items())
