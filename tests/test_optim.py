from minids.model.transformer import Transformer
from minids.train.optim import build_optimizer


def test_groups_partition_params_and_exclude_embedding_and_norms(tiny_cfg):
    m = Transformer(tiny_cfg)
    opt = build_optimizer(m, lr=1e-3)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))
    ids_no = {id(p) for p in no_decay["params"]}
    assert id(m.tok_emb.weight) in ids_no and id(m.final_norm.weight) in ids_no and id(m.blocks[0].attn.kv_norm.weight) in ids_no
    assert id(m.lm_head.weight) not in ids_no and id(m.blocks[0].attn.wq.weight) not in ids_no
    assert opt.defaults["betas"] == (0.9, 0.95)
