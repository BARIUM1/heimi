"""加深算子：块扩展（LLaMA Pro 的做法）。

在每个原有块后面插入一个新块。新块从原块深拷贝权重，但把两个残差输出投影
（注意力的 wo 和前馈的 w_down）置零，于是新块的残差分支输出恒为零，
整个新模型与旧模型逐位等价。训练开始后这两个投影从零慢慢学出来。
"""

import torch

from minids.model.transformer import Transformer


def grow_depth(model: Transformer) -> Transformer:
    new_cfg = model.cfg.grown_depth()
    ref = next(model.parameters())
    new = Transformer(new_cfg).to(device=ref.device, dtype=ref.dtype)

    with torch.no_grad():
        new.tok_emb.load_state_dict(model.tok_emb.state_dict())
        new.final_norm.load_state_dict(model.final_norm.state_dict())
        new.lm_head.load_state_dict(model.lm_head.state_dict())
        for i, old_block in enumerate(model.blocks):
            sd = old_block.state_dict()
            new.blocks[2 * i].load_state_dict(sd)
            inserted = new.blocks[2 * i + 1]
            inserted.load_state_dict(sd)
            inserted.attn.wo.weight.zero_()
            inserted.mlp.w_down.weight.zero_()

    new.train(model.training)
    return new
