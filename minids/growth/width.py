"""加宽算子：HyperCloning。

约定增长后残差流里的每个向量是原向量的两份拼接 [x; x]。在这个约定下：
- 输入输出都翻倍的线性层：W' = 1/2 · [[W, W], [W, W]]，输入 [x; x] 时输出 [Wx; Wx]
- 只有输入翻倍的层（wkpe、lm_head）：W' = 1/2 · [W, W]，输出不变
- 词嵌入按列复制 [E, E]（不乘 1/2），让残差流起点就是 [x; x]
- RMSNorm 的缩放参数复制一份拼接；rms([x; x]) = rms(x)
头数翻倍后新头与原头两两相同，注意力分数不变。RoPE 只作用于每头内部的维度，不受影响。

复制出来的两半完全对称，训练中会一直保持相同，所以最后给"第二份输出"
加一点相对幅度 noise_std_rel 的高斯噪声打破对称。
"""

import torch
from torch import Tensor

from minids.model.transformer import Transformer


def _both(w: Tensor) -> Tensor:
    """输入维和输出维都翻倍。nn.Linear 的 weight 形状为 (out, in)。"""
    wide = torch.cat([w, w], dim=1)
    return 0.5 * torch.cat([wide, wide], dim=0)


def _in_only(w: Tensor) -> Tensor:
    return 0.5 * torch.cat([w, w], dim=1)


def _vec(g: Tensor) -> Tensor:
    return torch.cat([g, g])


def _noise_like(p: Tensor, std: Tensor, generator: torch.Generator | None) -> Tensor:
    device = generator.device if generator is not None else p.device
    noise = torch.randn(p.shape, generator=generator, device=device, dtype=torch.float32)
    return (noise * std.float()).to(device=p.device, dtype=p.dtype)


def grow_width(
    model: Transformer,
    noise_std_rel: float = 1e-3,
    generator: torch.Generator | None = None,
) -> Transformer:
    new_cfg = model.cfg.grown_width()
    ref = next(model.parameters())
    new = Transformer(new_cfg).to(device=ref.device, dtype=ref.dtype)

    with torch.no_grad():
        emb = model.tok_emb.weight
        new.tok_emb.weight.copy_(torch.cat([emb, emb], dim=1))
        new.final_norm.weight.copy_(_vec(model.final_norm.weight))
        new.lm_head.weight.copy_(_in_only(model.lm_head.weight))

        for ob, nb in zip(model.blocks, new.blocks):
            nb.attn_norm.weight.copy_(_vec(ob.attn_norm.weight))
            nb.mlp_norm.weight.copy_(_vec(ob.mlp_norm.weight))

            oa, na = ob.attn, nb.attn
            na.wq.weight.copy_(_both(oa.wq.weight))
            na.wdkv.weight.copy_(_both(oa.wdkv.weight))
            na.kv_norm.weight.copy_(_vec(oa.kv_norm.weight))
            na.wukv.weight.copy_(_both(oa.wukv.weight))
            na.wkpe.weight.copy_(_in_only(oa.wkpe.weight))
            na.wo.weight.copy_(_both(oa.wo.weight))

            nb.mlp.w_gate.weight.copy_(_both(ob.mlp.w_gate.weight))
            nb.mlp.w_up.weight.copy_(_both(ob.mlp.w_up.weight))
            nb.mlp.w_down.weight.copy_(_both(ob.mlp.w_down.weight))

        if noise_std_rel > 0:
            _break_symmetry(new, noise_std_rel, generator)

    new.train(model.training)
    return new


def _break_symmetry(new: Transformer, noise_std_rel: float, generator: torch.Generator | None) -> None:
    """给每个输出维翻倍的矩阵的后半行、词嵌入的后半列加噪。只复制输入的层和 norm 不加。"""
    emb = new.tok_emb.weight
    half = emb.shape[1] // 2
    emb[:, half:] += _noise_like(emb[:, half:], noise_std_rel * emb.std(), generator)

    for nb in new.blocks:
        for p in (
            nb.attn.wq.weight,
            nb.attn.wdkv.weight,
            nb.attn.wukv.weight,
            nb.attn.wo.weight,
            nb.mlp.w_gate.weight,
            nb.mlp.w_up.weight,
            nb.mlp.w_down.weight,
        ):
            rows = p.shape[0] // 2
            p[rows:] += _noise_like(p[rows:], noise_std_rel * p.std(), generator)
