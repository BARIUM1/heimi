"""一个 pre-norm Transformer 块：x = x + Attn(Norm(x))；x = x + MLP(Norm(x))。

增长算子依赖属性名 attn_norm、attn、mlp_norm、mlp。
"""

from torch import Tensor, nn

from minids.config import ModelConfig
from minids.model.mla import MLA
from minids.model.mlp import SwiGLU
from minids.model.norm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x
