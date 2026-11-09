"""SwiGLU 前馈网络：down(silu(gate(x)) * up(x))。

增长算子依赖属性名 w_gate、w_up、w_down。
"""

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
