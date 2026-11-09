"""整模型：词嵌入 → N 个 Block → 最终 RMSNorm → 输出头。

嵌入和输出头不共享权重。RoPE 的 cos/sin 表注册为非持久 buffer，
不进 state_dict，所以增长时只需要搬运参数。
增长算子依赖属性名 tok_emb、blocks、final_norm、lm_head 和 cfg。
"""

import math

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn

from minids.config import ModelConfig
from minids.model.block import Block
from minids.model.norm import RMSNorm
from minids.model.rope import precompute_rope


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差分支的输出投影按层数缩放，防止深层残差流方差随层数线性增长
        scaled_std = cfg.init_std / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=scaled_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx: LongTensor, targets: LongTensor | None = None) -> tuple[Tensor, Tensor | None]:
        _, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"sequence length {T} exceeds seq_len {self.cfg.seq_len}")
        if idx.numel() > 0:
            max_id = int(idx.max())
            if max_id >= self.cfg.vocab_size:
                raise ValueError(f"token id {max_id} is out of range for vocab_size {self.cfg.vocab_size}")

        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> LongTensor:
        """朴素生成：每步全量前向。temperature 为 0 时贪心。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def count_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.lm_head.weight.numel()
        return n
