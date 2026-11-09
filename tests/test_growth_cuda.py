"""GPU 上的增长测试：CPU 单测看不到设备相关的问题，这里补上。"""

import pytest
import torch

from minids.growth.depth import grow_depth
from minids.growth.verify import verify_growth
from minids.growth.width import grow_width
from minids.model.transformer import Transformer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


@cuda
def test_width_growth_on_cuda_model_with_cpu_generator(tiny_cfg):
    # 计划三最自然的可复现调用：GPU 上的模型 + CPU 上按种子建的生成器
    torch.manual_seed(0)
    old = Transformer(tiny_cfg).cuda()
    new = grow_width(old, noise_std_rel=1e-3, generator=torch.Generator().manual_seed(1))
    tokens = torch.randint(0, 256, (2, 32), device="cuda")
    r = verify_growth(old, new, tokens)
    assert r.max_rel_err <= 1e-2
    assert next(new.parameters()).device.type == "cuda"


@cuda
def test_depth_growth_on_cuda_is_exact(tiny_cfg):
    old = Transformer(tiny_cfg).cuda()
    tokens = torch.randint(0, 256, (2, 32), device="cuda")
    assert verify_growth(old, grow_depth(old), tokens).exact
