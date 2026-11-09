"""环境检查：训练必须跑在支持 Blackwell 的 CUDA 版 PyTorch 上。"""

import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_blackwell_capability():
    major, _ = torch.cuda.get_device_capability(0)
    assert major == 12


def test_bf16_supported():
    assert torch.cuda.is_bf16_supported()
