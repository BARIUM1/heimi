import json

import torch
import torch.nn.functional as F

from minids.eval.val_loss import fixed_val_windows, val_loss
from minids.model.transformer import Transformer
from tests.helpers_data import write_shard


def _index(tmp_path, tokens):
    write_shard(tmp_path / "web" / "val_0000.bin", tokens)
    idx = {"vocab_size": 256, "dtype": "uint16", "eot_id": 0,
           "sources": {"web": {"train": [], "val": [{"file": "val_0000.bin", "tokens": len(tokens)}]}}}
    p = tmp_path / "index.json"
    p.write_text(json.dumps(idx), encoding="utf-8")
    return p


def test_fixed_windows_are_stable_and_shaped(tmp_path):
    p = _index(tmp_path, list(range(16, 16 + 200)))
    w1, w2 = fixed_val_windows(p, "web", 3, 16), fixed_val_windows(p, "web", 3, 16)
    assert w1.shape == (3, 17) and w1.dtype == torch.int64 and torch.equal(w1, w2)
    assert w1[1, 0].item() == 16 + 17


def test_val_loss_matches_manual_mean_over_uneven_batches(tiny_cfg):
    torch.manual_seed(0)
    m = Transformer(tiny_cfg).train()
    windows = torch.randint(16, 256, (5, 9))
    got = val_loss(m, windows, "cpu", micro_batch=2)  # 5 个窗口分成 2+2+1
    assert m.training is True  # 评测后恢复原来的训练模式
    m.eval()
    logits, _ = m(windows[:, :-1])
    want = F.cross_entropy(logits.reshape(-1, 256), windows[:, 1:].reshape(-1)).item()
    assert abs(got - want) < 1e-5
