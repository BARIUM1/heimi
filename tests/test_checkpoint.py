import pytest
import torch

from minids.model.transformer import Transformer
from minids.train.checkpoint import (
    CheckpointError,
    TrainState,
    checkpoint_name,
    latest_step_checkpoint,
    load_checkpoint,
    load_model,
    prune_step_checkpoints,
    save_checkpoint,
)


def _trained(tiny_cfg):
    m = Transformer(tiny_cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (2, 16))
    m(x, x)[1].backward()
    opt.step()
    return m, opt


def test_round_trip_weights_optimizer_state(tmp_path, tiny_cfg):
    m, opt = _trained(tiny_cfg)
    state = TrainState("s0", step=3, global_step=7, micro_step=14, tokens_stage=300, tokens_total=700)
    p = tmp_path / checkpoint_name("s0", 3)
    save_checkpoint(p, m, opt, state, {"seed": 1})
    assert p.name == "s0_step3.pt" and not list(tmp_path.glob("*.tmp"))
    payload = load_checkpoint(p, "cpu")
    m2 = Transformer(tiny_cfg)
    m2.load_state_dict(payload["model"])
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(payload["optimizer"])
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
    assert TrainState.from_dict(payload["state"]) == state
    assert payload["ladder_config"] == {"seed": 1} and payload["model_config"] == tiny_cfg.to_dict()
    assert "git_commit" in payload and "saved_at" in payload


def test_load_model_rebuilds_from_config(tmp_path, tiny_cfg):
    m, _ = _trained(tiny_cfg)
    p = tmp_path / checkpoint_name("s0", None)
    save_checkpoint(p, m, None, TrainState("s0"), {})
    m2, meta = load_model(p, "cpu")
    x = torch.randint(0, 256, (1, 8))
    assert torch.equal(m.eval()(x)[0], m2(x)[0]) and not m2.training
    assert "model" not in meta and "optimizer" not in meta and meta["state"]["stage"] == "s0"


def test_corrupt_file_raises_named_error(tmp_path):
    p = tmp_path / "s0_step1.pt"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="s0_step1.pt"):
        load_checkpoint(p, "cpu")


def test_latest_is_numeric_not_lexical(tmp_path):
    for n in (300, 1200, 50):
        (tmp_path / f"s0_step{n}.pt").write_bytes(b"x")
    (tmp_path / "s1_step9999.pt").write_bytes(b"x")
    (tmp_path / "s0_final.pt").write_bytes(b"x")
    assert latest_step_checkpoint(tmp_path, "s0").name == "s0_step1200.pt"
    assert latest_step_checkpoint(tmp_path, "s2") is None


def test_prune_survives_a_file_locked_by_another_process(tmp_path, monkeypatch):
    # Windows 上别的进程（窗口的生成工作进程）正在读最旧的检查点时 unlink 会 PermissionError，训练不能因此死掉
    from pathlib import Path

    for n in (100, 200, 300):
        (tmp_path / f"s0_step{n}.pt").write_bytes(b"x")
    real_unlink = Path.unlink

    def locked_unlink(self, *a, **k):
        if self.name == "s0_step100.pt":
            raise PermissionError(32, "另一个程序正在使用此文件")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    deleted = prune_step_checkpoints(tmp_path, "s0", keep=1)
    assert [p.name for p in deleted] == ["s0_step200.pt"]
    assert (tmp_path / "s0_step100.pt").exists() and (tmp_path / "s0_step300.pt").exists()


def test_prune_keeps_newest_and_final_and_other_stages(tmp_path):
    for n in (100, 200, 300, 400):
        (tmp_path / f"s0_step{n}.pt").write_bytes(b"x")
    (tmp_path / "s0_final.pt").write_bytes(b"x")
    (tmp_path / "s1_step50.pt").write_bytes(b"x")
    deleted = prune_step_checkpoints(tmp_path, "s0", keep=2)
    assert sorted(p.name for p in deleted) == ["s0_step100.pt", "s0_step200.pt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s0_final.pt", "s0_step300.pt", "s0_step400.pt", "s1_step50.pt"]
