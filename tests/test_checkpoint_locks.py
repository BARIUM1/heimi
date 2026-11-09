"""检查点文件被别的程序占着时训练不该停：窗口的生成进程读检查点只占几秒，等它放开再写。"""

import os
import sys
import threading

import pytest

from minids.model.transformer import Transformer
from minids.train import checkpoint
from minids.train.checkpoint import CheckpointError, TrainState, load_checkpoint, save_checkpoint
from minids.train.loop import run_ladder
from tests.helpers_train import tiny_ladder


def _save(path, tiny_cfg, step=1):
    save_checkpoint(path, Transformer(tiny_cfg), None, TrainState("s0", step=step), {})


def test_save_waits_out_a_briefly_locked_target(tmp_path, tiny_cfg, monkeypatch):
    monkeypatch.setattr(checkpoint, "REPLACE_DELAY_S", 0.01)
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "另一个程序正在使用此文件", str(dst))
        return real(src, dst)

    monkeypatch.setattr(checkpoint.os, "replace", flaky)
    p = tmp_path / "s0_step1.pt"
    _save(p, tiny_cfg)
    assert calls["n"] == 3 and load_checkpoint(p, "cpu")["state"]["step"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_save_gives_a_named_error_if_the_target_stays_locked(tmp_path, tiny_cfg, monkeypatch):
    monkeypatch.setattr(checkpoint, "REPLACE_DELAY_S", 0.0)
    monkeypatch.setattr(checkpoint, "REPLACE_ATTEMPTS", 3)

    def locked(src, dst):
        raise PermissionError(13, "另一个程序正在使用此文件", str(dst))

    monkeypatch.setattr(checkpoint.os, "replace", locked)
    with pytest.raises(CheckpointError, match="s0_step1.pt"):
        _save(tmp_path / "s0_step1.pt", tiny_cfg)


@pytest.mark.skipif(sys.platform != "win32", reason="文件被占用时替换失败是 Windows 的行为")
def test_save_succeeds_once_a_real_reader_lets_go(tmp_path, tiny_cfg, monkeypatch):
    monkeypatch.setattr(checkpoint, "REPLACE_DELAY_S", 0.05)
    p = tmp_path / "s0_step1.pt"
    _save(p, tiny_cfg, step=1)
    reader = open(p, "rb")  # 像窗口的生成进程那样占着这个文件
    timer = threading.Timer(0.3, reader.close)
    timer.start()
    try:
        _save(p, tiny_cfg, step=2)  # 没有重试的话这里直接 PermissionError
    finally:
        timer.join()
        reader.close()
    assert load_checkpoint(p, "cpu")["state"]["step"] == 2


def test_resume_survives_a_briefly_locked_corrupt_checkpoint(tmp_path, monkeypatch):
    cfg = tiny_ladder(tmp_path, stage_steps=(4,), ckpt_interval=2)
    run = tmp_path / "run"
    run_ladder(cfg, run, "cpu", max_steps=2)
    (run / "ckpt" / "smoke0_step4.pt").write_bytes(b"truncated by a power cut")
    monkeypatch.setattr(checkpoint, "REPLACE_DELAY_S", 0.01)
    real = os.replace
    seen = {"n": 0}

    def flaky(src, dst):
        if str(dst).endswith(".corrupt"):
            seen["n"] += 1
            if seen["n"] == 1:
                raise PermissionError(13, "另一个程序正在使用此文件", str(dst))
        return real(src, dst)

    monkeypatch.setattr(checkpoint.os, "replace", flaky)
    run_ladder(cfg, run, "cpu")
    assert seen["n"] == 2 and (run / "ckpt" / "smoke0_final.pt").exists()
