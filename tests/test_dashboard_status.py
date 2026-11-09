from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from dashboard import data
from dashboard.data import eta_seconds, status, training_state
from minids.logging_io import EVENTS_FILE, EvalRecord, EventRecord, MetricsRecord, RunLogger, append_jsonl

# 所有记录的时间都由测试给定，不用 RunLogger.event（它盖的是真实的当前时间）
T0 = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone(timedelta(hours=-5)))


def _metric(stage, step, tok_per_s, when, tokens_per_step=256):
    return MetricsRecord(stage, step, step, step * tokens_per_step, step * tokens_per_step, 4.0, 1e-3, 0.5,
                         tok_per_s, 8.0, 55.0, when.isoformat(timespec="seconds"))


def test_training_state_thresholds():
    assert training_state(T0, T0 + timedelta(seconds=60)) == "running"
    assert training_state(T0, T0 + timedelta(seconds=600)) == "quiet"  # 级末评测期间
    assert training_state(T0, T0 + timedelta(seconds=3600)) == "stopped"
    assert training_state(None, T0) == "stopped"


def test_eta_uses_median_throughput_and_remaining_steps():
    events = [{"type": "stage_start", "stage": "s0", "data": {"steps": 100}}]
    metrics = [asdict(_metric("s0", s, tps, T0)) for s, tps in ((10, 1000.0), (20, 1000.0), (30, 10.0), (40, 1000.0))]
    assert eta_seconds(metrics, events) == pytest.approx(60 * 256 / 1000.0)  # 第 30 步那条被评测拖慢，不影响中位数
    assert eta_seconds([], events) is None and eta_seconds(metrics, []) is None


def test_eta_is_zero_when_stage_finished():
    events = [{"type": "stage_start", "stage": "s0", "data": {"steps": 40}}]
    assert eta_seconds([asdict(_metric("s0", 40, 1000.0, T0))], events) == 0.0


def test_status_picks_latest_run_and_reports_state(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "gpu_status", lambda: {"temp_c": 50.0, "mem_used_gb": 3.0, "mem_total_gb": 16.0, "util_pct": 5.0})
    old, new = RunLogger(tmp_path / "old"), RunLogger(tmp_path / "heimi")
    old.metrics(_metric("s0", 10, 500.0, T0 - timedelta(hours=5)))
    append_jsonl(new.path(EVENTS_FILE), EventRecord(
        "stage_start", "stage1", 0, 0, (T0 - timedelta(minutes=5)).isoformat(timespec="seconds"),
        {"params_non_embedding": 1, "steps": 100, "accum_steps": 2, "micro_batch": 4, "growth": "none"}))
    new.metrics(_metric("stage1", 10, 1000.0, T0))
    new.eval(EvalRecord("stage1", 10, 2560, 4.2, 3.3, None, None, T0.isoformat(timespec="seconds")))
    s = status(tmp_path, now=T0 + timedelta(seconds=30))
    assert s["run"] == "heimi" and s["state"] == "running" and s["running"] is True
    assert (s["stage"], s["step"], s["steps"], s["tok_per_s"], s["loss"]) == ("stage1", 10, 100, 1000.0, 4.0)
    assert s["eta_seconds"] == pytest.approx(90 * 256 / 1000.0) and s["last_eval"]["val_loss_web"] == 4.2
    assert s["gpu"]["temp_c"] == 50.0 and s["seconds_since_update"] == pytest.approx(30.0, abs=1.0)
    later = status(tmp_path, now=T0 + timedelta(hours=2))
    assert later["state"] == "stopped" and later["running"] is False


def test_status_with_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "gpu_status", lambda: None)
    s = status(tmp_path / "missing")
    assert s["run"] is None and s["state"] == "stopped" and s["running"] is False
    assert s["gpu"] is None and s["eta_seconds"] is None and s["stage"] is None and s["last_eval"] is None


def test_status_can_follow_a_chosen_run_and_still_names_the_active_one(tmp_path, monkeypatch):
    # 页面正在看 old 这次训练时，状态栏报的应该是 old，而不是别的更晚动过的 run；同时告诉页面谁正在训练
    monkeypatch.setattr(data, "gpu_status", lambda: None)
    old, new = RunLogger(tmp_path / "old"), RunLogger(tmp_path / "heimi")
    old.metrics(_metric("s0", 10, 500.0, T0 - timedelta(hours=5)))
    new.metrics(_metric("stage1", 7, 1000.0, T0))
    now = T0 + timedelta(seconds=30)
    s = status(tmp_path, now=now, run="old")
    assert (s["run"], s["state"], s["step"]) == ("old", "stopped", 10)
    assert (s["active_run"], s["active_state"]) == ("heimi", "running")
    default = status(tmp_path, now=now)
    assert (default["run"], default["active_run"]) == ("heimi", "heimi")
    with pytest.raises(data.RunNotFound):
        status(tmp_path, now=now, run="..")
