"""窗口在不理想情况下的表现：中断重跑留下的重复记录、坏掉的日志、事后重算的评测、生成进程的显存与重启。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard import data, worker
from dashboard.data import list_runs, metrics_since, overview, status
from dashboard.server import create_app
from dashboard.worker import ModelCache, WorkerClient, WorkerTimeout, worker_main
from minids.logging_io import METRICS_FILE, EvalRecord, MetricsRecord, RunLogger, now_iso
from tests.helpers_dashboard import FakeWorker, dying_worker, flaky_worker


def _metric(gs, loss, stage="s0"):
    return MetricsRecord(stage, gs, gs, gs * 256, gs * 256, loss, 1e-3, 0.5, 1000.0, 8.0, 55.0, now_iso())


def _stage_start(log, stage, gs, params, growth):
    log.event("stage_start", stage, gs, gs * 256, params_non_embedding=params, steps=3, accum_steps=2, micro_batch=4, growth=growth)


# ---------- 中断重跑留下的重复记录 ----------

def test_restarted_stage_leaves_one_growth_mark_and_one_staircase_step(tmp_path):
    log = RunLogger(tmp_path)
    _stage_start(log, "s0", 0, 75_136, "none")
    for _ in range(2):  # 进入 s1 两次：第一次在头 500 步内被打断，没有检查点，重跑同一命令又增长了一遍
        log.event("growth", "s1", 3, 768, kind="depth", params_before=75_136, params_after=150_208,
                  exact_report={"exact": True}, noisy_report=None, seconds=0.1)
        _stage_start(log, "s1", 3, 150_208, "depth")
    o = overview(tmp_path)
    assert len(o["growth"]) == 1 and o["growth"][0]["label"] == "75K → 150K 加深"
    assert [(p["tokens_total"], p["params"]) for p in o["params"]] == [(0, 75_136), (768, 150_208)]


def test_metrics_drop_lines_superseded_by_a_rewind(tmp_path):
    log = RunLogger(tmp_path)
    # 写到第 70 步后从第 50 步的检查点恢复：60、70 又写了一遍
    for gs, loss in ((50, 5.0), (60, 4.9), (70, 4.8), (60, 4.95), (70, 4.85), (80, 4.7)):
        log.metrics(_metric(gs, loss))
    rows = metrics_since(tmp_path)
    assert [(r["global_step"], r["loss"]) for r in rows] == [(50, 5.0), (60, 4.95), (70, 4.85), (80, 4.7)]
    assert [r["global_step"] for r in metrics_since(tmp_path, 60)] == [70, 80]
    with TestClient(create_app(tmp_path.parent, worker=FakeWorker())) as c:
        got = c.get(f"/api/runs/{tmp_path.name}/metrics").json()
    assert [r["global_step"] for r in got] == [50, 60, 70, 80]


# ---------- 坏掉的日志 ----------

def _torn_run(run_dir: Path) -> None:
    log = RunLogger(run_dir)
    log.metrics(_metric(10, 5.0))
    with log.path(METRICS_FILE).open("a", encoding="utf-8") as f:
        f.write('{"stage": "s0", "glo\n')  # 断电时写了半行
    log.metrics(_metric(20, 4.9))  # 之后又接着写：坏行落在了中间


def test_one_unreadable_log_does_not_take_down_the_other_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "gpu_status", lambda: None)
    RunLogger(tmp_path / "heimi").metrics(_metric(10, 5.0))
    _torn_run(tmp_path / "broken")
    runs = {r["name"]: r for r in list_runs(tmp_path)}
    assert set(runs) == {"heimi", "broken"}
    assert runs["heimi"]["global_step"] == 10 and not runs["heimi"].get("error")
    assert "metrics.jsonl" in runs["broken"]["error"]
    assert status(tmp_path, run="heimi")["step"] == 10
    assert status(tmp_path)["run"] == "heimi"  # 默认挑最近更新的、读得出来的那个
    bad = status(tmp_path, run="broken")
    assert bad["run"] == "broken" and "metrics.jsonl" in bad["error"]


def test_unreadable_log_is_a_clear_error_for_that_run_only(tmp_path):
    RunLogger(tmp_path / "heimi").metrics(_metric(10, 5.0))
    _torn_run(tmp_path / "broken")
    with TestClient(create_app(tmp_path, worker=FakeWorker())) as c:
        assert c.get("/api/runs").status_code == 200
        assert c.get("/api/status").status_code == 200
        assert c.get("/api/runs/heimi/metrics").status_code == 200
        r = c.get("/api/runs/broken/metrics")
        assert r.status_code == 500 and "metrics.jsonl" in r.json()["detail"]


# ---------- 事后重算的评测 ----------

def test_rerun_eval_merges_instead_of_erasing_and_status_reports_the_latest_point(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "gpu_status", lambda: None)
    log = RunLogger(tmp_path / "heimi")
    log.metrics(_metric(10, 4.0, stage="stage1"))
    log.eval(EvalRecord("stage0", 3, 768, 5.0, 4.0, 0.25, 0.0, now_iso()))    # 级末评测：全套数字
    log.eval(EvalRecord("stage1", 9, 2304, 4.5, 3.5, None, None, now_iso()))  # 之后的例行评测
    log.eval(EvalRecord("stage0", 3, 768, 5.01, 4.01, None, 0.0, now_iso()))  # 事后对旧检查点跑 evaluate.py --no-hellaswag
    first = overview(tmp_path / "heimi")["eval"][0]
    assert (first["global_step"], first["val_loss_web"], first["hellaswag_acc"]) == (3, 5.01, 0.25)  # 新值覆盖旧值，空值不抹掉已有的
    assert status(tmp_path, run="heimi")["last_eval"]["global_step"] == 9  # "最近一次评测"按进度算，不按写入先后


# ---------- 生成进程：显存与重启 ----------

class _StubModel:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_checkpoints_are_loaded_on_cpu_and_only_the_model_moves_to_the_gpu(dash_runs, monkeypatch):
    # 检查点里还有优化器状态（约两倍于模型）：直接按 cuda 加载会把它也搬上显卡
    runs_dir, cfg = dash_runs
    seen = {}
    stub = _StubModel()

    def fake_load(path, device):
        seen["device"] = device
        return stub, {"ladder_config": {"data": {"tokenizer": cfg.data.tokenizer}}}

    monkeypatch.setattr(worker, "load_model", fake_load)
    model, _ = ModelCache().get(str(runs_dir / "heimi" / "ckpt" / "smoke0_final.pt"), "cuda")
    assert seen["device"] == "cpu" and model is stub and stub.moved_to == "cuda"


class _ScriptedConn:
    """假的管道：poll 按脚本回答，recv 永远给 None（请它退出）。"""

    def __init__(self, polls):
        self.polls = list(polls)
        self.poll_calls = 0

    def poll(self, timeout=None):
        self.poll_calls += 1
        return self.polls.pop(0)

    def recv(self):
        return None

    def send(self, obj):
        raise AssertionError("不该有回话")


def test_idle_worker_exits_only_if_it_holds_gpu_memory(monkeypatch):
    # 用过 GPU 的进程空闲下来就整个退出，把显存连同 CUDA 上下文还给训练；只用 CPU 的留着，下次不用重新加载
    monkeypatch.setattr(ModelCache, "uses_gpu", property(lambda self: True))
    conn = _ScriptedConn([False])
    worker_main(conn, None, idle_exit_s=0.01)
    assert conn.poll_calls == 1  # 第一次空闲就退出了，没有等到下一轮

    monkeypatch.setattr(ModelCache, "uses_gpu", property(lambda self: False))
    conn = _ScriptedConn([False, False, True])
    worker_main(conn, None, idle_exit_s=0.01)
    assert conn.poll_calls == 3  # 空闲两轮都没退，直到收到 None


def test_request_after_a_timeout_is_served_by_a_fresh_worker(tmp_path):
    client = WorkerClient(timeout_s=1.0, repo_root=tmp_path, target=flaky_worker)
    try:
        with pytest.raises(WorkerTimeout):
            client.generate({"prefix": "first"})
        assert client.generate({"prefix": "second"})["text"] == "second"  # 卡住的那个已被杀掉，新进程接手
    finally:
        client.close()


def test_request_that_meets_an_exiting_worker_is_retried_once(tmp_path):
    client = WorkerClient(timeout_s=20.0, repo_root=tmp_path, target=dying_worker)
    try:
        assert client.generate({"prefix": "hello"})["text"] == "hello"
    finally:
        client.close()


# ---------- 页面 ----------

def test_page_warns_before_using_the_gpu_while_a_run_is_training():
    html = Path("dashboard/static/index.html").read_text(encoding="utf-8")
    assert 'id="pg-gpu-warning"' in html and "抢显存" in html
