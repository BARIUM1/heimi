import pytest
from fastapi.testclient import TestClient

from dashboard.server import create_app
from dashboard.worker import WorkerClient, WorkerError, WorkerTimeout
from tests.helpers_dashboard import FakeWorker


@pytest.fixture
def client(dash_runs):
    worker = FakeWorker()
    with TestClient(create_app(dash_runs[0], worker=worker)) as c:
        c.worker = worker
        c.runs_dir = dash_runs[0]
        yield c


def test_runs_listing(client):
    r = client.get("/api/runs").json()
    assert r[0]["name"] == "heimi" and r[0]["checkpoints"] and r[0]["stages"][1]["growth"] == "depth"


def test_metrics_since_and_full_logs(client):
    assert [m["global_step"] for m in client.get("/api/runs/heimi/metrics").json()] == [1, 2, 3, 4]
    assert [m["global_step"] for m in client.get("/api/runs/heimi/metrics", params={"since": 2}).json()] == [3, 4]
    for name in ("eval", "events", "samples"):
        body = client.get(f"/api/runs/heimi/{name}").json()
        assert isinstance(body, list) and body
    o = client.get("/api/runs/heimi/overview").json()
    assert o["growth"][0]["kind"] == "depth" and o["growth"][0]["label"].endswith("加深") and len(o["params"]) >= 2


def test_unknown_or_traversing_run_is_404(client):
    for path in ("/api/runs/nope/metrics", "/api/runs/%2E%2E/metrics", "/api/runs/..%2F..%2Fx/events", "/api/runs/nope/overview"):
        assert client.get(path).status_code == 404, path


def test_status_shape(client):
    s = client.get("/api/status").json()
    assert s["run"] == "heimi" and s["state"] in ("running", "quiet", "stopped") and "gpu" in s and "eta_seconds" in s


def test_generate_hands_the_worker_a_validated_absolute_path(client):
    body = {"run": "heimi", "checkpoint": "smoke1_final.pt", "prefix": "The cat", "max_new_tokens": 8, "temperature": 0.5, "device": "cpu"}
    r = client.post("/api/generate", json=body)
    assert r.status_code == 200 and r.json()["text"] == " ok" and r.json()["checkpoint"] == "smoke1_final.pt" and r.json()["device"] == "cpu"
    req = client.worker.calls[0]
    assert req["ckpt_path"] == str(client.runs_dir / "heimi" / "ckpt" / "smoke1_final.pt")
    assert req["suffix"] is None and req["max_new_tokens"] == 8 and req["temperature"] == 0.5


def test_generate_rejects_bad_names_and_out_of_range_values(client):
    base = {"run": "heimi", "checkpoint": "smoke1_final.pt", "prefix": "x"}
    assert client.post("/api/generate", json={**base, "checkpoint": "../../outside.pt"}).status_code == 404
    assert client.post("/api/generate", json={**base, "run": ".."}).status_code == 404
    assert client.post("/api/generate", json={**base, "max_new_tokens": 9999}).status_code == 422
    assert client.post("/api/generate", json={**base, "device": "mps"}).status_code == 422
    assert client.post("/api/generate", json={**base, "prefix": ""}).status_code == 422
    assert client.worker.calls == []  # 不合法的请求一个都没到生成进程


def test_generate_timeout_is_504_and_worker_error_is_500(dash_runs):
    body = {"run": "heimi", "checkpoint": "smoke1_final.pt", "prefix": "x"}
    with TestClient(create_app(dash_runs[0], worker=FakeWorker(exc=WorkerTimeout("生成超过 60 秒")))) as c:
        r = c.post("/api/generate", json=body)
        assert r.status_code == 504 and "60" in r.json()["detail"]
    with TestClient(create_app(dash_runs[0], worker=FakeWorker(exc=WorkerError("RuntimeError: CUDA 不可用")))) as c:
        r = c.post("/api/generate", json=body)
        assert r.status_code == 500 and "CUDA" in r.json()["detail"]


def test_generate_end_to_end_with_real_worker(dash_runs):
    with TestClient(create_app(dash_runs[0], worker=WorkerClient())) as c:
        r = c.post("/api/generate", json={"run": "heimi", "checkpoint": "smoke0_final.pt", "prefix": "The cat", "max_new_tokens": 4, "temperature": 0.0})
        assert r.status_code == 200 and isinstance(r.json()["text"], str) and r.json()["seconds"] >= 0


def test_worker_is_closed_on_shutdown(dash_runs):
    worker = FakeWorker()
    with TestClient(create_app(dash_runs[0], worker=worker)):
        pass
    assert worker.closed
