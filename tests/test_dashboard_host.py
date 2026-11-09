"""窗口只认以本机名字访问的请求：别的 Host（DNS 重绑定的网页会带自己的域名）一律拒绝。"""

from fastapi.testclient import TestClient

from dashboard.server import create_app
from tests.helpers_dashboard import FakeWorker


def test_requests_with_a_foreign_host_header_are_refused(dash_runs):
    worker = FakeWorker()
    app = create_app(dash_runs[0], worker=worker)
    with TestClient(app, base_url="http://evil.example") as c:
        assert c.get("/api/runs").status_code == 400
        assert c.get("/").status_code == 400
        body = {"run": "heimi", "checkpoint": "smoke0_final.pt", "prefix": "x"}
        assert c.post("/api/generate", json=body).status_code == 400
    assert worker.calls == []  # 外来请求一个都没到生成进程


def test_local_names_are_accepted_on_any_port(dash_runs):
    app = create_app(dash_runs[0], worker=FakeWorker())
    for base in ("http://127.0.0.1:8787", "http://localhost:8787", "http://127.0.0.1:9000"):
        with TestClient(app, base_url=base) as c:
            assert c.get("/api/runs").status_code == 200, base
