from pathlib import Path


from dashboard.cli import build
from tests.helpers_dashboard import local_client


def test_defaults_bind_localhost_8787_and_open_browser(tmp_path):
    app, opts = build(["--runs", str(tmp_path)])
    assert (opts.host, opts.port, opts.open_browser, opts.runs) == ("127.0.0.1", 8787, True, tmp_path)
    with local_client(app) as c:
        assert c.get("/api/runs").json() == []  # 还没有任何 run 时窗口也能开
        assert c.get("/api/status").json()["run"] is None


def test_port_and_no_browser_flags(tmp_path):
    _, opts = build(["--runs", str(tmp_path), "--port", "9000", "--no-browser"])
    assert opts.port == 9000 and opts.open_browser is False and opts.host == "127.0.0.1"


def test_root_script_is_a_thin_shim():
    src = Path("dashboard.py").read_text(encoding="utf-8")
    assert "from dashboard.cli import main" in src and "SystemExit(main())" in src


def test_repo_root_does_not_depend_on_where_the_dashboard_is_launched_from(tmp_path, monkeypatch):
    # 从别的目录启动（python F:\...\dashboard.py）时，检查点里的相对分词器路径仍要按仓库根目录解析
    import dashboard.cli as cli

    seen = {}
    real = cli.create_app

    def recording(runs_dir, worker=None, repo_root=None):
        seen["repo_root"] = repo_root
        return real(runs_dir, worker=worker, repo_root=repo_root)

    monkeypatch.setattr(cli, "create_app", recording)
    monkeypatch.chdir(tmp_path)
    cli.build(["--runs", str(tmp_path)])
    assert (seen["repo_root"] / "minids" / "config.py").exists() and (seen["repo_root"] / "dashboard" / "cli.py").exists()
