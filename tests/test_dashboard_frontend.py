import hashlib
import re
from pathlib import Path


from dashboard.server import create_app
from tests.helpers_dashboard import FakeWorker, local_client

STATIC = Path("dashboard/static")


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_index_is_served_with_four_regions_in_chinese(dash_runs):
    with local_client(create_app(dash_runs[0], worker=FakeWorker())) as c:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert c.get("/static/chart.umd.min.js").status_code == 200
    html = r.text
    assert "<title>黑米 Heimi" in html and 'lang="zh-CN"' in html
    for section_id, heading in (("overview", "成长总览"), ("status", "当前状态"), ("samples", "成长样本墙"), ("playground", "试玩台")):
        assert f'id="{section_id}"' in html and heading in html
    assert 'id="loss-chart"' in html and 'id="params-chart"' in html and 'id="empty"' in html and "还没有训练记录" in html


def test_page_is_self_contained_and_offline():
    html = _html()
    assert '<script src="/static/chart.umd.min.js"></script>' in html
    assert not re.search(r"""(src|href)\s*=\s*["'](https?:)?//""", html)  # 不引用任何外网资源
    assert not re.search(r"<link[^>]+stylesheet", html)  # CSS 内联
    js = (STATIC / "chart.umd.min.js").read_bytes()
    assert hashlib.sha256(js).hexdigest() == "87364cc2b2a76a6684929e7d6b80b18a897794db38400caff1fcf1791c6925d2"  # Chart.js 4.6.1


def test_model_text_is_never_inserted_as_html():
    html = _html()
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "document.write" not in html


def test_page_polls_every_five_seconds_and_uses_every_endpoint():
    html = _html()
    assert "5000" in html
    for path in ("/api/runs", "/api/status", "/overview", "/metrics?since=", "/samples", "/api/generate"):
        assert path in html, path


def test_status_panel_follows_the_run_being_viewed():
    assert "/api/status?run=" in _html()
