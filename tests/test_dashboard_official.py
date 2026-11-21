from pathlib import Path

import pytest

from dashboard import cli, data
from dashboard.server import create_app
from minids.official.results import OFFICIAL_FILE, append_record, save_reference
from tests.helpers_dashboard import FakeWorker, local_client
from tests.helpers_official import official_record

GPT2 = {"label": "GPT-2 small", "model_id": "openai-community/gpt2", "params_non_embedding": 85_000_000, "limit": None, "code_limit": 500,
        "metrics": {"blimp_acc": 0.8}, "harness_version": "0.4.13", "time": "t"}


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "runs" / "heimi"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def ref(tmp_path):
    p = tmp_path / "reference.json"
    save_reference(p, "gpt2", GPT2)
    return p


def test_rows_are_deduplicated_ordered_and_flagged_against_the_reference(run, ref):
    append_record(run, official_record("stage1_final.pt", 800, blimp_acc=0.7))
    append_record(run, official_record("stage0_final.pt", 300, limit=50, blimp_acc=0.6))
    append_record(run, official_record("stage1_final.pt", 800, limit=20, blimp_acc=0.9))
    append_record(run, official_record("stage2_final.pt", 2800, code_limit=0, blimp_acc=0.8))
    out = data.official(run, ref)
    assert [(r["checkpoint"], r["metrics"]["blimp_acc"], r["comparable"]) for r in out["rows"]] == [
        ("stage0_final.pt", 0.6, False),  # 只有抽样跑的：不能和全量的参照并排比
        ("stage1_final.pt", 0.7, True),  # 后来的抽样记录没有盖掉全量的
        ("stage2_final.pt", 0.8, False),  # 没跑代码评测：样本设置和参照不同
    ]
    assert out["reference"] == [{"name": "gpt2", "label": "GPT-2 small", "params_non_embedding": 85_000_000, "limit": None, "code_limit": 500, "metrics": {"blimp_acc": 0.8}}]


def test_same_settings_as_the_reference_are_comparable_and_no_reference_means_comparable(run, ref):
    append_record(run, official_record("stage0_final.pt", 300))
    assert data.official(run, ref)["rows"][0]["comparable"] is True
    assert data.official(run, None)["rows"][0]["comparable"] is True


def test_nothing_recorded_and_no_reference_is_an_empty_answer(run, tmp_path):
    assert data.official(run, None) == {"rows": [], "reference": []}
    assert data.official(run, tmp_path / "missing.json") == {"rows": [], "reference": []}


def test_a_log_damaged_in_the_middle_is_reported_as_unreadable(run):
    append_record(run, official_record("stage0_final.pt", 300))
    with open(run / OFFICIAL_FILE, "a", encoding="utf-8") as f:
        f.write("{broken\n")
    append_record(run, official_record("stage1_final.pt", 800))
    with pytest.raises(data.LogUnreadable, match="official.jsonl"):
        data.official(run, None)


def test_endpoint_serves_rows_and_reference_and_refuses_bad_run_names(run, ref):
    append_record(run, official_record("stage0_final.pt", 300, blimp_acc=0.6))
    with local_client(create_app(run.parent, worker=FakeWorker(), reference_path=ref)) as c:
        body = c.get("/api/runs/heimi/official").json()
        assert body["rows"][0]["metrics"] == {"blimp_acc": 0.6} and body["reference"][0]["name"] == "gpt2"
        assert c.get("/api/runs/nope/official").status_code == 404
        assert c.get("/api/runs/..%5C..%5Cx/official").status_code == 404


def test_endpoint_answers_without_a_reference_path(dash_runs):
    with local_client(create_app(dash_runs[0], worker=FakeWorker())) as c:
        assert c.get("/api/runs/heimi/official").json() == {"rows": [], "reference": []}


def test_cli_points_the_app_at_the_committed_reference_file(run, ref, monkeypatch):
    assert cli.REFERENCE_FILE == cli.REPO_ROOT / "docs" / "official" / "reference.json"
    monkeypatch.setattr(cli, "REFERENCE_FILE", ref)
    app, _ = cli.build(["--runs", str(run.parent), "--no-browser"])
    with local_client(app) as c:
        assert c.get("/api/runs/heimi/official").json()["reference"][0]["label"] == "GPT-2 small"


def test_page_has_an_official_region_fed_by_its_endpoint():
    html = (Path("dashboard/static") / "index.html").read_text(encoding="utf-8")
    assert 'id="official"' in html and "官方评测" in html and 'id="official-rows"' in html and "/official" in html
    for word in ("BLiMP", "HellaSwag", "bits/byte", "RepoBench", "单行填空", "official_eval.py", "抽样"):
        assert word in html, word


def test_page_isolates_official_failures_from_the_connection_banner():
    # 没有浏览器可跑脚本，只检查页面源码：官方评测的取数有自己的 catch，错误写进自己那一格
    html = (Path("dashboard/static") / "index.html").read_text(encoding="utf-8")
    assert "官方评测记录读不出来" in html
    fetch = next(line for line in html.splitlines() if "/official`" in line)
    assert ".catch(" in fetch
