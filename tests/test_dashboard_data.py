import pytest

from dashboard.data import (
    RunNotFound,
    format_params,
    list_checkpoints,
    list_runs,
    overview,
    run_summary,
    safe_checkpoint,
    safe_run_dir,
)
from minids.logging_io import EvalRecord, RunLogger, now_iso


def test_format_params():
    assert [format_params(n) for n in (12_934_528, 25_868_672, 103_248_640, 75_136, 595_328)] == ["13M", "26M", "103M", "75K", "595K"]


def test_safe_run_dir_rejects_traversal_and_unknown(tmp_path):
    runs = tmp_path / "runs"
    (runs / "heimi").mkdir(parents=True)
    (tmp_path / "secret").mkdir()
    assert safe_run_dir(runs, "heimi") == runs / "heimi"
    for bad in ("..", "../secret", "heimi/../../secret", "", ".", "nope", "a b", "heimi\\..\\..\\secret", "C:secret"):
        with pytest.raises(RunNotFound):
            safe_run_dir(runs, bad)


def test_list_runs_and_summary_on_real_tiny_run(dash_runs):
    runs_dir, _ = dash_runs
    runs = list_runs(runs_dir)
    assert [r["name"] for r in runs] == ["heimi"]
    r = runs[0]
    assert [s["name"] for s in r["stages"]] == ["smoke0", "smoke1"] and all(s["done"] for s in r["stages"])
    assert r["stages"][1]["growth"] == "depth" and r["stages"][0]["steps"] == 2 and r["stages"][1]["params"] > r["stages"][0]["params"]
    assert r["stage"] == "smoke1" and r["global_step"] == 4 and r["tokens_total"] == 4 * 256 and r["last_update"]
    names = [c["name"] for c in r["checkpoints"]]
    assert set(names) == {"smoke0_final.pt", "smoke1_final.pt", "smoke0_step2.pt", "smoke1_step2.pt"}
    assert all(n.endswith("_final.pt") for n in names[:2])


def test_list_checkpoints_ignores_partials_and_parses_names(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    for n in ("s0_step500.pt", "s0_final.pt", "s1_step20.pt", "s1_step40.tmp", "s0_step9.pt.corrupt", "notes.txt"):
        (d / n).write_bytes(b"x")
    out = list_checkpoints(tmp_path)
    assert out[0]["name"] == "s0_final.pt" and {c["name"] for c in out} == {"s0_final.pt", "s0_step500.pt", "s1_step20.pt"}
    by = {c["name"]: c for c in out}
    assert (by["s0_final.pt"]["kind"], by["s0_final.pt"]["step"], by["s0_final.pt"]["stage"]) == ("final", None, "s0")
    assert (by["s0_step500.pt"]["kind"], by["s0_step500.pt"]["step"], by["s0_step500.pt"]["stage"]) == ("step", 500, "s0")
    assert list_checkpoints(tmp_path / "no-such-run") == []


def test_safe_checkpoint_only_accepts_listed_names(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "s0_final.pt").write_bytes(b"x")
    (d / "s0_step9.pt.corrupt").write_bytes(b"x")
    (tmp_path / "outside.pt").write_bytes(b"x")
    assert safe_checkpoint(tmp_path, "s0_final.pt") == d / "s0_final.pt"
    for bad in ("../outside.pt", "s0_step9.pt.corrupt", "nope.pt", ""):
        with pytest.raises(RunNotFound):
            safe_checkpoint(tmp_path, bad)


def test_overview_dedupes_eval_and_labels_growth(tmp_path):
    log = RunLogger(tmp_path)
    log.event("stage_start", "s0", 0, 0, params_non_embedding=75_136, steps=3, accum_steps=2, micro_batch=4, growth="none")
    log.eval(EvalRecord("s0", 3, 768, 5.0, 4.0, None, None, now_iso()))
    log.eval(EvalRecord("s0", 3, 768, 4.9, 3.9, 0.25, 0.0, now_iso()))  # 同一步的第二条：evaluate.py 重算
    log.event("stage_end", "s0", 3, 768, loss_last=4.8, params_non_embedding=75_136, seconds=1.0)
    log.event("growth", "s1", 3, 768, kind="depth", params_before=75_136, params_after=150_208,
              exact_report={"exact": True}, noisy_report=None, seconds=0.1)
    log.event("stage_start", "s1", 3, 768, params_non_embedding=150_208, steps=2, accum_steps=2, micro_batch=4, growth="depth")
    log.eval(EvalRecord("s1", 5, 1280, 4.5, 3.5, None, None, now_iso()))
    o = overview(tmp_path)
    assert [(e["global_step"], e["val_loss_web"], e["hellaswag_acc"]) for e in o["eval"]] == [(3, 4.9, 0.25), (5, 4.5, None)]
    assert o["growth"] == [{"tokens_total": 768, "kind": "depth", "params_before": 75_136, "params_after": 150_208, "label": "75K → 150K 加深"}]
    assert [(p["tokens_total"], p["params"]) for p in o["params"]] == [(0, 75_136), (768, 150_208), (1280, 150_208)]


def test_empty_or_missing_run_gives_empty_results(tmp_path):
    assert overview(tmp_path) == {"eval": [], "growth": [], "params": []}
    s = run_summary(tmp_path)
    assert s["stages"] == [] and s["stage"] is None and s["global_step"] == 0 and s["last_update"] is None and s["checkpoints"] == []
    assert list_runs(tmp_path / "does-not-exist") == []
