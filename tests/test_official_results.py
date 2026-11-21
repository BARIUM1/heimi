import sys
from dataclasses import asdict

from minids.logging_io import append_jsonl
from minids.official.results import (
    CODE_METHOD, append_record, code_method_of, final_checkpoints, latest_per_checkpoint, load_reference, pending_checkpoints,
    read_records, save_reference,
)
from tests.helpers_official import legacy_record, official_record


def test_records_round_trip_and_a_missing_file_is_empty(tmp_path):
    assert read_records(tmp_path) == [] and read_records(tmp_path / "nope") == []
    append_record(tmp_path, official_record("stage0_final.pt", 300))
    append_record(tmp_path, official_record("stage1_final.pt", 800, limit=50, code_limit=0, blimp_acc=0.6))
    got = read_records(tmp_path)
    assert [r["checkpoint"] for r in got] == ["stage0_final.pt", "stage1_final.pt"]
    assert got[0]["metrics"] == {"blimp_acc": 0.5} and got[0]["limit"] is None and got[1]["code_limit"] == 0


def test_latest_per_checkpoint_prefers_the_most_complete_run_then_the_latest():
    recs = [asdict(r) for r in (
        official_record("stage1_final.pt", 800, blimp_acc=0.60),
        official_record("stage0_final.pt", 300, blimp_acc=0.55),
        official_record("stage0_final.pt", 300, limit=50, blimp_acc=0.90),  # 后来又抽样试了一次：不该盖掉全量
        official_record("stage1_final.pt", 800, blimp_acc=0.61),  # 同为全量：留后写的
        official_record("stage1_final.pt", 800, code_limit=0, blimp_acc=0.99),  # 后来又跑了一次不带代码评测的：不该盖掉带的
    )]
    got = latest_per_checkpoint(recs)
    assert [(r["checkpoint"], r["metrics"]["blimp_acc"]) for r in got] == [("stage0_final.pt", 0.55), ("stage1_final.pt", 0.61)]


def test_a_checkpoint_with_only_sampled_runs_shows_the_last_one():
    recs = [asdict(official_record("stage0_final.pt", 300, limit=20, blimp_acc=0.5)), asdict(official_record("stage0_final.pt", 300, limit=50, blimp_acc=0.7))]
    assert [r["limit"] for r in latest_per_checkpoint(recs)] == [50]


def test_pending_lists_final_checkpoints_without_a_record_for_these_settings(tmp_path):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    for name in ("stage0_final.pt", "stage0_step500.pt", "stage1_final.pt", "notes.txt"):
        (ck / name).write_bytes(b"x")
    append_record(tmp_path, official_record("stage0_final.pt", 300))
    assert [p.name for p in pending_checkpoints(tmp_path, None, 500)] == ["stage1_final.pt"]
    assert [p.name for p in pending_checkpoints(tmp_path, 50, 500)] == ["stage0_final.pt", "stage1_final.pt"]
    assert pending_checkpoints(tmp_path / "nope", None, 500) == []


def test_code_method_of_a_record_is_none_without_code_evaluation_and_plain_when_missing():
    assert code_method_of(asdict(official_record("stage0_final.pt", 300))) == CODE_METHOD == "healed"
    assert code_method_of(legacy_record("stage0_final.pt", 300)) == "plain"
    assert code_method_of(asdict(official_record("stage0_final.pt", 300, code_limit=0))) is None
    assert code_method_of(legacy_record("stage0_final.pt", 300, code_limit=0)) is None


def test_records_made_with_the_old_code_prompting_are_still_pending(tmp_path):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    for name in ("stage0_final.pt", "stage1_final.pt"):
        (ck / name).write_bytes(b"x")
    append_jsonl(tmp_path / "official.jsonl", legacy_record("stage0_final.pt", 300))  # 带代码评测、旧出题方式：要重评
    append_jsonl(tmp_path / "official.jsonl", legacy_record("stage1_final.pt", 800, code_limit=0))  # 没跑代码评测：出题方式无关
    assert [p.name for p in pending_checkpoints(tmp_path, None, 500)] == ["stage0_final.pt", "stage1_final.pt"]
    assert [p.name for p in pending_checkpoints(tmp_path, None, 0)] == ["stage0_final.pt"]
    append_record(tmp_path, official_record("stage0_final.pt", 300))
    assert [p.name for p in pending_checkpoints(tmp_path, None, 500)] == ["stage1_final.pt"]


def test_a_rerun_with_the_current_code_prompting_replaces_the_old_record_in_the_table():
    old = legacy_record("stage0_final.pt", 300, repobench_em=0.03)
    new = asdict(official_record("stage0_final.pt", 300, repobench_em=0.10))
    assert [r["metrics"]["repobench_em"] for r in latest_per_checkpoint([old, new])] == [0.10]
    # 旧代码后来又补写了一条（没有 code_method）：同样完整，但不是现在的出题方式，不盖掉
    assert [r["metrics"]["repobench_em"] for r in latest_per_checkpoint([new, old])] == [0.10]
    # 没跑代码评测的记录不分出题方式：照旧留后写的
    a, b = legacy_record("stage0_final.pt", 300, code_limit=0, blimp_acc=0.5), asdict(official_record("stage0_final.pt", 300, code_limit=0, blimp_acc=0.6))
    assert [r["metrics"]["blimp_acc"] for r in latest_per_checkpoint([b, a])] == [0.5]


def test_reference_file_round_trips_and_keeps_other_entries(tmp_path):
    p = tmp_path / "docs" / "official" / "reference.json"
    assert load_reference(p) == {}
    save_reference(p, "gpt2", {"label": "GPT-2 small", "metrics": {"blimp_acc": 0.8}})
    save_reference(p, "other", {"label": "X", "metrics": {}})
    assert load_reference(p)["gpt2"]["metrics"] == {"blimp_acc": 0.8} and set(load_reference(p)) == {"gpt2", "other"}
    assert [f.name for f in p.parent.iterdir()] == ["reference.json"]  # 没有留下临时文件


def test_an_unreadable_reference_file_reads_as_empty(tmp_path):
    p = tmp_path / "reference.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_reference(p) == {}
    p.write_text("[1, 2]", encoding="utf-8")
    assert load_reference(p) == {}


def test_results_module_does_not_pull_in_torch():
    import subprocess

    code = "import sys, minids.official.results; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _shown(recs):
    return [(r["limit"], r["code_limit"], r["metrics"]["blimp_acc"]) for r in latest_per_checkpoint([asdict(r) for r in recs])]


def test_full_run_wins_over_sampled_in_both_write_orders():
    sampled = official_record("stage0_final.pt", 300, limit=50, blimp_acc=0.9)
    full = official_record("stage0_final.pt", 300, blimp_acc=0.5)
    assert _shown([sampled, full]) == [(None, 500, 0.5)]
    assert _shown([full, sampled]) == [(None, 500, 0.5)]


def test_full_without_code_still_beats_sampled_with_code():
    sampled_code = official_record("stage0_final.pt", 300, limit=50, code_limit=500, blimp_acc=0.9)
    full_nocode = official_record("stage0_final.pt", 300, code_limit=0, blimp_acc=0.5)
    assert _shown([sampled_code, full_nocode]) == [(None, 0, 0.5)]
    assert _shown([full_nocode, sampled_code]) == [(None, 0, 0.5)]


def test_code_evaluation_wins_over_none_in_both_write_orders():
    with_code = official_record("stage0_final.pt", 300, code_limit=500, blimp_acc=0.5)
    no_code = official_record("stage0_final.pt", 300, code_limit=0, blimp_acc=0.9)
    assert _shown([with_code, no_code]) == [(None, 500, 0.5)]
    assert _shown([no_code, with_code]) == [(None, 500, 0.5)]
    # 同为抽样时同样先看代码评测
    s_code = official_record("stage0_final.pt", 300, limit=20, code_limit=500, blimp_acc=0.5)
    s_none = official_record("stage0_final.pt", 300, limit=20, code_limit=0, blimp_acc=0.9)
    assert _shown([s_code, s_none]) == [(20, 500, 0.5)]
    assert _shown([s_none, s_code]) == [(20, 500, 0.5)]


def test_latest_written_wins_when_otherwise_equal():
    a = official_record("stage0_final.pt", 300, blimp_acc=0.5)
    b = official_record("stage0_final.pt", 300, blimp_acc=0.6)
    assert _shown([a, b]) == [(None, 500, 0.6)]
    assert _shown([b, a]) == [(None, 500, 0.5)]


def test_final_checkpoints_lists_finals_by_mtime_then_name(tmp_path):
    import os

    assert final_checkpoints(tmp_path) == [] and final_checkpoints(tmp_path / "nope") == []
    ck = tmp_path / "ckpt"
    ck.mkdir()
    for name, t in (("b_final.pt", 100), ("a_final.pt", 200), ("c_final.pt", 100), ("a_step5.pt", 50), ("n.txt", 1)):
        p = ck / name
        p.write_bytes(b"x")
        os.utime(p, ns=(t * 10**9, t * 10**9))
    assert [p.name for p in final_checkpoints(tmp_path)] == ["b_final.pt", "c_final.pt", "a_final.pt"]
