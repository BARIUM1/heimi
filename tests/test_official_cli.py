import json
from pathlib import Path

import pytest

import official_eval
from minids.official.results import OfficialRecord, load_reference, read_records
from minids.train.checkpoint import CheckpointError


@pytest.fixture
def runs(tmp_path):
    ck = tmp_path / "runs" / "heimi" / "ckpt"
    ck.mkdir(parents=True)
    for name in ("stage0_final.pt", "stage1_final.pt", "stage1_step500.pt"):
        (ck / name).write_bytes(b"x")
    return tmp_path / "runs"


class _Calls(list):
    """记录假评测函数收到的调用；broken 里的检查点名一评就抛 CheckpointError。"""

    def __init__(self):
        super().__init__()
        self.broken: set[str] = set()


@pytest.fixture
def calls(monkeypatch):
    seen = _Calls()

    def fake(ckpt, device, data_dir, limit=None, code_limit=500, **kwargs):
        name = Path(ckpt).name
        seen.append((name, device, limit, code_limit))
        if name in seen.broken:
            raise CheckpointError(f"检查点 {name} 无法读取")
        return OfficialRecord(name, name.split("_")[0], 1, 100, 10, limit, code_limit, {"blimp_acc": 0.5}, "0.4.13", "t")

    monkeypatch.setattr(official_eval, "evaluate_checkpoint", fake)
    monkeypatch.setattr(official_eval, "ensure_code_data", lambda data_dir: seen.append(("data", str(data_dir))))
    monkeypatch.setattr(official_eval, "gpu_status", lambda: None)
    monkeypatch.setattr(official_eval, "harness_available", lambda: True)
    return seen


def _main(runs, *extra):
    return official_eval.main(["--run", "heimi", "--runs", str(runs), "--device", "cpu", "--data-dir", str(runs.parent / "eval"), *extra])


def _evals(calls):
    return [c for c in calls if c[0] != "data"]


def test_default_evaluates_each_pending_final_checkpoint_once(runs, calls, capsys):
    assert _main(runs) == 0
    assert _evals(calls) == [("stage0_final.pt", "cpu", None, 500), ("stage1_final.pt", "cpu", None, 500)]
    assert calls[0][0] == "data"  # 先确保代码评测数据在
    assert [r["checkpoint"] for r in read_records(runs / "heimi")] == ["stage0_final.pt", "stage1_final.pt"]
    printed = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [p["checkpoint"] for p in printed] == ["stage0_final.pt", "stage1_final.pt"]
    assert _main(runs) == 0 and len(_evals(calls)) == 2  # 第二次没有可评的
    assert "没有待评测" in capsys.readouterr().err
    assert _main(runs, "--force") == 0 and len(_evals(calls)) == 4


def test_named_checkpoint_and_sampling_options(runs, calls):
    assert _main(runs, "--ckpt", "stage1_step500.pt", "--limit", "20", "--code-limit", "30") == 0
    assert _main(runs, "--ckpt", "stage1_step500.pt", "--code-all") == 0
    assert _main(runs, "--ckpt", "stage1_step500.pt", "--skip-code") == 0
    assert _evals(calls) == [("stage1_step500.pt", "cpu", 20, 30), ("stage1_step500.pt", "cpu", None, None), ("stage1_step500.pt", "cpu", None, 0)]
    assert sum(c[0] == "data" for c in calls) == 2  # 第三次带 --skip-code：不去下载代码评测数据


def test_a_broken_checkpoint_is_named_and_the_rest_are_still_evaluated(runs, calls, capsys):
    calls.broken.add("stage0_final.pt")
    assert _main(runs) == 1
    assert [r["checkpoint"] for r in read_records(runs / "heimi")] == ["stage1_final.pt"]
    assert "stage0_final.pt" in capsys.readouterr().err


def test_a_busy_gpu_is_refused_unless_explicitly_allowed(runs, calls, monkeypatch, capsys):
    monkeypatch.setattr(official_eval, "gpu_status", lambda: {"temp_c": 70.0, "mem_used_gb": 12.0, "mem_total_gb": 16.0, "util_pct": 99.0})
    args = ["--run", "heimi", "--runs", str(runs), "--data-dir", str(runs.parent / "eval"), "--device", "cuda"]
    assert official_eval.main(args) == 2 and _evals(calls) == []
    err = capsys.readouterr().err
    assert "--allow-busy-gpu" in err and "--device cpu" in err
    assert official_eval.main([*args, "--allow-busy-gpu"]) == 0 and len(_evals(calls)) == 2
    assert _main(runs, "--force") == 0  # 用 CPU 时不看显卡


def test_unknown_run_or_checkpoint_name_is_a_usage_error(runs, calls, capsys):
    assert official_eval.main(["--run", "nope", "--runs", str(runs), "--device", "cpu"]) == 2
    assert _main(runs, "--ckpt", "missing_final.pt") == 2
    assert _main(runs, "--ckpt", "../heimi/ckpt/stage0_final.pt") == 2
    assert _evals(calls) == [] and "missing_final.pt" in capsys.readouterr().err


def test_missing_official_dependencies_print_the_install_command(runs, calls, monkeypatch, capsys):
    monkeypatch.setattr(official_eval, "harness_available", lambda: False)
    assert _main(runs) == 2 and _evals(calls) == []
    assert ".[official]" in capsys.readouterr().err


def test_reference_run_writes_the_reference_file(tmp_path, calls, monkeypatch):
    seen = []

    def fake_reference(device, data_dir, limit=None, code_limit=500, **kwargs):
        seen.append((device, limit, code_limit))
        return {"label": "GPT-2 small", "metrics": {"blimp_acc": 0.8}, "limit": limit, "code_limit": code_limit}

    monkeypatch.setattr(official_eval, "evaluate_reference", fake_reference)
    out = tmp_path / "docs" / "reference.json"
    code = official_eval.main(["--reference", "gpt2", "--device", "cpu", "--limit", "5", "--data-dir", str(tmp_path / "eval"), "--reference-out", str(out)])
    assert code == 0 and seen == [("cpu", 5, 500)]
    assert load_reference(out)["gpt2"]["metrics"] == {"blimp_acc": 0.8}


def test_run_and_reference_cannot_be_combined_or_both_omitted(runs, calls):
    with pytest.raises(SystemExit):
        official_eval.main(["--device", "cpu"])
    with pytest.raises(SystemExit):
        official_eval.main(["--run", "heimi", "--reference", "gpt2"])


def test_reference_run_prepares_code_data_before_evaluating(tmp_path, calls, monkeypatch):
    order = []
    monkeypatch.setattr(official_eval, "ensure_code_data", lambda data_dir: order.append("data"))
    monkeypatch.setattr(official_eval, "evaluate_reference", lambda device, data_dir, **kw: order.append("eval") or {"metrics": {}})
    out = tmp_path / "reference.json"
    assert official_eval.main(["--reference", "gpt2", "--device", "cpu", "--data-dir", str(tmp_path), "--reference-out", str(out)]) == 0
    assert order == ["data", "eval"]
    order.clear()
    out2 = tmp_path / "other.json"  # 另写一个文件：不带代码评测的一次不会撞上上面那条更完整的
    assert official_eval.main(["--reference", "gpt2", "--device", "cpu", "--skip-code", "--data-dir", str(tmp_path), "--reference-out", str(out2)]) == 0
    assert order == ["eval"]


def test_a_checkpoint_without_ladder_config_is_named_and_the_rest_continue(runs, calls, monkeypatch, capsys):
    import minids.official.run as official_run

    def fake_load_model(ckpt, device):
        meta = {"state": {"stage": "stage0", "global_step": 1, "tokens_total": 10}} if Path(ckpt).name == "stage0_final.pt" else {}
        return object(), meta

    monkeypatch.setattr(official_run, "load_model", fake_load_model)

    def real_then_fake(ckpt, device, data_dir, **kw):
        if Path(ckpt).name == "stage0_final.pt":
            return official_run.evaluate_checkpoint(ckpt, device, data_dir, **kw)
        return OfficialRecord("stage1_final.pt", "stage1", 1, 100, 10, None, 500, {"blimp_acc": 0.5}, "0.4.13", "t")

    monkeypatch.setattr(official_eval, "evaluate_checkpoint", real_then_fake)
    assert _main(runs) == 1
    assert [r["checkpoint"] for r in read_records(runs / "heimi")] == ["stage1_final.pt"]
    err = capsys.readouterr().err
    assert "stage0_final.pt" in err and "内容不完整" in err


def _reference_main(tmp_path, out, *extra):
    return official_eval.main(["--reference", "gpt2", "--device", "cpu", "--data-dir", str(tmp_path / "eval"), "--reference-out", str(out), *extra])


@pytest.fixture
def ref_calls(calls, monkeypatch):
    """假的参照评测：记下收到的 (limit, code_limit)，返回带这两项的条目。"""
    seen = []

    def fake_reference(device, data_dir, limit=None, code_limit=500, **kwargs):
        seen.append((limit, code_limit))
        return {"label": "GPT-2 small", "metrics": {"blimp_acc": 0.5}, "limit": limit, "code_limit": code_limit}

    monkeypatch.setattr(official_eval, "evaluate_reference", fake_reference)
    return seen


def _full_reference(out):
    from minids.official.results import save_reference

    save_reference(out, "gpt2", {"label": "GPT-2 small", "metrics": {"blimp_acc": 0.8}, "limit": None, "code_limit": 500})


def test_a_sampled_reference_run_does_not_replace_a_more_complete_one(tmp_path, ref_calls, calls, capsys):
    out = tmp_path / "reference.json"
    _full_reference(out)
    assert _reference_main(tmp_path, out, "--limit", "20") == 2
    err = capsys.readouterr().err
    assert "--force" in err and "--reference-out" in err and "更完整" in err
    assert ref_calls == [] and calls == []  # 评测和下载都没开始
    assert load_reference(out)["gpt2"]["metrics"] == {"blimp_acc": 0.8}
    assert _reference_main(tmp_path, out, "--skip-code") == 2  # 不带代码评测同样算不够完整
    assert ref_calls == []


def test_force_lets_a_less_complete_reference_run_overwrite(tmp_path, ref_calls):
    out = tmp_path / "reference.json"
    _full_reference(out)
    assert _reference_main(tmp_path, out, "--limit", "20", "--force") == 0
    assert ref_calls == [(20, 500)] and load_reference(out)["gpt2"]["limit"] == 20


def test_an_equally_or_more_complete_reference_run_overwrites(tmp_path, ref_calls):
    out = tmp_path / "reference.json"
    from minids.official.results import save_reference

    save_reference(out, "gpt2", {"metrics": {}, "limit": 20, "code_limit": 500})
    assert _reference_main(tmp_path, out, "--limit", "20") == 0  # 一样完整
    assert _reference_main(tmp_path, out) == 0  # 更完整
    assert ref_calls == [(20, 500), (None, 500)] and load_reference(out)["gpt2"]["limit"] is None


def test_reference_run_with_no_existing_file_or_a_damaged_one_just_writes(tmp_path, ref_calls):
    out = tmp_path / "reference.json"
    assert _reference_main(tmp_path, out, "--limit", "5") == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert _reference_main(tmp_path, bad, "--limit", "5") == 0
    assert ref_calls == [(5, 500), (5, 500)]


@pytest.mark.parametrize("flag", ["--limit", "--code-limit"])
@pytest.mark.parametrize("bad", ["0", "-3", "1.5", "abc"])
def test_limit_options_must_be_integers_of_at_least_one(runs, calls, flag, bad, capsys):
    with pytest.raises(SystemExit) as e:
        _main(runs, flag, bad)
    assert e.value.code == 2 and _evals(calls) == [] and flag in capsys.readouterr().err
