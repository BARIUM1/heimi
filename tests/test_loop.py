from dataclasses import replace

import pytest

from minids.logging_io import read_jsonl
from minids.train.loop import TrainingAborted, run_ladder
from tests.helpers_train import tiny_ladder


def _events(run_dir, kind):
    return [e for e in read_jsonl(run_dir / "events.jsonl") if e["type"] == kind]


def test_four_stages_write_all_logs_and_checkpoints(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(3, 2, 2, 2))
    run = tmp_path / "run"
    run_ladder(cfg, run, "cpu")
    for name in ("metrics.jsonl", "eval.jsonl", "events.jsonl", "samples.jsonl"):
        assert read_jsonl(run / name), name
    assert {p.name for p in (run / "ckpt").iterdir()} >= {"smoke0_final.pt", "smoke1_final.pt", "smoke2_final.pt", "smoke_task_final.pt"}
    growth = _events(run, "growth")
    assert [g["data"]["kind"] for g in growth] == ["depth", "width"]  # 进入 smoke_task 不是增长，不写 growth 事件
    assert growth[0]["data"]["exact_report"]["exact"] is True
    assert growth[1]["data"]["noisy_report"]["max_rel_err"] <= cfg.stages[2].rel_tol
    assert len(_events(run, "stage_start")) == 4 and len(_events(run, "stage_end")) == 4
    m = read_jsonl(run / "metrics.jsonl")
    assert [r["global_step"] for r in m] == list(range(1, 10))
    assert all(a["tokens_total"] < b["tokens_total"] for a, b in zip(m, m[1:]))
    assert m[3]["stage"] == "smoke1" and m[3]["step"] == 1 and m[3]["tokens_total"] == m[2]["tokens_total"] + 256
    last_eval = read_jsonl(run / "eval.jsonl")[-1]
    assert last_eval["stage"] == "smoke_task" and last_eval["hellaswag_acc"] is None and 0 <= last_eval["fim_em"] <= 1
    assert {s["prompt_id"] for s in read_jsonl(run / "samples.jsonl")} == {"en", "fim"}


def test_interrupted_run_resumes_bit_exactly(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(4,), ckpt_interval=2)
    run_ladder(cfg, tmp_path / "a", "cpu")
    run_ladder(cfg, tmp_path / "b", "cpu", max_steps=2)
    assert (tmp_path / "b" / "ckpt" / "smoke0_step2.pt").exists() and not (tmp_path / "b" / "ckpt" / "smoke0_final.pt").exists()
    run_ladder(cfg, tmp_path / "b", "cpu")
    a = [r["loss"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl")]
    b = [r["loss"] for r in read_jsonl(tmp_path / "b" / "metrics.jsonl")]
    assert a == b and len(a) == 4
    assert len(_events(tmp_path / "b", "resume")) == 1 and _events(tmp_path / "b", "checkpoint")


def test_resume_rejects_changed_model_config(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(4,), ckpt_interval=2)
    run_ladder(cfg, tmp_path / "run", "cpu", max_steps=2)
    wider = replace(cfg.stages[0], model=replace(cfg.stages[0].model, d_model=128, n_head=4, kv_lora_rank=64, ffn_hidden=256))
    with pytest.raises(ValueError, match="model_config"):
        run_ladder(replace(cfg, stages=(wider,)), tmp_path / "run", "cpu")


def test_finished_stage_is_skipped_untouched(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(2,))
    run = tmp_path / "run"
    run_ladder(cfg, run, "cpu")
    before = {p: p.stat().st_mtime_ns for p in run.rglob("*") if p.is_file()}
    run_ladder(cfg, run, "cpu")
    assert {p: p.stat().st_mtime_ns for p in run.rglob("*") if p.is_file()} == before


def test_only_stage_requires_previous_final(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(2, 2))
    with pytest.raises(FileNotFoundError, match="smoke0_final.pt"):
        run_ladder(cfg, tmp_path / "run", "cpu", only_stage="smoke1")


def test_nan_loss_aborts_with_event_and_keeps_checkpoint(tmp_path, monkeypatch):
    from minids.model import transformer as tr

    cfg = tiny_ladder(tmp_path, stage_steps=(4,), ckpt_interval=2)
    real = tr.Transformer.forward
    calls = {"n": 0}

    def poisoned(self, idx, targets=None):
        logits, loss = real(self, idx, targets)
        if self.training and targets is not None:  # 只数训练前向，评测在 eval 模式下不算
            calls["n"] += 1
            if calls["n"] > 4:  # 每步累积 2 个微步：第 5 次是第 3 个优化步的第一个微步
                loss = loss * float("nan")
        return logits, loss

    monkeypatch.setattr(tr.Transformer, "forward", poisoned)
    run = tmp_path / "run"
    with pytest.raises(TrainingAborted, match="nan_abort"):
        run_ladder(cfg, run, "cpu")
    assert _events(run, "nan_abort") and (run / "ckpt" / "smoke0_step2.pt").exists()
    assert not (run / "ckpt" / "smoke0_final.pt").exists()


def test_vocab_mismatch_between_index_and_model_rejected(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(1,))
    bad = replace(cfg.stages[0], model=replace(cfg.stages[0].model, vocab_size=512))
    cfg = replace(cfg, stages=(bad,))
    with pytest.raises(ValueError, match="vocab"):
        run_ladder(cfg, tmp_path / "run", "cpu")
