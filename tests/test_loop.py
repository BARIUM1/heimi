from dataclasses import replace

import pytest
import torch

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


class _PoisonGrad(torch.autograd.Function):
    """前向输出 0（损失保持有限），反向给出 NaN 梯度：模拟 bf16 反向溢出。"""

    @staticmethod
    def forward(ctx, x):
        ctx.shape = x.shape
        return x.sum() * 0.0

    @staticmethod
    def backward(ctx, g):
        return torch.full(ctx.shape, float("nan"))


def test_nonfinite_gradient_aborts_before_poisoning_checkpoint(tmp_path, monkeypatch):
    from minids.model import transformer as tr

    cfg = tiny_ladder(tmp_path, stage_steps=(4,), ckpt_interval=2)
    real = tr.Transformer.forward
    calls = {"n": 0}

    def poisoned(self, idx, targets=None):
        logits, loss = real(self, idx, targets)
        if self.training and targets is not None:
            calls["n"] += 1
            if calls["n"] > 2:  # 第 2 个优化步起梯度为 NaN，损失本身仍有限
                loss = loss + _PoisonGrad.apply(logits)
        return logits, loss

    monkeypatch.setattr(tr.Transformer, "forward", poisoned)
    run = tmp_path / "run"
    with pytest.raises(TrainingAborted, match="nan_abort"):
        run_ladder(cfg, run, "cpu")
    assert _events(run, "nan_abort")
    # 第 2 步的权重已被 NaN 污染，绝不能存成检查点，否则自动恢复会一直撞上它
    assert not (run / "ckpt" / "smoke0_step2.pt").exists()


def test_crash_in_final_eval_resumes_without_retraining(tmp_path, monkeypatch):
    from minids.eval import suite as suite_mod

    cfg = tiny_ladder(tmp_path, stage_steps=(3,), ckpt_interval=2)
    run = tmp_path / "run"
    real_run = suite_mod.EvalSuite.run

    def crash_on_final(self, model, *, micro_batch, hellaswag, fim):
        if fim:  # 级末评测才算 FIM
            raise RuntimeError("评测途中崩溃")
        return real_run(self, model, micro_batch=micro_batch, hellaswag=hellaswag, fim=fim)

    monkeypatch.setattr(suite_mod.EvalSuite, "run", crash_on_final)
    with pytest.raises(RuntimeError, match="崩溃"):
        run_ladder(cfg, run, "cpu")
    n_metrics = len(read_jsonl(run / "metrics.jsonl"))
    assert n_metrics == 3 and (run / "ckpt" / "smoke0_step3.pt").exists()  # 级末评测前已存档
    monkeypatch.setattr(suite_mod.EvalSuite, "run", real_run)
    run_ladder(cfg, run, "cpu")
    assert len(read_jsonl(run / "metrics.jsonl")) == n_metrics  # 没有重训任何一步
    assert (run / "ckpt" / "smoke0_final.pt").exists()
    assert len([r for r in read_jsonl(run / "eval.jsonl") if r["fim_em"] is not None]) == 1


def test_resume_skips_corrupt_latest_checkpoint(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(4,), ckpt_interval=2)
    run = tmp_path / "run"
    run_ladder(cfg, run, "cpu", max_steps=2)
    (run / "ckpt" / "smoke0_step4.pt").write_bytes(b"truncated by a power cut")
    run_ladder(cfg, run, "cpu")
    assert _events(run, "resume")[0]["data"]["checkpoint"] == "smoke0_step2.pt"
    assert (run / "ckpt" / "smoke0_step4.pt.corrupt").exists()  # 坏文件被挪开，不再被选中
    assert (run / "ckpt" / "smoke0_final.pt").exists()
    # 级末重新写出的 smoke0_step4.pt 是一个能读的新检查点
    from minids.train.checkpoint import load_checkpoint
    assert load_checkpoint(run / "ckpt" / "smoke0_step4.pt", "cpu")["state"]["step"] == 4


def test_oom_during_eval_is_reported_as_oom(tmp_path, monkeypatch):
    from minids.model import transformer as tr

    cfg = tiny_ladder(tmp_path, stage_steps=(2,))
    real = tr.Transformer.forward

    def oom_in_eval(self, idx, targets=None):
        if not self.training:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (模拟)")
        return real(self, idx, targets)

    monkeypatch.setattr(tr.Transformer, "forward", oom_in_eval)
    run = tmp_path / "run"
    with pytest.raises(TrainingAborted, match="oom"):
        run_ladder(cfg, run, "cpu")
    assert _events(run, "oom") and "micro_batch" in _events(run, "oom")[0]["data"]["hint"]


def test_vocab_mismatch_between_index_and_model_rejected(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(1,))
    bad = replace(cfg.stages[0], model=replace(cfg.stages[0].model, vocab_size=512))
    cfg = replace(cfg, stages=(bad,))
    with pytest.raises(ValueError, match="vocab"):
        run_ladder(cfg, tmp_path / "run", "cpu")
