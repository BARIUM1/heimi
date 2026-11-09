"""训练主循环：按 LadderConfig 逐级训练，级间增长并验证，随时可中断恢复，持续写四个 jsonl。

一个优化步消耗 tokens_per_step 个 token，由 accum_steps 个微步累积；加载器的 step 参数是跨级累计的
micro_step，恢复后取到的数据序列与未中断时一致；学习率是本级 step 的纯函数。每级结束写 <stage>_final.pt，
再次运行时已有 final 的级直接跳过；有中间检查点的级从它恢复（最新的那个读不出来就退到更早的一个）。

检查点总是先于同一步的评测与采样写盘：评测（HellaSwag、上千次生成）是最长、最陌生的代码路径，
它崩了不能连带丢掉几十分钟训练。级末也先存一个步检查点，再评测，最后写 _final.pt。

四种中止（spec 14）：损失或梯度范数非有限 → nan_abort（梯度在 opt.step 之前查，NaN 权重绝不落盘）；
CUDA OOM（整级任何位置）或第一步后 reserved 显存超过预算（Windows 上超限不报错而是溢出到系统内存）→ oom；
增长验证失败由 grow_for_stage 抛出；torch.compile 首次前向失败 → compile_fallback 后用普通模式继续，不中止。
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from minids.data.loader import MixedLoader, ShardSet
from minids.data.tokenizer import HeimiTokenizer
from minids.eval.samples import generate_samples
from minids.eval.suite import EvalSuite
from minids.gpu import gpu_temperature_c
from minids.logging_io import EvalRecord, MetricsRecord, RunLogger, SampleRecord, now_iso
from minids.memguard import cuda_memory_snapshot, exceeds_memory_budget
from minids.model.transformer import Transformer
from minids.train.checkpoint import (
    CheckpointError,
    TrainState,
    checkpoint_name,
    list_step_checkpoints,
    load_checkpoint,
    load_model,
    prune_step_checkpoints,
    replace_with_retry,
    save_checkpoint,
)
from minids.train.config import LadderConfig, StageConfig
from minids.train.grow import grow_for_stage
from minids.train.optim import build_optimizer
from minids.train.schedule import lr_at, warmup_steps

log = logging.getLogger(__name__)


class TrainingAborted(RuntimeError):
    """nan_abort / oom / 显存超预算：消息里带处理建议。"""


class _BudgetExhausted(Exception):
    """--max-steps 用完，本次进程到此为止（内部控制流）。"""


@dataclass
class _Run:
    cfg: LadderConfig
    ckpt_dir: Path
    device: str
    tok: HeimiTokenizer
    logger: RunLogger
    suite: EvalSuite
    vocab_size: int
    remaining: int | None  # 本次调用还能执行的优化步数；None 表示不限


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if _is_cuda(device) else contextlib.nullcontext()


def run_ladder(
    cfg: LadderConfig,
    run_dir: Path,
    device: str = "cuda",
    only_stage: str | None = None,
    max_steps: int | None = None,
) -> None:
    run_dir = Path(run_dir)
    index = json.loads(Path(cfg.data.index).read_text(encoding="utf-8"))
    vocab_size = int(index["vocab_size"])
    for s in cfg.stages:
        if s.model.vocab_size != vocab_size:
            raise ValueError(f"index.json 的 vocab_size={vocab_size} 与级 {s.name} 的 model.vocab_size={s.model.vocab_size} 不一致")
    tok = HeimiTokenizer.load(Path(cfg.data.tokenizer))
    run = _Run(cfg, run_dir / "ckpt", device, tok, RunLogger(run_dir), EvalSuite(cfg, tok, device), vocab_size, max_steps)

    if only_stage is not None:
        i = cfg.stage_index(only_stage)
        if i > 0:
            prev_final = run.ckpt_dir / checkpoint_name(cfg.stages[i - 1].name, None)
            if not prev_final.exists():
                raise FileNotFoundError(f"只跑级 {only_stage} 需要前一级的最终检查点 {prev_final}，但它不存在")
        todo = [cfg.stages[i]]
    else:
        todo = list(cfg.stages)

    for stage in todo:
        if run.remaining is not None and run.remaining <= 0:
            return
        if (run.ckpt_dir / checkpoint_name(stage.name, None)).exists():
            log.info("级 %s 已有最终检查点，跳过", stage.name)
            continue
        i = cfg.stage_index(stage.name)
        prev_stage = cfg.stages[i - 1] if i > 0 else None
        try:
            _run_stage(run, stage, prev_stage)
        except _BudgetExhausted:
            return


def _make_loader(run: _Run, stage: StageConfig) -> MixedLoader:
    cfg = run.cfg
    sets = {
        src: ShardSet.from_index(Path(cfg.data.index), src, "train", cfg.seq_len + 1)
        for src in ("web", "code")
        if getattr(stage.mix, src) > 0
    }
    return MixedLoader(sets, stage.mix, batch_size=stage.micro_batch, seq_len=cfg.seq_len, seed=cfg.seed, vocab_size=run.vocab_size)


def _load_latest_resumable(run: _Run, stage: StageConfig) -> tuple[Path, dict] | None:
    """最新的能读出来的中间检查点；读不出来的（断电截断等）改名为 .corrupt 后退到更早的。"""
    for path in reversed(list_step_checkpoints(run.ckpt_dir, stage.name)):
        try:
            return path, load_checkpoint(path, run.device)
        except CheckpointError as e:
            bad = path.with_name(path.name + ".corrupt")
            replace_with_retry(path, bad)  # 别的程序正占着它时等一等，而不是让恢复失败
            log.warning("%s；已改名为 %s，改用更早的检查点", e, bad.name)
    return None


def _enter_stage(run: _Run, stage: StageConfig, prev_stage: StageConfig | None, steps: int, accum: int):
    """返回 (model, optimizer, state)：恢复中间检查点，或从零/从前一级增长进入新一级。"""
    cfg = run.cfg
    found = _load_latest_resumable(run, stage)
    if found is not None:
        latest, payload = found
        if payload["model_config"] != stage.model.to_dict():
            raise ValueError(
                f"检查点 {latest.name} 的 model_config 与级 {stage.name} 的配置不一致：{payload['model_config']} ≠ {stage.model.to_dict()}"
            )
        if payload.get("ladder_config") != cfg.to_dict():
            log.warning("检查点 %s 的 ladder_config 与当前配置不同，按当前配置继续", latest.name)
        model = Transformer(stage.model).to(run.device)
        model.load_state_dict(payload["model"])
        opt = build_optimizer(model, stage.lr)
        if payload["optimizer"] is not None:
            opt.load_state_dict(payload["optimizer"])
        state = TrainState.from_dict(payload["state"])
        model.train()
        run.logger.event("resume", stage.name, state.global_step, state.tokens_total, checkpoint=latest.name, step=state.step)
        log.info("级 %s 从 %s 恢复（第 %d 步）", stage.name, latest.name, state.step)
        return model, opt, state

    if prev_stage is None:
        torch.manual_seed(cfg.seed)
        model = Transformer(stage.model).to(run.device)
        state = TrainState(stage.name)
    else:
        prev_final = run.ckpt_dir / checkpoint_name(prev_stage.name, None)
        if not prev_final.exists():
            raise FileNotFoundError(f"级 {stage.name} 需要前一级的最终检查点 {prev_final}，但它不存在")
        prev_model, meta = load_model(prev_final, run.device)
        prev_state = TrainState.from_dict(meta["state"])
        model, outcome = grow_for_stage(prev_model, stage, run.suite.growth_tokens, cfg.seed)
        if model is not prev_model:
            del prev_model
        state = TrainState(
            stage.name, step=0, global_step=prev_state.global_step, micro_step=prev_state.micro_step,
            tokens_stage=0, tokens_total=prev_state.tokens_total,
        )
        if outcome.kind != "none":  # 数据切换不是增长，窗口只在 growth 事件处画竖线
            run.logger.event("growth", stage.name, state.global_step, state.tokens_total, **outcome.to_dict())
            log.info("级 %s：%s 增长 %d → %d 参数，验证通过", stage.name, outcome.kind, outcome.params_before, outcome.params_after)
    model.train()
    opt = build_optimizer(model, stage.lr)
    run.logger.event(
        "stage_start", stage.name, state.global_step, state.tokens_total,
        params_non_embedding=model.count_params(), steps=steps, accum_steps=accum, micro_batch=stage.micro_batch, growth=stage.growth,
    )
    return model, opt, state


def _maybe_compile(run: _Run, model: Transformer, stage: StageConfig, state: TrainState):
    if not run.cfg.compile:
        return model
    try:
        return torch.compile(model)
    except Exception as e:  # noqa: BLE001
        run.logger.event("compile_fallback", stage.name, state.global_step, state.tokens_total, error=f"{type(e).__name__}: {str(e)[:200]}")
        log.warning("torch.compile 失败，改用普通模式：%s", e)
        return model


def _evaluate(run: _Run, stage: StageConfig, model: Transformer, state: TrainState, final: bool) -> None:
    with _autocast(run.device):
        res = run.suite.run(
            model, micro_batch=stage.micro_batch,
            hellaswag=final and run.cfg.eval.hellaswag,
            fim=final or stage.mix.fim_rate > 0,
        )
    run.logger.eval(EvalRecord(
        stage.name, state.global_step, state.tokens_total,
        res["val_loss_web"], res["val_loss_code"], res["hellaswag_acc"], res["fim_em"], now_iso(),
    ))
    log.info("评测 %s 第 %d 步：val web %.4f / code %.4f，hellaswag %s，fim_em %s",
             stage.name, state.step, res["val_loss_web"], res["val_loss_code"], res["hellaswag_acc"], res["fim_em"])
    model.train()


def _sample(run: _Run, stage: StageConfig, model: Transformer, state: TrainState) -> None:
    cfg = run.cfg
    model.eval()
    with _autocast(run.device):
        rows = generate_samples(
            model, run.tok, cfg.prompts, cfg.eval.sample_tokens, run.device, seed=cfg.seed + state.global_step,
            max_prefix_tokens=run.suite.max_prefix_tokens, max_suffix_tokens=run.suite.max_suffix_tokens,
        )
    for prompt_id, shown, completion in rows:
        run.logger.sample(SampleRecord(stage.name, state.global_step, state.tokens_total, prompt_id, shown, completion, now_iso()))
    model.train()


def _save_step(run: _Run, stage: StageConfig, model: Transformer, opt, state: TrainState, reason: str) -> None:
    path = run.ckpt_dir / checkpoint_name(stage.name, state.step)
    save_checkpoint(path, model, opt, state, run.cfg.to_dict())
    run.logger.event("checkpoint", stage.name, state.global_step, state.tokens_total, file=path.name, reason=reason)
    prune_step_checkpoints(run.ckpt_dir, stage.name, run.cfg.keep_checkpoints)


def _abort_nan(run: _Run, stage: StageConfig, state: TrainState, what: str, value: float) -> None:
    run.logger.event("nan_abort", stage.name, state.global_step, state.tokens_total, what=what, value=value, step=state.step + 1)
    raise TrainingAborted(
        f"nan_abort：级 {stage.name} 第 {state.step + 1} 步{what}为 {value}；权重未更新也未存盘，"
        f"重跑同一命令会从最近检查点恢复，仍不行就调低该级 lr"
    )


def _abort_oom(run: _Run, stage: StageConfig, state: TrainState | None, hint: str, **data) -> None:
    gs = state.global_step if state else 0
    tt = state.tokens_total if state else 0
    run.logger.event("oom", stage.name, gs, tt, hint=hint, **data)
    raise TrainingAborted("oom：" + hint)


def _run_stage(run: _Run, stage: StageConfig, prev_stage: StageConfig | None) -> None:
    state: TrainState | None = None
    try:
        state = _train_stage(run, stage, prev_stage)
    except torch.cuda.OutOfMemoryError as e:
        hint = f"显存不足：把级 {stage.name} 的 micro_batch 从 {stage.micro_batch} 减半，梯度累积会自动补回每步 token 数"
        _abort_oom(run, stage, state if state else _state_holder.get("state"), hint, error=str(e)[:200])


# _train_stage 把正在用的 TrainState 放进来，OOM 发生在训练循环里时事件也能带上正确的步数
_state_holder: dict[str, TrainState] = {}


def _train_stage(run: _Run, stage: StageConfig, prev_stage: StageConfig | None) -> TrainState:
    cfg, device = run.cfg, run.device
    steps, accum = cfg.steps_for(stage), cfg.accum_steps(stage)
    _state_holder.pop("state", None)
    model, opt, state = _enter_stage(run, stage, prev_stage, steps, accum)
    _state_holder["state"] = state
    fwd = _maybe_compile(run, model, stage, state)
    loader = _make_loader(run, stage)
    cuda = _is_cuda(device)
    tokens_per_micro = stage.micro_batch * cfg.seq_len
    warmup = warmup_steps(steps, stage.warmup_frac)
    stage_t0 = time.perf_counter()
    t_log, tokens_at_log = stage_t0, state.tokens_total
    memory_checked = False
    loss_last = float("nan")
    log.info("级 %s：%d 步，每步 %d 个微步 × %d 条 × %d token，从第 %d 步开始", stage.name, steps, accum, stage.micro_batch, cfg.seq_len, state.step)

    while state.step < steps:
        lr = lr_at(state.step, steps, stage.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = loader.get_batch(state.micro_step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            try:
                with _autocast(device):
                    _, loss = fwd(x, y)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001 —— 编译后的模型常在首次前向才暴露工具链缺失
                if fwd is model:
                    raise
                run.logger.event("compile_fallback", stage.name, state.global_step, state.tokens_total, error=f"{type(e).__name__}: {str(e)[:200]}")
                log.warning("编译模式首次前向失败，改用普通模式：%s", e)
                fwd = model
                with _autocast(device):
                    _, loss = fwd(x, y)
            (loss / accum).backward()
            loss_acc = loss_acc + loss.detach()
            state.micro_step += 1
            state.tokens_stage += tokens_per_micro
            state.tokens_total += tokens_per_micro

        loss_val = (loss_acc / accum).item()
        if not math.isfinite(loss_val):
            _abort_nan(run, stage, state, "损失", loss_val)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if not math.isfinite(grad_norm):  # 损失有限但 bf16 反向溢出：在 opt.step 之前拦住，NaN 权重不进检查点
            _abort_nan(run, stage, state, "梯度范数", grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        state.step += 1
        state.global_step += 1
        loss_last = loss_val

        if cuda and not memory_checked:
            memory_checked = True
            snap = cuda_memory_snapshot()
            if exceeds_memory_budget(snap["reserved_gb"], snap["total_gb"], cfg.max_mem_frac):
                hint = (f"第一步后 reserved 显存 {snap['reserved_gb']:.1f} GB 超过物理显存 {snap['total_gb']:.1f} GB 的 "
                        f"{cfg.max_mem_frac:.0%}，Windows 上会溢出到系统内存变得极慢：把级 {stage.name} 的 micro_batch 减半")
                _abort_oom(run, stage, state, hint, reserved_gb=round(snap["reserved_gb"], 2), total_gb=round(snap["total_gb"], 2))

        is_last = state.step == steps
        if state.step % cfg.log_interval == 0 or is_last:
            now = time.perf_counter()
            tok_per_s = (state.tokens_total - tokens_at_log) / max(now - t_log, 1e-9)
            if cuda:
                mem_gb = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
                temp = gpu_temperature_c()
            else:
                mem_gb, temp = 0.0, None
            run.logger.metrics(MetricsRecord(
                stage.name, state.step, state.global_step, state.tokens_stage, state.tokens_total,
                loss_val, lr, grad_norm, tok_per_s, mem_gb, temp, now_iso(),
            ))
            log.info("%s %d/%d loss %.4f lr %.2e gnorm %.2f %.0f tok/s %.1f GB", stage.name, state.step, steps, loss_val, lr, grad_norm, tok_per_s, mem_gb)
            t_log, tokens_at_log = now, state.tokens_total

        if is_last:
            break  # 级末的存档、评测、采样在循环外按固定顺序做

        if run.remaining is not None:
            run.remaining -= 1
        budget_hit = run.remaining is not None and run.remaining <= 0
        # 先存档再评测：评测是最长的代码路径，它崩了不能连带丢掉这段训练
        if budget_hit or state.step % cfg.ckpt_interval == 0:
            _save_step(run, stage, model, opt, state, reason="max_steps" if budget_hit else "interval")
        if state.step % cfg.eval.interval == 0:
            _evaluate(run, stage, model, state, final=False)
        if state.step % cfg.eval.sample_interval == 0:
            _sample(run, stage, model, state)
        if budget_hit:
            log.info("--max-steps 用完，已保存检查点，退出")
            raise _BudgetExhausted

    if run.remaining is not None:
        run.remaining -= 1
    _save_step(run, stage, model, opt, state, reason="stage_end")  # 级末评测前先落盘，评测崩了重跑直接接到这里
    _evaluate(run, stage, model, state, final=True)
    _sample(run, stage, model, state)
    save_checkpoint(run.ckpt_dir / checkpoint_name(stage.name, None), model, opt, state, cfg.to_dict())
    run.logger.event(
        "stage_end", stage.name, state.global_step, state.tokens_total,
        loss_last=loss_last, params_non_embedding=model.count_params(), seconds=round(time.perf_counter() - stage_t0, 1),
    )
    log.info("级 %s 完成：%d 步，末步损失 %.4f，用时 %.0f 秒", stage.name, state.step, loss_last, time.perf_counter() - stage_t0)
    return state
