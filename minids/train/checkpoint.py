"""检查点：模型权重、优化器状态、训练状态、配置快照、git 提交号一起落盘。

写法：先写 <path>.tmp 再 os.replace，断电留下的半成品不会被当成检查点。
读法：torch.load(weights_only=True)，所以 payload 里只放张量、数字、字符串、列表、字典，配置以 dict 保存。
学习率是 step 的纯函数（schedule.py），没有调度器状态要存；数据顺序由 TrainState.micro_step 决定。
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from minids.config import ModelConfig
from minids.logging_io import now_iso
from minids.model.transformer import Transformer


class CheckpointError(RuntimeError):
    pass


@dataclass
class TrainState:
    stage: str
    step: int = 0  # 本级内已完成的优化步
    global_step: int = 0  # 跨级累计优化步
    micro_step: int = 0  # 加载器 get_batch 的 step 参数，跨级累计
    tokens_stage: int = 0
    tokens_total: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        return cls(**d)


def git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False, cwd=Path(__file__).resolve().parents[2],
        )
        s = r.stdout.strip()
        return s if r.returncode == 0 and s else None
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_name(stage: str, step: int | None) -> str:
    return f"{stage}_step{step}.pt" if step is not None else f"{stage}_final.pt"


_STEP_RE = re.compile(r"^(?P<stage>.+)_step(?P<n>\d+)\.pt$")


def _step_checkpoints(ckpt_dir: Path, stage: str) -> list[tuple[int, Path]]:
    """该级的全部中间检查点，按步数升序。"""
    found = []
    for p in Path(ckpt_dir).glob(f"{stage}_step*.pt"):
        m = _STEP_RE.match(p.name)
        if m and m.group("stage") == stage:
            found.append((int(m.group("n")), p))
    return sorted(found)


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer | None,
    state: TrainState,
    ladder_cfg: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "state": state.to_dict(),
        "model_config": model.cfg.to_dict(),
        "ladder_config": dict(ladder_cfg),
        "git_commit": git_commit(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "saved_at": now_iso(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: str) -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e:  # noqa: BLE001 —— 坏文件会抛出各种 pickle/zip 错误，统一成一个带文件名的错误
        raise CheckpointError(f"检查点 {path.name} 无法读取（文件损坏或不是检查点）：{type(e).__name__}: {str(e)[:200]}") from e


def load_model(path: Path, device: str) -> tuple[Transformer, dict]:
    """按 model_config 重建模型并加载权重，eval 模式；第二项是去掉权重与优化器的元数据。"""
    payload = load_checkpoint(path, device)
    try:
        model = Transformer(ModelConfig.from_dict(payload["model_config"]))
        model.load_state_dict(payload["model"])
    except (KeyError, TypeError, RuntimeError) as e:
        raise CheckpointError(f"检查点 {Path(path).name} 内容不完整：{e}") from e
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k not in ("model", "optimizer")}
    return model, meta


def latest_step_checkpoint(ckpt_dir: Path, stage: str) -> Path | None:
    found = _step_checkpoints(ckpt_dir, stage)
    return found[-1][1] if found else None


def prune_step_checkpoints(ckpt_dir: Path, stage: str, keep: int) -> list[Path]:
    """只留该级最新的 keep 个中间检查点，不碰 _final 与其他级；返回删掉的路径。"""
    found = _step_checkpoints(ckpt_dir, stage)
    victims = found[: max(0, len(found) - keep)]
    deleted = []
    for _, p in victims:
        p.unlink()
        deleted.append(p)
    return deleted
