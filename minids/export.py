"""导出推理用的模型目录（spec 11.2）：model.safetensors、config.json、tokenizer.json。

导出目录不依赖训练代码里的检查点格式，complete.py 与窗口的试玩台都从这里加载。
config.json 里带模型结构、训练到哪一级、累计 token 数和代码提交号，方便事后对照。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from minids.config import ModelConfig
from minids.data.tokenizer import HeimiTokenizer
from minids.logging_io import now_iso
from minids.model.transformer import Transformer
from minids.train.checkpoint import load_model


def export_model(ckpt_path: str | Path, tokenizer_path: str | Path, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_model(Path(ckpt_path), "cpu")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    state = meta.get("state", {})
    cfg = {
        "name": "heimi",
        "model": model.cfg.to_dict(),
        "stage": state.get("stage"),
        "global_step": state.get("global_step"),
        "tokens_total": state.get("tokens_total"),
        "git_commit": meta.get("git_commit"),
        "exported_at": now_iso(),
    }
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(Path(tokenizer_path), out / "tokenizer.json")
    return out


def load_exported(model_dir: str | Path, device: str) -> tuple[Transformer, HeimiTokenizer]:
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    model = Transformer(ModelConfig.from_dict(cfg["model"]))
    model.load_state_dict(load_file(str(d / "model.safetensors"), device="cpu"))
    return model.to(device).eval(), HeimiTokenizer.load(d / "tokenizer.json")
