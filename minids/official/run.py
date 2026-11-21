"""评一个检查点：框架任务 + 代码评测，合成一条 OfficialRecord。

不 import lm_eval（HeimiLM、run_harness、harness_version 都在函数内 import），
所以没装 official 额外依赖时 import 本模块也不报错。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from minids.data.tokenizer import HeimiTokenizer
from minids.logging_io import now_iso
from minids.official.codebench import infill_scores, repobench_scores, sample_rows
from minids.official.codedata import INFILL_LOCAL, REPOBENCH_LOCAL, load_infill, load_repobench
from minids.official.completers import heimi_infill, heimi_next_line
from minids.official.results import CODE_METHOD, OfficialRecord
from minids.train.checkpoint import CheckpointError, load_model

CODE_LIMIT = 500
REPO_ROOT = Path(__file__).resolve().parents[2]


def code_metrics(
    next_line: Callable[[str], str], infill: dict[str, Callable[[str, str], str]], data_dir: Path, code_limit: int | None
) -> dict[str, float]:
    """RepoBench 下一行补全 + 单行填空；infill 里每个名字给一个 infill_em_<名字>。"""
    data_dir = Path(data_dir)
    rb_path, if_path = data_dir / REPOBENCH_LOCAL, data_dir / INFILL_LOCAL
    for p in (rb_path, if_path):
        if not p.exists():
            raise FileNotFoundError(f"缺代码评测数据文件 {p}；先运行 official_eval.py，它会下载")
    rb = sample_rows(load_repobench(rb_path), code_limit)
    fl = sample_rows(load_infill(if_path), code_limit)
    r = repobench_scores(next_line, rb)
    metrics: dict[str, float] = {"repobench_em": r["em"], "repobench_es": r["es"], "repobench_n": r["n"]}
    for name, fn in infill.items():
        s = infill_scores(fn, fl)
        metrics[f"infill_em_{name}"] = s["em"]
        metrics["infill_n"] = s["n"]
    metrics.setdefault("infill_n", len(fl))
    return metrics


def evaluate_checkpoint(
    ckpt: Path, device: str, data_dir: Path, limit: int | None = None, code_limit: int | None = CODE_LIMIT, harness: Callable | None = None
) -> OfficialRecord:
    from minids.official.harness import HeimiLM, harness_version

    if harness is None:
        from minids.official.harness import run_harness as harness
    ckpt = Path(ckpt)
    model, meta = load_model(ckpt, "cpu")
    try:
        tok_path = Path(meta["ladder_config"]["data"]["tokenizer"])
        state = meta["state"]
        stage, global_step, tokens_total = state["stage"], state["global_step"], state["tokens_total"]
    except (KeyError, TypeError) as e:
        raise CheckpointError(f"检查点 {ckpt.name} 内容不完整：缺 {e}") from e
    if not tok_path.is_absolute():
        tok_path = REPO_ROOT / tok_path
    model.to(device)
    tok = HeimiTokenizer.load(tok_path)
    metrics = dict(harness(HeimiLM(model, tok, device), limit=limit))
    if code_limit != 0:
        metrics.update(
            code_metrics(
                heimi_next_line(model, tok, device),
                {"fim": heimi_infill(model, tok, device, "fim"), "prefix": heimi_infill(model, tok, device, "prefix")},
                data_dir,
                code_limit,
            )
        )
    return OfficialRecord(
        checkpoint=ckpt.name, stage=stage, global_step=global_step, tokens_total=tokens_total,
        params_non_embedding=model.count_params(), limit=limit, code_limit=code_limit, metrics=metrics,
        harness_version=harness_version(), time=now_iso(), code_method=CODE_METHOD,
    )
