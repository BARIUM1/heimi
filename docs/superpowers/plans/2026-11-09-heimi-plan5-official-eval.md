# 黑米 计划五：官方评测接入 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用公开的标准评测、在标准评测框架下给黑米的每个阶段打分，同机同法跑一条 GPT-2 small 参照线，结果进窗口。

**Architecture:** `minids/official/scoring.py` 是不依赖任何框架的打分核心（续写的对数似然、滚动窗口）；`harness.py` 把它包成 lm-evaluation-harness 的 `LM` 子类，跑 `blimp`、`hellaswag`、`wikitext` 三个官方任务；`codebench.py`、`codedata.py`、`completers.py` 是不执行代码的两项代码补全评测（RepoBench 下一行、HumanEval 单行填空）；`reference.py` 让 GPT-2 走同一套流程；`results.py` 管 `runs/<run>/official.jsonl` 与 `docs/official/reference.json`；根目录的 `official_eval.py` 是命令行入口；窗口加一个只读接口和第五个区域。

**Tech Stack:** Python 3.12、PyTorch 2.11（cu128）、`lm_eval[hf]==0.4.13`（带来 transformers 5.22、datasets 5.2）、pyarrow、huggingface_hub、FastAPI、原生 HTML/JS、pytest。

**Spec:** `docs/superpowers/specs/2026-11-09-heimi-official-eval-design.md`（本计划实现它的全部内容）。它是主设计文档 `docs/superpowers/specs/2026-11-08-heimi-growth-lm-design.md` 的补充；两者冲突时，官方评测的事以前者为准，其余以后者为准。

## 本计划在整体中的位置

计划一到四（已合并）交付了模型、数据、训练和窗口。级 0 已经真训完，级 1、级 2、任务阶段和对照组由用户夜间运行。本计划不依赖它们训完：用现有的 `runs/heimi/ckpt/stage0_final.pt` 就能把整条链路做出来并用真实数据验证；之后每训完一级，用户运行一次 `python official_eval.py --run heimi`，表里多一行。

依赖已有接口：`minids.train.checkpoint.load_model(path, device) -> (Transformer, meta)`（`meta["state"]` 有 `stage`、`global_step`、`tokens_total`，`meta["ladder_config"]["data"]["tokenizer"]` 是分词器路径）、`Transformer.forward(idx, targets=None) -> (logits, loss)`、`Transformer.generate(idx, max_new_tokens, temperature, top_k)`、`Transformer.count_params()`、`minids.data.tokenizer.{HeimiTokenizer, EOT, PAD}`、`minids.completion.{complete_middle, greedy_line}`、`minids.logging_io.{append_jsonl, read_jsonl, now_iso}`、`minids.gpu.gpu_status()`、`dashboard.data.{read_log, safe_run_dir, LogUnreadable}`、`dashboard.server.create_app`、`dashboard.cli.{build, REPO_ROOT}`。

## 已核实的事实（2026-11-09）

- `uv pip install --dry-run "lm_eval[hf]==0.4.13"` 在本项目环境里新增 57 个包（transformers 5.22.2、accelerate 1.15.0、peft 0.21.2、datasets 5.2.0、evaluate 0.4.6 等），不改动 torch 2.11.0+cu128 与 numpy 2.5.4。
- 在一个仓库外的临时环境里实际装了 `lm_eval==0.4.13` 并用一个假模型跑通：`python -W error -c "import lm_eval, lm_eval.api.model, lm_eval.api.instance, lm_eval.utils"` 无警告；核心包不 import torch。
- `lm_eval.api.model.LM` 的抽象方法恰好三个：`loglikelihood`、`loglikelihood_rolling`、`generate_until`。子类不需要定义别的属性，`lm_eval.simple_evaluate(model=<LM 实例>, tasks=[...], limit=N, bootstrap_iters=0, log_samples=False)` 就能跑。
- 请求是 `lm_eval.api.instance.Instance`，字段 `request_type, doc, arguments, idx, ...`，参数从 `.args` 取。BLiMP 的一条请求是 `('', " Katherine can't help herself.")`：上下文为空串，续写带前导空格。WikiText 的滚动请求是一元组 `(整篇文本,)`。
- 返回值：`res["results"][任务名]` 是字典，数值指标的键形如 `"acc,none"`、`"acc_norm,none"`、`"word_perplexity,none"`、`"byte_perplexity,none"`、`"bits_per_byte,none"`；同名的 `"..._stderr,none"` 在 `bootstrap_iters=0` 时是字符串 `"N/A"`；另有 `name`、`alias`、`sample_len`、`sample_count` 等非指标键。任务组 `blimp` 的汇总同时出现在 `res["results"]["blimp"]` 和 `res["groups"]["blimp"]`，键是 `"acc,none"`；`res["group_subtasks"]["blimp"]` 有 67 项。`lm_eval.__version__ == "0.4.13"`。
- 框架对因果语言模型的打分方式（`lm_eval/api/model.py` 与 `lm_eval/models/huggingface.py`）：上下文为空时用 `[prefix_token_id]` 作上下文；否则把上下文末尾的空白挪到续写开头，`whole = encode(context + continuation)`、`ctx = encode(context)`、`cont = whole[len(ctx):]`；输入是 `(ctx + cont)[-(max_length + 1):][:-1]`；分数是续写各 token 的 log-softmax 之和，另返回"续写是否每个 token 都是 argmax"。滚动打分用 `get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len=1)` 再逐窗 `make_disjoint_window`。
- 数据规模（Hugging Face datasets-server）：`nyu-mll/blimp` 67 个子集各 1,000 行；`EleutherAI/wikitext_document_level` 的 `wikitext-2-raw-v1` 测试集 62 行；`tianyang/repobench_python_v1.1` 的 `in_file` 7,910 行（文件 `data/in_file-00000-of-00001-9721c68f6224a587.parquet`，156 MB）；`loubnabnl/humaneval_infilling` 的 SingleLine 1,033 行（文件 `data/HumanEval-SingleLineInfilling.jsonl`，1.65 MB，字段 `task_id, entry_point, prompt, suffix, canonical_solution, test`）。BLiMP、HellaSwag、WikiText 三份数据已经在本机的 Hugging Face 缓存里（首次下载 BLiMP 的 67 个子集约 1 分钟）。
- RepoBench 行的字段：`repo_name, file_path, context, import_statement, token_num, cropped_code, all_code, next_line, gold_snippet_index, created_at, level`。官方脚本（`Leolty/repobench` 的 `data/utils.py`、`run.py`、`evaluation/metrics.py`）：当前文件部分的提示是 `f"# Path: {file_path}\n{import_statement}\n{cropped_code.rstrip()}\n"`，整个提示再做 `re.sub(r'\n{4,}', '\n\n', prompt)`；预测取生成文本里第一行不是注释的代码；EM 是 `pred.split() == gt.split()`；ES 是 `fuzzywuzzy.fuzz.ratio(gt, pred)`。
- 仓库根目录的 `evaluate.py` 与 Hugging Face 的 `evaluate` 库同名，从仓库根目录运行时前者会遮住后者。框架只在一个指标不在自己的注册表里时才 `import evaluate`（`lm_eval/api/registry.py:632`，函数内）；本计划的三个任务的指标（`acc`、`acc_norm`、三个困惑度指标）都在注册表里，不触发。所以本计划的命令行不提供自选任务的选项。
- `minids` 在环境里是可编辑安装（`__editable__.minids-0.1.0.pth`）。`pytest` 当前 217 个测试全过，带 `-W error`。
- 检查点文件大小约为总参数量 × 12 字节（权重加 AdamW 的两份状态）：级 0 是 306 MB。`load_model(path, "cpu")` 之后再 `.to(device)` 可以避免把优化器状态搬上显卡（窗口的生成进程就是这么做的）。

## Global Constraints

- 不执行模型写出来的代码，任何情况下都不（设计说明第 3 节）。
- `minids/train/`、`train.py`、`evaluate.py`、`minids/eval/` 一行不改。官方评测是训练之外单独运行的命令（设计说明第 3 节）。
- 不装 `official` 额外依赖时，训练、自有评测、窗口必须照常工作：只有 `minids/official/harness.py` 在模块顶层 import `lm_eval`；`transformers` 只在 `minids/official/reference.py` 的 `load_gpt2` 函数体内 import；`minids/official/results.py` 不 import torch（窗口进程要用它）。
- 框架版本写死 `lm_eval[hf]==0.4.13`，每条记录带 `harness_version`（设计说明第 4 节第 1 条）。
- 官方打分用 float32，不开 autocast；解码一律贪心（设计说明第 4 节第 2、3 条）。
- 空上下文的条件 token 是 `EOT`（id 0）（设计说明第 4 节第 6 条）。
- 代码评测默认抽 500 条，种子 1337，取样后保持原顺序（设计说明第 4 节第 4 条）。
- 记录里 `limit` 为 `None` 表示框架任务全量；`code_limit` 为 `None` 表示代码评测全量，为 `0` 表示没跑代码评测，为正数表示抽样条数。
- 指标名固定为设计说明第 6 节列出的那 12 个。
- 窗口只读：`official.jsonl` 只由 `official_eval.py` 写，窗口不创建、不修改任何文件。
- 模型产出的文本只能以文本节点写进页面（计划四的规则，`tests/test_dashboard_frontend.py` 已有检查）。
- 单元测试在 CPU 上运行、不访问网络、不读 `runs/` 与 `data/` 下的真实文件、带 `-W error` 通过；只有 Task 9 的实机验证用真实数据和显卡。注释中文、标识符英文。
- 每个 commit 信息用英文 conventional commits 格式，末尾加一行 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`。

## Review Focus

1. 训练正在跑的时候在 GPU 上启动官方评测：Windows 上显存不够不报错而是溢出到系统内存，训练和评测一起慢一到两个数量级，用户一觉醒来训练没跑完。期望：拒绝启动并说明三条出路。测试在 Task 7。
2. 抽样跑的结果混进全量结果：用户先 `--limit 50` 试了一下，再跑全量，或者反过来；表里同一个检查点只能有一行，全量优先，抽样的行必须有标记，不能和 GPT-2 的全量数字并排冒充可比。测试在 Task 5 与 Task 8。
3. 一批检查点里有一个坏的（文件损坏、或是别的程序留下的同名文件）：报出它的名字，其余照常评完，退出码非零；不能因为第一个坏了就一个都不评。测试在 Task 7。
4. 模型在代码补全里什么都没写、只写了注释、或第一个 token 就是特殊标记（级 0 的小模型和没训过 FIM 的检查点经常这样）：按答错计分，不崩溃，特殊标记的字面串不进入预测。测试在 Task 3 与 Task 4。
5. 没装官方评测依赖、或数据下载到一半断网：前者给出一句安装命令然后退出，后者不留下半个文件（否则下次运行会把半个文件当成完整数据读）。测试在 Task 4 与 Task 7。

---

## 文件结构

```
pyproject.toml                        # 修改：加 official 额外依赖
minids/official/__init__.py           # 空
minids/official/scoring.py            # encode_pair、continuation_logprobs、rolling_windows、rolling_logprob
minids/official/harness.py            # HeimiLM、HARNESS_TASKS、harness_version、flatten_results、run_harness
minids/official/codebench.py          # first_code_line、edit_similarity、lines_match、repobench_prompt、sample_rows、repobench_scores、infill_scores
minids/official/codedata.py           # download_dataset_file、ensure_code_data、load_repobench、load_infill 与文件名常量
minids/official/completers.py         # greedy_text、heimi_next_line、heimi_infill
minids/official/results.py            # OfficialRecord、append_record、read_records、latest_per_checkpoint、pending_checkpoints、load_reference、save_reference
minids/official/run.py                # code_metrics、evaluate_checkpoint
minids/official/reference.py          # load_gpt2、hf_next_line、hf_infill_prefix、evaluate_reference
official_eval.py                      # 命令行入口
dashboard/data.py                     # 修改：加 official()
dashboard/server.py                   # 修改：create_app 加 reference_path，加 GET /api/runs/{run}/official
dashboard/cli.py                      # 修改：REFERENCE_FILE，传给 create_app
dashboard/static/index.html           # 修改：第五个区域"官方评测"
docs/official-eval.md                 # 使用说明与读数指南
docs/official/reference.json          # GPT-2 参照结果（Task 9 生成）
docs/dashboard/official.jpg           # 第五个区域的截图（Task 9）
docs/dashboard.md                     # 修改：补一段第五个区域
tests/helpers_official.py             # official_record、write_code_data、SpyScripted
tests/test_official_scoring.py
tests/test_official_harness.py
tests/test_official_codebench.py
tests/test_official_codedata.py
tests/test_official_completers.py
tests/test_official_results.py
tests/test_official_run.py
tests/test_official_cli.py
tests/test_dashboard_official.py
```

所有命令都从仓库根目录运行。测试命令的完整形式是 `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q -W error <路径>`，下文简写成 `pytest <路径>`。

---

### Task 1: 依赖与打分核心

**Files:**
- Modify: `pyproject.toml`
- Create: `minids/official/__init__.py`、`minids/official/scoring.py`
- Test: `tests/test_official_scoring.py`

**Interfaces:**
- Consumes: `Transformer.forward`、`minids.data.tokenizer.{EOT, PAD}`。
- Produces（都在 `minids/official/scoring.py`）:
  - `encode_pair(tok, context: str, continuation: str, prefix_token: int = EOT) -> tuple[list[int], list[int]]`。`tok` 只需要有 `encode(str) -> list[int]`。
  - `continuation_logprobs(model, pairs: Sequence[tuple[list[int], list[int]]], device: str, max_len: int, batch_size: int = 64, pad_id: int = PAD) -> list[tuple[float, bool]]`，返回值与 `pairs` 同序。
  - `rolling_windows(tokens: Sequence[int], prefix_token: int, max_len: int) -> list[tuple[list[int], list[int]]]`
  - `rolling_logprob(model, tokens: Sequence[int], device: str, max_len: int, prefix_token: int = EOT, batch_size: int = 8) -> float`

- [ ] **Step 1: 加额外依赖并安装**

`pyproject.toml` 的 `[project.optional-dependencies]` 加一行 `official = ["lm_eval[hf]==0.4.13"]`。

Run: `uv pip install --python .venv/Scripts/python.exe -e ".[dev,official]"`
然后: `.venv/Scripts/python.exe -c "import torch, lm_eval, transformers; print(torch.__version__, lm_eval.__version__, transformers.__version__, torch.cuda.is_available())"`
Expected: `2.11.0+cu128 0.4.13 5.22.2 True`（transformers 的小版本号以实际解析为准；torch 那一项必须原样不变，变了就停下来查）

Run: `pytest`
Expected: `217 passed`，新装的包没有让已有测试出警告。

- [ ] **Step 2: 写失败的测试**

`tests/test_official_scoring.py`：

```python
import math

import pytest
import torch
import torch.nn.functional as F

from minids.model.transformer import Transformer
from minids.official.scoring import continuation_logprobs, encode_pair, rolling_logprob, rolling_windows
from tests.helpers_data import tiny_tokenizer


@pytest.fixture
def model(tiny_cfg):
    torch.manual_seed(0)
    return Transformer(tiny_cfg).eval()  # 词表 256、上下文 64


def _direct(model, ctx, cont):
    """不经过被测代码，直接算续写的对数似然之和与是否贪心。"""
    logp = F.log_softmax(model(torch.tensor([ctx + cont[:-1]]))[0][0].double(), dim=-1)
    rows = logp[len(ctx) - 1 :]
    want = torch.tensor(cont)
    return rows[torch.arange(len(cont)), want].sum().item(), bool((rows.argmax(-1) == want).all())


def _random_ids(n, seed):
    return torch.randint(16, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_score_is_the_summed_log_probability_of_the_continuation(model):
    ctx, cont = [20, 21, 22, 23], [30, 31, 32]
    ((lp, greedy),) = continuation_logprobs(model, [(ctx, cont)], "cpu", max_len=64)
    want, want_greedy = _direct(model, ctx, cont)
    assert math.isclose(lp, want, rel_tol=1e-4, abs_tol=1e-4) and greedy == want_greedy


def test_batching_and_padding_change_neither_scores_nor_order(model):
    shapes = [(3, 2), (20, 5), (1, 1), (40, 7), (9, 9), (2, 30), (15, 1)]
    pairs = [(_random_ids(n, i), _random_ids(m, 100 + i)) for i, (n, m) in enumerate(shapes)]
    one = [continuation_logprobs(model, [p], "cpu", 64)[0] for p in pairs]
    batched = continuation_logprobs(model, pairs, "cpu", 64, batch_size=3)
    assert [g for _, g in batched] == [g for _, g in one]
    assert all(math.isclose(a[0], b[0], rel_tol=1e-4, abs_tol=1e-4) for a, b in zip(batched, one))


def test_long_context_is_cut_from_the_left_and_the_continuation_kept(model):
    ctx, cont = _random_ids(100, 2), [40, 41, 42, 43, 44]
    full = continuation_logprobs(model, [(ctx, cont)], "cpu", max_len=64)[0]
    kept = continuation_logprobs(model, [(ctx[-(64 + 1 - len(cont)) :], cont)], "cpu", max_len=64)[0]
    assert math.isclose(full[0], kept[0], rel_tol=1e-6, abs_tol=1e-6)


def test_greedy_flag_is_true_only_when_every_token_is_the_models_first_choice(model):
    ctx = [20, 21, 22]
    best = model.generate(torch.tensor([ctx]), 4, temperature=0.0)[0, len(ctx) :].tolist()
    assert continuation_logprobs(model, [(ctx, best)], "cpu", 64)[0][1] is True
    other = best[:3] + [(best[3] + 1) % 256]
    assert continuation_logprobs(model, [(ctx, other)], "cpu", 64)[0][1] is False


def test_empty_continuation_scores_zero_without_running_the_model():
    assert continuation_logprobs(None, [([5], [])], "cpu", 64) == [(0.0, True)]


def test_a_continuation_longer_than_the_context_window_is_refused(model):
    continuation_logprobs(model, [([0], list(range(16, 16 + 64)))], "cpu", max_len=64)  # 恰好等于窗口：可以
    with pytest.raises(ValueError, match="rolling"):
        continuation_logprobs(model, [([0], list(range(16, 16 + 65)))], "cpu", max_len=64)


def test_empty_context_is_replaced_by_the_document_separator(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    assert encode_pair(tok, "", " The cat sat.") == ([0], tok.encode(" The cat sat."))


def test_trailing_space_of_the_context_moves_onto_the_continuation(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    ctx, cont = encode_pair(tok, "The cat ", "sat")
    assert ctx == tok.encode("The cat") and cont == tok.encode("The cat sat")[len(ctx) :]


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 128, 150])
def test_rolling_windows_cover_every_token_exactly_once(n):
    tokens = list(range(100, 100 + n))
    windows = rolling_windows(tokens, prefix_token=0, max_len=64)
    assert [t for _, cont in windows for t in cont] == tokens
    assert all(len(ctx) >= 1 and len(ctx) + len(cont) <= 65 for ctx, cont in windows)
    if windows:
        assert windows[0][0] == [0]
    for ctx, cont in windows[1:]:
        start = tokens.index(cont[0])
        assert ctx == tokens[start - len(ctx) : start]  # 紧挨在这一窗之前的原文


def test_rolling_logprob_is_the_sum_over_windows(model):
    tokens = _random_ids(150, 3)
    want = sum(lp for lp, _ in continuation_logprobs(model, rolling_windows(tokens, 0, 64), "cpu", 64))
    assert math.isclose(rolling_logprob(model, tokens, "cpu", 64), want, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(rolling_logprob(model, [30, 31, 32], "cpu", 64), _direct(model, [0], [30, 31, 32])[0], rel_tol=1e-4, abs_tol=1e-4)
    assert rolling_logprob(model, [], "cpu", 64) == 0.0
```

- [ ] **Step 3: 运行，确认失败**

Run: `pytest tests/test_official_scoring.py`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'minids.official'`

- [ ] **Step 4: 实现 `minids/official/scoring.py`**

建空的 `minids/official/__init__.py`。四个函数的规则：

`encode_pair`：`context == ""` 时返回 `([prefix_token], tok.encode(continuation))`。否则先把 `context` 末尾的空白（`len(context) - len(context.rstrip())` 个字符）挪到 `continuation` 开头，再 `whole = tok.encode(context + continuation)`、`ctx = tok.encode(context)`，返回 `(ctx, whole[len(ctx):])`。

`continuation_logprobs`：续写为空的条目直接得 `(0.0, True)`，不进模型；续写长于 `max_len` 抛 `ValueError`，消息里写明这种文本应走 `rolling_logprob`。其余条目的输入是 `(ctx + cont)[-(max_len + 1):][:-1]`，按输入长度从长到短排成批，批内用 `pad_id` 右填充（因果注意力下右填充不影响前面的位置），`torch.no_grad()` 下前向，不开 autocast，`logits.float()` 后 `log_softmax`；第 i 条取最后 `len(cont)` 个位置，对数似然用 float64 求和，同时比较这些位置的 argmax 是否全等于 `cont`。结果按 `pairs` 的原顺序返回。函数不改变模型的 train/eval 状态，调用方保证是 eval。

`rolling_windows`（与框架 `make_disjoint_window(get_rolling_token_windows(..., context_len=1))` 逐窗相同，Task 2 有测试对着框架断言）：

```python
def rolling_windows(tokens, prefix_token, max_len):
    tokens = list(tokens)
    if not tokens:
        return []
    first = min(max_len, len(tokens))
    out = [([prefix_token], tokens[:first])]
    done = first
    while done < len(tokens):
        n = min(len(tokens) - done, max_len)
        end = done + n
        out.append((tokens[end - max_len - 1 : end - n], tokens[end - n : end]))
        done = end
    return out
```

`rolling_logprob`：`rolling_windows` 的每一窗交给 `continuation_logprobs`，返回对数似然之和；空序列返回 `0.0`。

- [ ] **Step 5: 运行，确认通过**

Run: `pytest tests/test_official_scoring.py`
Expected: `16 passed`

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml minids/official/__init__.py minids/official/scoring.py tests/test_official_scoring.py
git commit -m "feat(official): harness-independent log-likelihood scoring core"
```

---

### Task 2: 评测框架适配

**Files:**
- Create: `minids/official/harness.py`
- Test: `tests/test_official_harness.py`

**Interfaces:**
- Consumes: Task 1 的四个函数；`lm_eval.api.model.LM`、`lm_eval.simple_evaluate`、`lm_eval.__version__`。
- Produces（都在 `minids/official/harness.py`）:
  - `HARNESS_TASKS: tuple[str, ...] = ("blimp", "hellaswag", "wikitext")`
  - `class HeimiLM(LM)`：`__init__(self, model, tok, device: str, batch_size: int = 64)`，属性 `model`、`tok`、`device`、`batch_size`、`max_len`（取 `model.cfg.seq_len`）；方法 `loglikelihood(requests) -> list[tuple[float, bool]]`、`loglikelihood_rolling(requests) -> list[float]`、`generate_until(requests)`（抛 `NotImplementedError`）。
  - `harness_version() -> str`
  - `flatten_results(results: dict, tasks: Sequence[str]) -> dict[str, float]`
  - `run_harness(model, tasks: Sequence[str] = HARNESS_TASKS, limit: int | None = None, **kwargs) -> dict[str, float]`。`model` 是 `LM` 实例或框架的模型名（如 `"hf"`），`kwargs` 原样传给 `simple_evaluate`。

- [ ] **Step 1: 写失败的测试**

`tests/test_official_harness.py`：

```python
import math
from dataclasses import replace

import pytest
import torch

pytest.importorskip("lm_eval")  # 没装 official 额外依赖时整份跳过；本机必须是装了的，见完成标准

from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window  # noqa: E402

from minids.model.transformer import Transformer  # noqa: E402
from minids.official.harness import HARNESS_TASKS, HeimiLM, flatten_results, harness_version  # noqa: E402
from minids.official.scoring import continuation_logprobs, encode_pair, rolling_logprob, rolling_windows  # noqa: E402
from tests.helpers_data import tiny_tokenizer  # noqa: E402
from tests.helpers_train import CountingModel  # noqa: E402


class _NumTok:
    """把 "1 2 3" 编成 [1, 2, 3]：让测试能直接指定 token。"""

    def encode(self, text):
        return [int(x) for x in text.split()]


def _req(kind, *args):
    return Instance(request_type=kind, doc={}, arguments=args, idx=0)


@pytest.fixture
def lm(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    torch.manual_seed(0)
    return HeimiLM(Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size)).eval(), tok, "cpu")


def test_loglikelihood_scores_each_request_like_the_scoring_core(lm):
    reqs = [_req("loglikelihood", "The quick brown", " fox"), _req("loglikelihood", "", " The cat sat."), _req("loglikelihood", "A man is ", "sitting")]
    got = lm.loglikelihood(reqs)
    want = continuation_logprobs(lm.model, [encode_pair(lm.tok, *r.args) for r in reqs], "cpu", lm.max_len)
    assert len(got) == 3 and all(type(g[0]) is float and type(g[1]) is bool for g in got)
    assert all(math.isclose(g[0], w[0], rel_tol=1e-4, abs_tol=1e-4) and g[1] == w[1] for g, w in zip(got, want))


def test_empty_context_is_conditioned_on_the_document_separator():
    # CountingModel 认为"下一个 token = 当前 + 1"。空上下文若真的以 id 0 为条件，"1 2 3" 就是它的首选
    lm = HeimiLM(CountingModel(vocab_size=64), _NumTok(), "cpu")
    good, bad = lm.loglikelihood([_req("loglikelihood", "", "1 2 3"), _req("loglikelihood", "", "1 3 2")])
    assert good[1] is True and bad[1] is False and good[0] > bad[0]


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 128, 150])
def test_rolling_windows_are_the_harness_windows(n):
    tokens = list(range(100, 100 + n))
    theirs = [make_disjoint_window(w) for w in get_rolling_token_windows(token_list=tokens, prefix_token=0, max_seq_len=64, context_len=1)]
    assert rolling_windows(tokens, 0, 64) == [(list(a), list(b)) for a, b in theirs]


def test_loglikelihood_rolling_scores_the_whole_text_across_windows(lm):
    text = "The quick brown fox jumps over the lazy dog. " * 12
    ids = lm.tok.encode(text)
    assert len(ids) > lm.max_len  # 确实跨了窗口
    (got,) = lm.loglikelihood_rolling([_req("loglikelihood_rolling", text)])
    assert type(got) is float and math.isclose(got, rolling_logprob(lm.model, ids, "cpu", lm.max_len), rel_tol=1e-4)


CANNED = {
    "results": {
        "blimp": {"alias": "blimp", "name": "blimp", "sample_len": 134, "acc,none": 0.75, "acc_stderr,none": "N/A", "sample_count": {"acc,none": 134}},
        "blimp_anaphor_gender_agreement": {"alias": " - x", "acc,none": 1.0, "acc_stderr,none": "N/A"},
        "hellaswag": {"name": "hellaswag", "sample_len": 4, "acc,none": 0.25, "acc_stderr,none": 0.01, "acc_norm,none": 0.5, "acc_norm_stderr,none": 0.02},
        "wikitext": {
            "word_perplexity,none": 128.03, "word_perplexity_stderr,none": "N/A",
            "byte_perplexity,none": 2.588, "byte_perplexity_stderr,none": "N/A",
            "bits_per_byte,none": 1.372, "bits_per_byte_stderr,none": "N/A",
        },
    }
}


def test_flatten_keeps_the_numbers_of_the_requested_tasks():
    assert flatten_results(CANNED, HARNESS_TASKS) == {
        "blimp_acc": 0.75, "hellaswag_acc": 0.25, "hellaswag_acc_norm": 0.5,
        "wikitext_word_perplexity": 128.03, "wikitext_byte_perplexity": 2.588, "wikitext_bits_per_byte": 1.372,
    }


def test_flatten_names_a_task_the_harness_did_not_return():
    with pytest.raises(ValueError, match="wikitext"):
        flatten_results({"results": {"blimp": {"acc,none": 0.5}}}, ("blimp", "wikitext"))


def test_generation_is_not_offered_through_the_harness(lm):
    with pytest.raises(NotImplementedError):
        lm.generate_until([])


def test_harness_version_is_the_pinned_one():
    assert harness_version() == "0.4.13"
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_official_harness.py`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'minids.official.harness'`（不是 skipped；若显示 skipped，说明 Task 1 的安装没生效，回去查）

- [ ] **Step 3: 实现 `minids/official/harness.py`**

`HeimiLM.__init__` 先调 `super().__init__()`。`loglikelihood`：对每条请求 `encode_pair(self.tok, *req.args)`，一次交给 `continuation_logprobs(self.model, pairs, self.device, self.max_len, self.batch_size)`，把每个结果转成 `(float, bool)`。`loglikelihood_rolling`：对每条请求 `rolling_logprob(self.model, self.tok.encode(req.args[0]), self.device, self.max_len)`。`generate_until` 抛 `NotImplementedError`，消息说明代码补全走 `minids.official` 自己的评测。两个打分方法开头各记一条 `logging` 信息（请求条数），全量 BLiMP 有 13.4 万条请求，没有任何输出会像卡死。

`flatten_results`：对 `tasks` 里的每个任务取 `results["results"][task]`，没有就抛 `ValueError`（消息带任务名）；条目里键以 `",none"` 结尾、不含 `"_stderr"`、值是 `int` 或 `float`（不含 `bool`）的，记成 `f"{task}_{键逗号前的部分}"`。

`run_harness`：`lm_eval.simple_evaluate(model=model, tasks=list(tasks), limit=limit, bootstrap_iters=0, log_samples=False, **kwargs)` 的结果交给 `flatten_results`。它需要联网取数据，单元测试不覆盖，Task 9 用真实数据跑。

若本机环境里 `import lm_eval` 在 `-W error` 下出第三方的弃用警告，在测试文件顶部用 `pytestmark = pytest.mark.filterwarnings(...)` 精确忽略那一条并记入台账，不要全局关警告。

- [ ] **Step 4: 运行，确认通过**

Run: `pytest tests/test_official_harness.py -rs`
Expected: `14 passed`，没有 skipped

- [ ] **Step 5: 提交**

```bash
git add minids/official/harness.py tests/test_official_harness.py
git commit -m "feat(official): lm-evaluation-harness adapter for Heimi checkpoints"
```

---

### Task 3: 代码补全评测的计分

**Files:**
- Create: `minids/official/codebench.py`
- Test: `tests/test_official_codebench.py`

**Interfaces:**
- Consumes: 无（只用标准库）。
- Produces（都在 `minids/official/codebench.py`）:
  - `SAMPLE_SEED = 1337`
  - `first_code_line(text: str) -> tuple[str, bool]`：官方取行规则下的预测行，以及这一行是否已经写完（后面跟着换行）。
  - `edit_similarity(pred: str, gt: str) -> int`：0–100。
  - `lines_match(pred: str, gt: str) -> bool`
  - `repobench_prompt(row: dict) -> str`
  - `sample_rows(rows: Sequence[dict], limit: int | None, seed: int = SAMPLE_SEED) -> list[dict]`
  - `repobench_scores(complete: Callable[[str], str], rows: Sequence[dict]) -> dict`：`{"em": float, "es": float, "n": int}`。`complete` 收到提示，返回模型生成的原始文本。
  - `infill_scores(complete: Callable[[str, str], str], rows: Sequence[dict]) -> dict`：`{"em": float, "n": int}`。`complete` 收到 `(prompt, suffix)`，返回生成的文本。

- [ ] **Step 1: 写失败的测试**

`tests/test_official_codebench.py`：

```python
import pytest

from minids.official.codebench import (
    edit_similarity, first_code_line, infill_scores, lines_match, repobench_prompt, repobench_scores, sample_rows,
)

ROW = {
    "repo_name": "o/r", "file_path": "pkg/mod.py", "import_statement": "import os\nimport sys",
    "cropped_code": "def f():\n    x = 1\n\n\n\n\n    return x\n  \n", "next_line": "print(f())", "level": "2k",
}


def test_first_code_line_skips_blank_lines_and_comments():
    assert first_code_line("\n\n# a comment\n    x = 1\ny = 2\n") == ("    x = 1", True)


def test_first_code_line_skips_a_docstring_block():
    assert first_code_line('"""doc\nmore\n"""\nreturn 1\n') == ("return 1", True)


def test_a_one_line_docstring_opens_a_block_as_in_the_official_script():
    # 官方脚本只看行首的三引号就进入"多行注释"状态，不看同一行有没有收尾；照做才是同一把尺子
    assert first_code_line('"""one line"""\nx = 1\n"""\nz = 3\n') == ("z = 3", True)


def test_a_line_still_being_written_is_reported_as_incomplete():
    assert first_code_line("# note\nx = compute(") == ("x = compute(", False)
    assert first_code_line("x = 1\n") == ("x = 1", True)


def test_text_with_no_code_line_falls_back_to_its_first_line():
    assert first_code_line("# only a comment\n# another\n") == ("# only a comment", False)
    assert first_code_line("") == ("", False)


def test_edit_similarity_follows_fuzzywuzzy_ratio():
    assert edit_similarity("x = 1", "x = 1") == 100 and edit_similarity("", "") == 100
    assert edit_similarity("", "x = 1") == 0 and edit_similarity("x = 1", "") == 0
    assert edit_similarity("abef", "abcd") == 50  # 2 * 2 / 8
    assert edit_similarity("return a - b", "return a + b") == 92  # 2 * 11 / 24 = 0.9167


def test_lines_match_ignores_whitespace_differences_only():
    assert lines_match("  return a+b ", "return a+b") and lines_match("x  =  1", "x = 1")
    assert not lines_match("return a + b", "return a+b")


def test_prompt_is_the_official_in_file_part_with_blank_runs_collapsed():
    assert repobench_prompt(ROW) == "# Path: pkg/mod.py\nimport os\nimport sys\ndef f():\n    x = 1\n\n    return x\n"


def test_sampling_is_fixed_keeps_order_and_none_means_everything():
    rows = [{"i": i} for i in range(50)]
    a = sample_rows(rows, 10)
    assert a == sample_rows(rows, 10) and len(a) == 10 and a != rows[:10]
    assert [r["i"] for r in a] == sorted(r["i"] for r in a)
    assert sample_rows(rows, None) == rows and sample_rows(rows, 80) == rows


def test_repobench_scores_use_the_first_code_line_of_what_the_model_wrote():
    rows = [dict(ROW, next_line="print(f())"), dict(ROW, next_line="return 2"), dict(ROW, next_line="y = 3")]
    answers = iter(["\n# thinking\n  print(f())  \nmore\n", "return 3\n", ""])
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return next(answers)

    # 第 1 条：EM 命中，ES = round(100 * 20 / 24) = 83；第 2 条：ES = round(87.5) = 88；第 3 条：什么都没写，0
    assert repobench_scores(complete, rows) == {"em": 0.33333, "es": 57.0, "n": 3}
    assert seen == [repobench_prompt(r) for r in rows]


def test_infill_compares_the_first_generated_line_with_the_reference_line():
    rows = [
        {"task_id": "t0", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n"},
        {"task_id": "t1", "prompt": "p", "suffix": "s", "canonical_solution": "    x = 1\n"},
    ]
    calls = []

    def complete(prefix, suffix):
        calls.append((prefix, suffix))
        return ["return a + b\n    junk", "x = 2"][len(calls) - 1]

    assert infill_scores(complete, rows) == {"em": 0.5, "n": 2}
    assert calls == [("def f(a, b):\n", "\n"), ("p", "s")]


def test_scoring_nothing_is_an_error():
    with pytest.raises(ValueError):
        repobench_scores(lambda p: "", [])
    with pytest.raises(ValueError):
        infill_scores(lambda p, s: "", [])
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_official_codebench.py`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'minids.official.codebench'`

- [ ] **Step 3: 实现 `minids/official/codebench.py`**

`first_code_line` 的规则（与官方 `get_first_line_not_comment` 的 Python 分支同义，自己写，不照抄源码）：

1. 去掉文本开头的换行符，按 `"\n"` 切成行，状态 `in_block = False`。
2. 逐行看去掉首尾空白后的内容 `s`：`s` 为空则跳过；不在块内且 `s` 以 `"""` 或 `'''` 开头，则进入块并跳过；在块内且 `s` 以 `"""` 或 `'''` 结尾，则退出块并跳过；在块内则跳过；`s` 以 `#` 开头则跳过；否则这一行（原样，不去空白）就是预测行，它不是最后一段（下标小于行数减一）时"已写完"为真。
3. 没有找到就返回 `(第一行, False)`。

`edit_similarity`（`fuzzywuzzy.fuzz.ratio` 在没装 python-Levenshtein 时的纯 Python 路径）：两串相等返回 100；任一为空返回 0；否则 `int(round(100 * difflib.SequenceMatcher(None, gt, pred).ratio()))`，参数顺序是先 `gt` 后 `pred`，与官方 `eval.py` 的调用一致。

`lines_match`：`pred.split() == gt.split()`。

`repobench_prompt`：`f"# Path: {row['file_path']}\n{row['import_statement']}\n{row['cropped_code'].rstrip()}\n"` 再 `re.sub(r"\n{4,}", "\n\n", ...)`。

`sample_rows`：`limit` 为 `None` 或不小于行数时原样返回全部；否则 `random.Random(seed).sample(range(len(rows)), limit)` 排序后按下标取行。

`repobench_scores`：对每行 `pred = first_code_line(complete(repobench_prompt(row)))[0]`，与 `row["next_line"]` 算 `lines_match` 和 `edit_similarity`，返回两者的平均（各 `round(..., 5)`）和条数；空输入抛 `ValueError`。

`infill_scores`：对每行取 `complete(row["prompt"], row["suffix"])` 的第一行（第一个换行之前），与 `row["canonical_solution"]` 各自去首尾空白后比较是否相等；返回命中比例（`round(..., 5)`）和条数；空输入抛 `ValueError`。

- [ ] **Step 4: 运行，确认通过**

Run: `pytest tests/test_official_codebench.py`
Expected: `12 passed`

- [ ] **Step 5: 提交**

```bash
git add minids/official/codebench.py tests/test_official_codebench.py
git commit -m "feat(official): RepoBench and single-line infilling scoring without code execution"
```

---

### Task 4: 评测数据与黑米的补全函数

**Files:**
- Create: `minids/official/codedata.py`、`minids/official/completers.py`、`tests/helpers_official.py`
- Test: `tests/test_official_codedata.py`、`tests/test_official_completers.py`

**Interfaces:**
- Consumes: `first_code_line`（Task 3）；`minids.completion.{complete_middle, greedy_line}`；`huggingface_hub.hf_hub_download`；`tests.helpers_train.ScriptedModel`。
- Produces（`minids/official/codedata.py`）:
  - `REPOBENCH_REPO = "tianyang/repobench_python_v1.1"`、`REPOBENCH_FILE = "data/in_file-00000-of-00001-9721c68f6224a587.parquet"`、`REPOBENCH_LOCAL = "repobench_python_in_file.parquet"`
  - `INFILL_REPO = "loubnabnl/humaneval_infilling"`、`INFILL_FILE = "data/HumanEval-SingleLineInfilling.jsonl"`、`INFILL_LOCAL = "humaneval_single_line.jsonl"`
  - `download_dataset_file(repo: str, filename: str, dest: Path) -> Path`
  - `ensure_code_data(data_dir: Path) -> None`：两份文件缺哪份下哪份。
  - `load_repobench(path: Path) -> list[dict]`：每行只留 `repo_name, file_path, import_statement, cropped_code, next_line, level`。
  - `load_infill(path: Path) -> list[dict]`：每行只留 `task_id, prompt, suffix, canonical_solution`。
- Produces（`minids/official/completers.py`）:
  - `CODE_MAX_NEW_TOKENS = 64`
  - `greedy_text(model, tok, prompt_ids: list[int], max_new_tokens: int, device: str, done: Callable[[str], bool], n_special: int = 16) -> str`
  - `heimi_next_line(model, tok, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str], str]`
  - `heimi_infill(model, tok, device: str, mode: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str, str], str]`，`mode` 是 `"fim"` 或 `"prefix"`。
- Produces（`tests/helpers_official.py`，Task 5 到 Task 8 也用）:
  - `class SpyScripted(ScriptedModel)`：属性 `lengths`（每次 `generate` 收到的序列长度）、`first`（第一次收到的整段 id 列表）。
  - `write_code_data(d: Path) -> Path`：在 `d` 下写出 `REPOBENCH_LOCAL`（3 行）与 `INFILL_LOCAL`（3 行），返回 `d`。
  - `official_record(checkpoint: str, tokens_total: int, limit: int | None = None, code_limit: int | None = 500, **metrics) -> OfficialRecord`（Task 5 实现 `OfficialRecord` 后补上）。

- [ ] **Step 1: 写测试共用件与失败的测试**

`tests/helpers_official.py`（本任务先写前两项）：

```python
"""官方评测测试共用：记录提示长度的脚本模型、迷你代码评测数据、造记录的小函数。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from minids.official.codedata import INFILL_LOCAL, REPOBENCH_LOCAL
from tests.helpers_train import ScriptedModel


class SpyScripted(ScriptedModel):
    """记下每次 generate 收到的序列长度，和第一次收到的整段提示。"""

    def __init__(self, script, seq_len=64):
        super().__init__(script, seq_len)
        self.lengths: list[int] = []
        self.first: list[int] | None = None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.lengths.append(idx.shape[1])
        if self.first is None:
            self.first = idx[0].tolist()
        return super().generate(idx, max_new_tokens, temperature, top_k)


def write_code_data(d: Path) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    n = 3
    pq.write_table(
        pa.table({
            "repo_name": ["o/r"] * n, "file_path": [f"m{i}.py" for i in range(n)], "import_statement": ["import os"] * n,
            "cropped_code": ["def f():\n    x = 1\n"] * n, "next_line": ["    return x"] * n, "level": ["2k"] * n,
            "all_code": ["unused"] * n,
        }),
        d / REPOBENCH_LOCAL,
    )
    rows = [
        {"task_id": f"t{i}", "entry_point": "f", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n", "test": "unused"}
        for i in range(n)
    ]
    (d / INFILL_LOCAL).write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return d
```

`tests/test_official_codedata.py`：

```python
from pathlib import Path

import pytest

from minids.official import codedata
from minids.official.codedata import INFILL_LOCAL, REPOBENCH_LOCAL, load_infill, load_repobench
from tests.helpers_official import write_code_data


def test_download_fetches_once_and_leaves_nothing_behind_on_failure(tmp_path, monkeypatch):
    calls = []

    def fake(repo_id, filename, repo_type, local_dir):
        calls.append((repo_id, filename, repo_type))
        got = Path(local_dir) / "got.bin"
        got.write_bytes(b"data")
        return str(got)

    monkeypatch.setattr(codedata, "hf_hub_download", fake)
    dest = tmp_path / "eval" / "x.parquet"
    assert codedata.download_dataset_file("a/b", "data/x.parquet", dest) == dest and dest.read_bytes() == b"data"
    codedata.download_dataset_file("a/b", "data/x.parquet", dest)
    assert calls == [("a/b", "data/x.parquet", "dataset")]  # 已有就不再下

    def broken(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(codedata, "hf_hub_download", broken)
    with pytest.raises(OSError):
        codedata.download_dataset_file("a/b", "f", tmp_path / "eval" / "y.jsonl")
    assert sorted(p.name for p in (tmp_path / "eval").iterdir()) == ["x.parquet"]  # 没有半个文件，也没有临时目录


def test_ensure_code_data_downloads_only_what_is_missing(tmp_path, monkeypatch):
    asked = []

    def fake(repo, filename, dest):
        asked.append((repo, filename, Path(dest).name))
        Path(dest).write_bytes(b"x")
        return Path(dest)

    monkeypatch.setattr(codedata, "download_dataset_file", fake)
    (tmp_path / REPOBENCH_LOCAL).write_bytes(b"x")
    codedata.ensure_code_data(tmp_path)
    assert asked == [(codedata.INFILL_REPO, codedata.INFILL_FILE, INFILL_LOCAL)]


def test_loaders_return_only_the_fields_the_scorers_need(tmp_path):
    d = write_code_data(tmp_path)
    rb = load_repobench(d / REPOBENCH_LOCAL)
    assert len(rb) == 3 and rb[0] == {
        "repo_name": "o/r", "file_path": "m0.py", "import_statement": "import os",
        "cropped_code": "def f():\n    x = 1\n", "next_line": "    return x", "level": "2k",
    }
    inf = load_infill(d / INFILL_LOCAL)  # 文件末尾的空行不算一条
    assert len(inf) == 3 and inf[0] == {"task_id": "t0", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n"}
```

`tests/test_official_completers.py`：

```python
import pytest

from minids.official.codebench import first_code_line
from minids.official.completers import heimi_infill, heimi_next_line
from tests.helpers_data import tiny_tokenizer
from tests.helpers_official import SpyScripted
from tests.helpers_train import ScriptedModel


def test_next_line_stops_as_soon_as_a_code_line_is_complete(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    head = tok.encode("# note\n") + tok.encode("x = 1\n")
    m = ScriptedModel(head + tok.encode("y = 2\n" * 5))
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=40)("def f():\n")
    assert first_code_line(text) == ("x = 1", True)
    assert m.pos == len(head)  # 第一行代码写完就停，没有把预算用完


def test_next_line_keeps_only_the_tail_of_a_long_prompt(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("x = 1\n"), seq_len=32)
    heimi_next_line(m, tok, "cpu", max_new_tokens=8)("a = 1\n" * 200)
    assert m.lengths[0] == 32 - 8 and max(m.lengths) <= 32


def test_generation_budget_never_exceeds_half_the_context(tmp_path):
    # 默认 64 个新 token 对上下文只有 32 的迷你模型太多：预算收到一半，提示才不会被截成空的
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("a") * 100, seq_len=32)
    heimi_next_line(m, tok, "cpu")("a = 1\n" * 200)
    assert m.lengths[0] == 16 and m.pos == 16


def test_next_line_stops_at_a_special_token_and_never_returns_its_literal(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    body = tok.encode("x = 1")
    assert heimi_next_line(ScriptedModel(body + [0] + body), tok, "cpu", 40)("p") == "x = 1"
    assert heimi_next_line(ScriptedModel([3] + body), tok, "cpu", 40)("p") == ""  # 第一个 token 就是控制标记：什么都没写


def test_infill_fim_mode_builds_the_fim_prompt_within_the_context(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("return a + b\n"), seq_len=64)
    out = heimi_infill(m, tok, "cpu", "fim", max_new_tokens=16)("def f(a, b):\n" * 30, "\nprint(f(1, 2))\n" * 30)
    assert out == "return a + b" and max(m.lengths) <= 64
    assert m.first[0] == 1 and m.first[-1] == 2 and 3 in m.first  # <|fim_prefix|> … <|fim_suffix|> … <|fim_middle|>
    assert len(m.first) == 30 + 15 + 3  # 余量 45 个 token：前缀占三分之二，后缀占三分之一


def test_infill_prefix_mode_ignores_the_suffix(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("return a + b\n"), seq_len=64)
    out = heimi_infill(m, tok, "cpu", "prefix", max_new_tokens=16)("def f(a, b):\n", "SUFFIX")
    assert out == "return a + b" and m.first == tok.encode("def f(a, b):\n")


def test_unknown_infill_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        heimi_infill(ScriptedModel([0]), tiny_tokenizer(tmp_path), "cpu", "both")
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_official_codedata.py tests/test_official_completers.py`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'minids.official.codedata'`

- [ ] **Step 3: 实现 `minids/official/codedata.py`**

`download_dataset_file`：`dest` 已存在直接返回；否则在 `dest.parent` 下开 `tempfile.TemporaryDirectory`，`hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=tmp)`，成功后 `shutil.move` 到 `dest`（与 `minids/eval/hellaswag.py` 的 `download_hellaswag` 同一做法；失败时临时目录连同半个文件一起消失）。`ensure_code_data`：对两份文件各自 `if not (data_dir / LOCAL).exists(): download_dataset_file(REPO, FILE, data_dir / LOCAL)`，通过模块里的名字调用以便测试替换。`load_repobench`：`pq.read_table(path, columns=[六个字段])` 转成行字典列表。`load_infill`：逐行 `json.loads`，跳过空行，只留四个字段。

- [ ] **Step 4: 实现 `minids/official/completers.py`**

`greedy_text`：从 `prompt_ids` 起每步 `model.generate(idx, 1, temperature=0.0)` 取一个 token；生成的 id 小于 `n_special` 就停（这个 id 不计入结果）；每加一个 token 就解码全部已生成的 id，`done(text)` 为真就停；用完 `max_new_tokens` 也停。返回解码文本。带 `torch.no_grad()`。

两个工厂函数都先定预算：`budget = max(1, min(max_new_tokens, model.cfg.seq_len // 2))`。

`heimi_next_line` 返回的函数：`ids = tok.encode(prompt)[-(seq_len - budget):]`，`greedy_text(..., budget, device, done=lambda t: first_code_line(t)[1])`，返回生成的原始文本（取行由计分器做）。

`heimi_infill`：`mode` 不是 `"fim"` 或 `"prefix"` 时立刻抛 `ValueError`（消息含 `mode`）。`"fim"` 返回的函数调 `complete_middle(model, tok, prefix, suffix, budget, device, max_prefix_tokens, max_suffix_tokens)`，其中 `room = seq_len - budget - 3`、`max_prefix_tokens = room * 2 // 3`、`max_suffix_tokens = room - max_prefix_tokens`。`"prefix"` 返回的函数忽略后缀，`ids = tok.encode(prefix)[-(seq_len - budget):]`，调 `greedy_line(model, tok, ids, budget, device)`。

- [ ] **Step 5: 运行，确认通过**

Run: `pytest tests/test_official_codedata.py tests/test_official_completers.py`
Expected: `10 passed`

- [ ] **Step 6: 提交**

```bash
git add minids/official/codedata.py minids/official/completers.py tests/helpers_official.py tests/test_official_codedata.py tests/test_official_completers.py
git commit -m "feat(official): code benchmark data loading and Heimi line completers"
```

---

### Task 5: 结果记录与参照文件

**Files:**
- Create: `minids/official/results.py`
- Modify: `tests/helpers_official.py`（补 `official_record`）
- Test: `tests/test_official_results.py`

**Interfaces:**
- Consumes: `minids.logging_io.{append_jsonl, read_jsonl}`。
- Produces（都在 `minids/official/results.py`，本模块不 import torch）:
  - `OFFICIAL_FILE = "official.jsonl"`
  - `@dataclass class OfficialRecord`，字段按顺序：`checkpoint: str, stage: str, global_step: int, tokens_total: int, params_non_embedding: int, limit: int | None, code_limit: int | None, metrics: dict[str, float], harness_version: str, time: str`
  - `append_record(run_dir: Path, rec: OfficialRecord) -> None`
  - `read_records(run_dir: Path) -> list[dict]`：文件或目录不存在返回 `[]`。
  - `latest_per_checkpoint(records: list[dict]) -> list[dict]`
  - `pending_checkpoints(run_dir: Path, limit: int | None, code_limit: int | None) -> list[Path]`
  - `load_reference(path: Path) -> dict`、`save_reference(path: Path, name: str, entry: dict) -> None`

- [ ] **Step 1: 写失败的测试**

`tests/helpers_official.py` 末尾加：

```python
def official_record(checkpoint, tokens_total, limit=None, code_limit=500, **metrics):
    from minids.official.results import OfficialRecord

    return OfficialRecord(
        checkpoint=checkpoint, stage=checkpoint.split("_")[0], global_step=tokens_total // 100, tokens_total=tokens_total,
        params_non_embedding=1000, limit=limit, code_limit=code_limit, metrics=metrics or {"blimp_acc": 0.5},
        harness_version="0.4.13", time="2026-11-09T12:00:00-05:00",
    )
```

`tests/test_official_results.py`：

```python
import sys
from dataclasses import asdict

from minids.official.results import (
    append_record, latest_per_checkpoint, load_reference, pending_checkpoints, read_records, save_reference,
)
from tests.helpers_official import official_record


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
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_official_results.py`
Expected: `ModuleNotFoundError: No module named 'minids.official.results'`

- [ ] **Step 3: 实现 `minids/official/results.py`**

`append_record` 用 `append_jsonl(run_dir / OFFICIAL_FILE, rec)`；`read_records` 在文件不存在时返回 `[]`，否则 `read_jsonl`。

`latest_per_checkpoint`：按 `checkpoint` 分组；组内取 `(limit is None, code_limit != 0, 写入顺序)` 最大的一条，也就是先要框架任务全量的，再要带代码评测的，都一样就要最后写入的；结果按 `(tokens_total, global_step)` 升序。

`pending_checkpoints`：`run_dir / "ckpt"` 下名字以 `_final.pt` 结尾的文件里，去掉在 `read_records` 中已有 `checkpoint`、`limit`、`code_limit` 三者都相同的记录的；按修改时间升序，同一时刻按名字；目录不存在返回 `[]`。

`load_reference`：文件不存在、不是合法 JSON、或顶层不是对象，都返回 `{}`。`save_reference`：读出已有内容，`data[name] = entry`，写到同目录的临时文件（`ensure_ascii=False, indent=2`，UTF-8）后 `os.replace`，目录不存在先建。

已核实：`minids/__init__.py` 是空的，`minids/logging_io.py` 只用标准库，`read_jsonl` 对不存在的文件返回 `[]`、对末尾写了一半的行跳过、对中间的坏行抛错。所以本模块只要自己不 import torch，`test_results_module_does_not_pull_in_torch` 就能过。

- [ ] **Step 4: 运行，确认通过**

Run: `pytest tests/test_official_results.py`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
git add minids/official/results.py tests/helpers_official.py tests/test_official_results.py
git commit -m "feat(official): result records, pending-checkpoint selection and reference file"
```

---

### Task 6: 评一个检查点，评参照模型

**Files:**
- Create: `minids/official/run.py`、`minids/official/reference.py`
- Test: `tests/test_official_run.py`

**Interfaces:**
- Consumes: Task 2 的 `HeimiLM`、`run_harness`、`harness_version`；Task 3 的 `sample_rows`、`repobench_scores`、`infill_scores`；Task 4 的数据读取、常量与两个补全工厂；Task 5 的 `OfficialRecord`；`load_model`、`HeimiTokenizer`、`now_iso`。
- Produces（`minids/official/run.py`）:
  - `CODE_LIMIT = 500`、`REPO_ROOT = Path(__file__).resolve().parents[2]`
  - `code_metrics(next_line: Callable[[str], str], infill: dict[str, Callable[[str, str], str]], data_dir: Path, code_limit: int | None) -> dict[str, float]`：键 `repobench_em`、`repobench_es`、`repobench_n`、`infill_n`，以及 `infill` 里每个名字对应的 `infill_em_<名字>`。
  - `evaluate_checkpoint(ckpt: Path, device: str, data_dir: Path, limit: int | None = None, code_limit: int | None = CODE_LIMIT, harness: Callable | None = None) -> OfficialRecord`。`code_limit == 0` 表示不跑代码评测。`harness` 默认是 `run_harness`，调用形式 `harness(lm, limit=limit)`。
- Produces（`minids/official/reference.py`）:
  - `GPT2_ID = "openai-community/gpt2"`、`GPT2_LABEL = "GPT-2 small"`
  - `load_gpt2(device: str) -> tuple[object, object, int]`：`(模型, 分词器, 非嵌入参数量)`，函数体内 import transformers。
  - `hf_next_line(hf_model, hf_tok, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str], str]`
  - `hf_infill_prefix(hf_model, hf_tok, device: str, max_new_tokens: int = CODE_MAX_NEW_TOKENS) -> Callable[[str, str], str]`
  - `evaluate_reference(device: str, data_dir: Path, limit: int | None = None, code_limit: int | None = CODE_LIMIT, harness: Callable | None = None, loader: Callable | None = None) -> dict`：返回 `{"label", "model_id", "params_non_embedding", "limit", "code_limit", "metrics", "harness_version", "time"}`。`harness` 的调用形式 `harness("hf", limit=limit, model_args=f"pretrained={GPT2_ID}", device=device, batch_size=64)`；`loader` 默认是 `load_gpt2`。

- [ ] **Step 1: 写失败的测试**

`tests/test_official_run.py`：

```python
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from minids.official.reference import GPT2_ID, evaluate_reference, hf_infill_prefix, hf_next_line  # noqa: E402
from minids.official.run import evaluate_checkpoint  # noqa: E402
from tests.helpers_data import tiny_tokenizer  # noqa: E402
from tests.helpers_official import write_code_data  # noqa: E402


@pytest.fixture
def ckpt(dash_runs):
    return dash_runs[0] / "heimi" / "ckpt" / "smoke0_final.pt"


def test_checkpoint_record_carries_training_state_and_every_metric_group(ckpt, tmp_path):
    calls = []

    def fake_harness(lm, limit=None, **kwargs):
        calls.append((type(lm).__name__, limit))
        return {"blimp_acc": 0.5, "hellaswag_acc_norm": 0.25, "wikitext_bits_per_byte": 2.0}

    rec = evaluate_checkpoint(ckpt, "cpu", write_code_data(tmp_path / "eval"), limit=7, code_limit=2, harness=fake_harness)
    assert (rec.checkpoint, rec.stage, rec.global_step, rec.tokens_total) == ("smoke0_final.pt", "smoke0", 2, 512)
    assert rec.limit == 7 and rec.code_limit == 2 and rec.params_non_embedding > 0 and rec.harness_version == "0.4.13"
    assert calls == [("HeimiLM", 7)]
    m = rec.metrics
    assert m["blimp_acc"] == 0.5 and m["repobench_n"] == 2 and m["infill_n"] == 2
    assert all(0.0 <= m[k] <= 1.0 for k in ("repobench_em", "infill_em_fim", "infill_em_prefix")) and 0.0 <= m["repobench_es"] <= 100.0


def test_code_benchmarks_can_be_skipped_and_then_need_no_data(ckpt, tmp_path):
    rec = evaluate_checkpoint(ckpt, "cpu", tmp_path / "missing", code_limit=0, harness=lambda lm, **kw: {"blimp_acc": 0.5})
    assert rec.metrics == {"blimp_acc": 0.5} and rec.code_limit == 0


def test_missing_code_data_is_reported_by_file_name(ckpt, tmp_path):
    with pytest.raises(FileNotFoundError, match="repobench_python_in_file.parquet"):
        evaluate_checkpoint(ckpt, "cpu", tmp_path / "missing", harness=lambda lm, **kw: {})


def _stub_hf(tmp_path):
    """像 transformers 的模型与分词器那样说话的替身：generate 把固定的新 token 接在输入后面。"""
    tok = tiny_tokenizer(tmp_path)
    new = tok.encode("x = 1\ny = 2\n")
    seen = {}

    def generate(input_ids, **kwargs):
        seen["n"], seen["kwargs"] = input_ids.shape[1], kwargs
        return torch.cat([input_ids, torch.tensor([new])], dim=1)

    hf_model = SimpleNamespace(config=SimpleNamespace(n_positions=32), generate=generate)
    hf_tok = SimpleNamespace(encode=tok.encode, decode=lambda ids, skip_special_tokens=True: tok.decode(list(ids)), eos_token_id=0)
    return hf_model, hf_tok, seen


def test_hf_completers_send_the_prompt_tail_and_return_only_the_new_text(tmp_path):
    hf_model, hf_tok, seen = _stub_hf(tmp_path)
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)("a = 1\n" * 100) == "x = 1\ny = 2\n"
    assert seen["n"] == 32 - 8 and seen["kwargs"]["do_sample"] is False and seen["kwargs"]["max_new_tokens"] == 8
    assert hf_infill_prefix(hf_model, hf_tok, "cpu", max_new_tokens=8)("def f():\n", "IGNORED") == "x = 1"


def test_reference_entry_has_no_fim_number_and_runs_the_harness_on_gpt2(tmp_path):
    hf_model, hf_tok, _ = _stub_hf(tmp_path)
    calls = []

    def fake_harness(model, limit=None, **kwargs):
        calls.append((model, limit, kwargs.get("model_args")))
        return {"blimp_acc": 0.8}

    entry = evaluate_reference("cpu", write_code_data(tmp_path / "eval"), limit=5, code_limit=2, harness=fake_harness, loader=lambda device: (hf_model, hf_tok, 85_000_000))
    assert calls == [("hf", 5, f"pretrained={GPT2_ID}")]
    assert (entry["label"], entry["model_id"], entry["params_non_embedding"], entry["limit"], entry["code_limit"]) == ("GPT-2 small", GPT2_ID, 85_000_000, 5, 2)
    m = entry["metrics"]
    assert m["blimp_acc"] == 0.8 and m["repobench_n"] == 2 and "infill_em_prefix" in m and "infill_em_fim" not in m
    assert entry["harness_version"] == "0.4.13" and entry["time"]
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_official_run.py`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'minids.official.reference'`

- [ ] **Step 3: 实现 `minids/official/run.py`**

`code_metrics`：两份数据文件任一不存在就抛 `FileNotFoundError`，消息带完整路径并提示先运行 `official_eval.py`（它会下载）。读入后各自 `sample_rows(rows, code_limit)`，`repobench_scores(next_line, ...)` 给出三个 `repobench_*`，对 `infill` 的每个 `(名字, 函数)` 调 `infill_scores` 给出 `infill_em_<名字>`，`infill_n` 取条数。

`evaluate_checkpoint`：`model, meta = load_model(ckpt, "cpu")` 后 `model.to(device)`；分词器路径取 `meta["ladder_config"]["data"]["tokenizer"]`，相对路径相对 `REPO_ROOT`；`harness` 为 `None` 时在函数内 `from minids.official.harness import run_harness`；`metrics = dict(harness(HeimiLM(model, tok, device), limit=limit))`；`code_limit != 0` 时并入 `code_metrics(heimi_next_line(model, tok, device), {"fim": heimi_infill(model, tok, device, "fim"), "prefix": heimi_infill(model, tok, device, "prefix")}, data_dir, code_limit)`；`stage`、`global_step`、`tokens_total` 取自 `meta["state"]`，`params_non_embedding = model.count_params()`，`harness_version()`，`now_iso()`。`HeimiLM` 与 `harness_version` 在函数内 import，这样 `import minids.official.run` 不要求装了框架。

- [ ] **Step 4: 实现 `minids/official/reference.py`**

`load_gpt2`：函数体内 `from transformers import AutoModelForCausalLM, AutoTokenizer`，加载 `GPT2_ID`，`.to(device).eval()`；非嵌入参数量 = 全部参数量减去 `model.transformer.wte` 与 `model.transformer.wpe` 的参数量（GPT-2 的输出层与 `wte` 共享权重，只算一次）。

`hf_next_line` 返回的函数：`budget = max(1, min(max_new_tokens, n_positions // 2))`，`ids = hf_tok.encode(prompt)[-(n_positions - budget):]`，`out = hf_model.generate(torch.tensor([ids], device=device), max_new_tokens=budget, do_sample=False, pad_token_id=hf_tok.eos_token_id)`，返回 `hf_tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)`。`n_positions` 取 `hf_model.config.n_positions`。带 `torch.no_grad()`。`hf_infill_prefix` 返回的函数忽略后缀，用同样的方式生成后取第一个换行之前的部分。

`evaluate_reference`：`harness` 为 `None` 时在函数内 import `run_harness`；按 Interfaces 里的形式调用 `harness`；`code_limit != 0` 时 `loader(device)` 取模型，并入 `code_metrics(hf_next_line(...), {"prefix": hf_infill_prefix(...)}, data_dir, code_limit)`；`code_limit == 0` 时也要 `loader(device)` 取参数量。

- [ ] **Step 5: 运行，确认通过**

Run: `pytest tests/test_official_run.py -rs`
Expected: `5 passed`，没有 skipped

- [ ] **Step 6: 提交**

```bash
git add minids/official/run.py minids/official/reference.py tests/test_official_run.py
git commit -m "feat(official): evaluate one checkpoint or the GPT-2 reference into a record"
```

---

### Task 7: 命令行入口

**Files:**
- Create: `official_eval.py`
- Test: `tests/test_official_cli.py`

**Interfaces:**
- Consumes: `evaluate_checkpoint`、`CODE_LIMIT`（Task 6）、`evaluate_reference`（Task 6）、`ensure_code_data`（Task 4）、`append_record`、`pending_checkpoints`、`save_reference`（Task 5）、`minids.gpu.gpu_status`、`minids.train.checkpoint.CheckpointError`。
- Produces（`official_eval.py`，以下名字都在模块顶层，测试会替换它们）:
  - `evaluate_checkpoint`、`evaluate_reference`、`ensure_code_data`、`gpu_status`（从各自模块 import 进来）
  - `harness_available() -> bool`：`importlib.util.find_spec("lm_eval") is not None`
  - `BUSY_GPU_FRAC = 0.5`
  - `main(argv: Sequence[str] | None = None) -> int`：0 全部成功；1 至少一个检查点评测失败；2 用法或前提不满足。

命令行参数：`--run NAME` 与 `--reference gpt2` 二选一且必须给一个；`--runs DIR`（默认 `runs`）；`--ckpt NAME`（可重复，只接受文件名）；`--device {cuda,cpu}`（默认有 CUDA 用 `cuda`）；`--limit N`；`--code-limit N`（默认 500）；`--code-all`；`--skip-code`；`--force`；`--allow-busy-gpu`；`--data-dir DIR`（默认仓库根目录下 `data/eval`）；`--reference-out FILE`（默认仓库根目录下 `docs/official/reference.json`）。

- [ ] **Step 1: 写失败的测试**

`tests/test_official_cli.py`：

```python
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
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_official_cli.py`
Expected: 收集阶段 `ModuleNotFoundError: No module named 'official_eval'`

- [ ] **Step 3: 实现 `official_eval.py`**

文件头的文档字符串写用法（设计说明第 7 节的三条命令）。给用户看的提示（没有待评测的检查点、显卡被占、安装命令、坏检查点的名字、用法错误）一律 `print(..., file=sys.stderr)`，不走 `logging`：`logging.basicConfig` 只在进程里第一次调用时绑定标准错误，之后的测试用 `capsys` 就抓不到了；`logging` 只用来报进度。`main` 的顺序：

1. 解析参数（`--run` 与 `--reference` 放进 `add_mutually_exclusive_group(required=True)`，`--reference` 的 `choices=["gpt2"]`）。`code_limit`：`--skip-code` 为 0，`--code-all` 为 `None`，否则 `--code-limit` 的值。日志用 `logging.basicConfig` 打到标准错误。`os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")`。
2. `harness_available()` 为假：标准错误打印"没有安装官方评测依赖，先运行：uv pip install -e ".[official]""，返回 2。
3. 设备是 `cuda` 且没给 `--allow-busy-gpu`：`s = gpu_status()`，若 `s` 不为空且 `s["mem_used_gb"] > BUSY_GPU_FRAC * s["mem_total_gb"]`，打印显卡已用多少、多半是训练在跑、三条出路（等训练停下、`--device cpu`、`--allow-busy-gpu`），返回 2。
4. `--reference`：`code_limit != 0` 时 `ensure_code_data(data_dir)`；`entry = evaluate_reference(device, data_dir, limit=limit, code_limit=code_limit)`；`save_reference(reference_out, "gpt2", entry)`；标准输出打印这一条的 JSON；返回 0。
5. `--run`：`run_dir = runs / run`，`run_dir / "ckpt"` 不是目录则报错返回 2。给了 `--ckpt`：每个名字必须等于自己的 `Path(name).name` 且对应文件存在，否则报出这个名字返回 2；目标就是这些文件（已有同设置记录的也评，等同 `--force`）。没给 `--ckpt`：`--force` 时目标是全部 `*_final.pt`（按 `pending_checkpoints` 的排序），否则是 `pending_checkpoints(run_dir, limit, code_limit)`；为空则在标准错误打印"没有待评测的检查点（都评过了；加 --force 重评）"返回 0。
6. `code_limit != 0` 时 `ensure_code_data(data_dir)`。逐个 `evaluate_checkpoint(ckpt, device, data_dir, limit=limit, code_limit=code_limit)`；成功就 `append_record` 并在标准输出打印一行 JSON（`ensure_ascii=False`）；抛 `CheckpointError`、`FileNotFoundError`、`RuntimeError` 时在标准错误打印检查点名和原因，继续下一个。有失败返回 1，否则 0。

- [ ] **Step 4: 运行，确认通过**

Run: `pytest tests/test_official_cli.py`
Expected: `8 passed`

- [ ] **Step 5: 提交**

```bash
git add official_eval.py tests/test_official_cli.py
git commit -m "feat(official): official_eval.py command line with busy-GPU guard"
```

---

### Task 8: 窗口的第五个区域

**Files:**
- Modify: `dashboard/data.py`、`dashboard/server.py`、`dashboard/cli.py`、`dashboard/static/index.html`
- Test: `tests/test_dashboard_official.py`

**Interfaces:**
- Consumes: `OFFICIAL_FILE`、`latest_per_checkpoint`、`load_reference`（Task 5）；`dashboard.data.read_log`、`safe_run_dir`；`tests.helpers_official.official_record`。
- Produces:
  - `dashboard.data.official(run_dir: Path, reference_path: Path | None) -> dict`：`{"rows": [...], "reference": [...]}`。`rows` 的每项是 `latest_per_checkpoint` 的记录加一个 `comparable: bool`；`reference` 的每项是 `{"name", "label", "params_non_embedding", "limit", "code_limit", "metrics"}`。
  - `dashboard.server.create_app(runs_dir, worker=None, repo_root=None, reference_path: Path | None = None)`，新路由 `GET /api/runs/{run}/official`。
  - `dashboard.cli.REFERENCE_FILE = REPO_ROOT / "docs" / "official" / "reference.json"`，`build` 把它传给 `create_app`。
  - 页面：`<section id="official">`，标题"官方评测"，表体 `id="official-rows"`。

- [ ] **Step 1: 写失败的测试**

`tests/test_dashboard_official.py`：

```python
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
```

- [ ] **Step 2: 运行，确认失败**

Run: `pytest tests/test_dashboard_official.py`
Expected: 8 个失败，前几个是 `AttributeError: module 'dashboard.data' has no attribute 'official'`

- [ ] **Step 3: 实现后端**

`dashboard/data.py` 加 `official`：`rows = latest_per_checkpoint(read_log(run_dir, OFFICIAL_FILE))`（`read_log` 对中间损坏的文件抛 `LogUnreadable`，对不存在的文件返回空，与其它日志一致；若 `read_log` 对不存在的文件不是返回空，先判断文件存在）；`reference` 由 `load_reference(reference_path)`（`reference_path` 为 `None` 时为空）转成列表，只收有 `metrics` 字典的条目，每项取 Interfaces 里的六个键；`comparable`：没有参照时为真，否则当且仅当这一行的 `limit` 与 `code_limit` 都等于第一条参照的。不要把 `OFFICIAL_FILE` 加进 `LOG_FILES`：那个元组决定"训练最后一次更新的时间"，官方评测不是训练活动。

`dashboard/server.py`：`create_app` 加第四个参数 `reference_path`，加路由 `GET /api/runs/{run}/official` 返回 `data.official(data.safe_run_dir(runs_dir, run), reference_path)`。`dashboard/cli.py`：加 `REFERENCE_FILE`，`build` 里 `create_app(opts.runs, repo_root=REPO_ROOT, reference_path=REFERENCE_FILE)`。

- [ ] **Step 4: 实现页面**

动手前先读 `dataviz` 与 `frontend-design` 两个技能（与计划四相同）。在"成长总览 / 当前状态"那一格之后、"成长样本墙"之前加：

- `<section id="official">`，`<h2>官方评测</h2>`，一句说明（标准评测框架下的公开评测；第一行是同机同法测的 GPT-2 small）。
- 一张表，沿用页面已有的表格样式（`table-scroll` 那一套）。表头：模型、参数量、BLiMP、HellaSwag、WikiText bits/byte、RepoBench EM、RepoBench ES、单行填空（FIM）、单行填空（仅前缀）。表体 `id="official-rows"`。
- 表下一句固定的读法：BLiMP 的随机水平是 50%，HellaSwag 是 25%；bits/byte 越低越好，其余越高越好；单行填空是精确匹配，不是官方的 pass@1；带"抽样"标记的行样本数与参照不同，不能直接比。
- 脚本：`state.official = new Map()`；`refresh()` 里与 `overview` 并列取 `${base}/official` 存进去；`renderOfficial()` 仿照 `renderEvalTable()`（同样的签名缓存、同样只用 `el(...)` 生成文本节点），先出参照行（模型一栏写 `label` 加"（参照）"），再出各检查点行（模型一栏写 `stage`，检查点不是 `<stage>_final.pt` 时改写文件名；`comparable` 为假时后面加"（抽样）"）。数值取自 `metrics`：`blimp_acc`、`hellaswag_acc_norm`、`repobench_em`、`infill_em_fim`、`infill_em_prefix` 用已有的百分比格式，`wikitext_bits_per_byte` 三位小数，`repobench_es` 一位小数，缺的写"—"。`rows` 与 `reference` 都为空时表体只有一行："还没有官方评测结果。训练停着的时候运行 python official_eval.py --run <run 名>。"；`render()` 里调用它。

- [ ] **Step 5: 运行，确认通过**

Run: `pytest tests/test_dashboard_official.py tests/test_dashboard_frontend.py tests/test_dashboard_server.py`
Expected: 全部通过（新文件 8 个；已有的前端与接口测试不受影响，特别是"模型文本不作为 HTML 插入"那一条）

- [ ] **Step 6: 提交**

```bash
git add dashboard/data.py dashboard/server.py dashboard/cli.py dashboard/static/index.html tests/test_dashboard_official.py
git commit -m "feat(dashboard): official evaluation table with the GPT-2 reference row"
```

---

### Task 9: 实机验证与文档

**Files:**
- Create: `docs/official-eval.md`、`docs/official/reference.json`（由命令生成）、`docs/dashboard/official.jpg`
- Modify: `docs/dashboard.md`

**Interfaces:**
- Consumes: 前八个任务的全部产出；`runs/heimi/ckpt/stage0_final.pt`；显卡。

开始前看一眼显卡：`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`。已用显存超过一半说明用户的训练在跑：不要抢显卡，本任务的三条评测命令全部改成 `--device cpu --limit 20 --code-limit 20` 只验证通路，参照文件写到临时位置不提交，文档里"首次实测"一节写明全量结果待训练结束后由用户运行同一命令补上，并在交付说明里把这件事放在最前面。

- [ ] **Step 1: 小样本通路验证**

Run: `PYTHONUTF8=1 .venv/Scripts/python.exe official_eval.py --reference gpt2 --limit 20 --code-limit 20 --reference-out <临时目录>/reference.json`
Expected: 退出码 0。首次运行会下载 GPT-2（约 550 MB）与两份代码评测数据（约 158 MB，落在 `data/eval/`）。打印的 JSON 里 12 个指标少一个 `infill_em_fim`，其余都是有限的数；`blimp_acc` 明显高于 0.5。

Run: `PYTHONUTF8=1 .venv/Scripts/python.exe official_eval.py --run heimi --ckpt stage0_final.pt --limit 20 --code-limit 20`
Expected: 退出码 0，`runs/heimi/official.jsonl` 多一行，12 个指标齐全；`wikitext_bits_per_byte` 在 1 到 4 之间（级 0 的 web 验证损失 4.47 奈特每 token，折合约 1.8 bits/byte，WikiText 会略高）。若它大得离谱，先查 `HeimiLM` 用的分词器是不是检查点里记的那一个。

记下两条命令各自的耗时。

- [ ] **Step 2: 全量参照**

Run: `PYTHONUTF8=1 .venv/Scripts/python.exe official_eval.py --reference gpt2`
Expected: 退出码 0，生成 `docs/official/reference.json`。把数字和设计说明第 9 节的范围对一下：`blimp_acc` ≥ 0.75、`hellaswag_acc_norm` 在 0.28–0.34、`wikitext_word_perplexity` 在 30–45。落在范围外先查环境与数据，不要改范围；查不出原因就停下来把数字和怀疑的原因告诉用户。

- [ ] **Step 3: 全量评级 0**

Run: `PYTHONUTF8=1 .venv/Scripts/python.exe official_eval.py --run heimi`
Expected: 退出码 0；它会评 `runs/heimi/ckpt/` 下每个 `*_final.pt`（当前只有 `stage0_final.pt`；若用户已经训出了更多，一并评完）。记下每个检查点的耗时。

再跑一次同样的命令，Expected: 标准错误出现"没有待评测的检查点"，退出码 0。

- [ ] **Step 4: 看窗口**

启动 `python dashboard.py --runs runs/ --no-browser`，在浏览器里看第五个区域（应用内预览只从会话根目录读 `.claude/launch.json`，做法见计划四 Task 6；用完删掉那个文件）。逐条核对：第一行是 GPT-2 small（参照）；级 0 一行排在后面；Step 1 留下的抽样记录没有盖掉全量记录；数字与 `official.jsonl`、`reference.json` 一致；窄窗口下表格横向滚动而不是撑破页面；深色与浅色下都能读；换到一个没有官方结果的 run（`smoke`）时显示那句运行提示。截一张图存成 `docs/dashboard/official.jpg`。

- [ ] **Step 5: 写文档**

`docs/official-eval.md`，面向用户，中文，内容按这个顺序：一句话说明这是什么；怎么运行（三条命令、每个选项、要多久：用 Step 1–3 记下的实测耗时）；每个数字是什么、随机水平是多少、哪个方向好；哪些是官方指标、哪些不是、为什么不执行代码；为什么只和 GPT-2 这一行比而不和排行榜比（设计说明第 4 节）；训练在跑的时候怎么办；"首次实测"一节放 GPT-2 与级 0 的全量数字表，并用一两句话说清级 0 现在在什么位置（哪些接近随机，哪些已经明显高于随机）。`docs/dashboard.md` 补一段第五个区域并引用截图。

- [ ] **Step 6: 全量测试并提交**

Run: `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q -W error -rs`
Expected: 全部通过，`tests/test_official_*.py` 与 `tests/test_dashboard_official.py` 没有 skipped。

```bash
git add docs/official-eval.md docs/official/reference.json docs/dashboard/official.jpg docs/dashboard.md
git commit -m "docs(official): usage guide, GPT-2 reference numbers and first measured results"
```

---

## 完成标准

1. `pytest`（带 `-W error`）全部通过，官方评测相关的测试没有被跳过的。
2. 不装 `official` 额外依赖时 `import minids.official.results`、`import official_eval`、`import dashboard.server` 都成功；`minids/train/`、`train.py`、`evaluate.py`、`minids/eval/` 在本计划的全部提交里没有改动（`git diff --stat <起点>..HEAD -- minids/train train.py evaluate.py minids/eval` 为空）。
3. `docs/official/reference.json` 里有 GPT-2 small 的全量结果，三个框架指标落在设计说明第 9 节的范围内。
4. `runs/heimi/official.jsonl` 里有级 0 的全量记录，窗口第五个区域里它和 GPT-2 并排显示。
5. `docs/official-eval.md` 写明了运行方法、实测耗时、每个数字的读法，以及哪些指标不是官方指标。
6. 若 Task 9 因训练占着显卡只做了小样本通路验证，第 3、4 条改为"命令在小样本下跑通"，并在交付说明的第一句话里告诉用户全量结果要等训练结束后运行哪一条命令。
