# 黑米 计划三：训练、评测与任务阶段 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把计划一的模型与增长算子和计划二的数据接成一条能按 `ladder.yaml` 逐级训练、级间自动增长并验证、随时可中断恢复、持续写出窗口所需日志的训练流水线，附带全部评测、文本生成、模型导出和命令行 FIM 补全工具，并在真实数据上跑完级 0。

**Architecture:** `minids/train/` 负责配置、学习率、检查点、增长步骤、优化器和主循环；`minids/eval/` 负责验证损失、HellaSwag、单行 FIM 精确匹配、固定提示采样，并由 `EvalSuite` 统一预加载供训练循环复用；`minids/logging_io.py` 定义四个 jsonl 的记录格式，训练端写、窗口端（计划四）读；`minids/completion.py` 是 FIM 提示构造与逐行贪心生成，被评测、采样和 `complete.py` 共用。仓库根目录五个入口脚本只做参数解析。单元测试全部在 CPU 上用迷你配置和 tmp 目录里的合成分片运行；只有最后一个任务碰 GPU 和真实数据。

**Tech Stack:** Python 3.12、PyTorch 2.11 cu128（bf16 autocast，`torch.compile` 可选且本机不可用）、numpy、pyyaml、tokenizers、pyarrow、huggingface_hub（下载 HellaSwag）、safetensors（导出）、pytest。

**Spec:** `docs/superpowers/specs/2026-11-08-heimi-growth-lm-design.md`（本计划实现第 7.3 的事件记录、7.4、8、9、10、11、14 节，以及第 13.1 节中 `test_checkpoint`、`test_schedule`、`test_logging_io` 三项和第 13.2 节烟雾测试）

## 本计划在整体中的位置

计划一（模型、增长算子，已合并）和计划二（数据管线，已合并）交付了模型与数据。本计划把两者接起来训练，完成时仓库里有能跑完整阶梯的 `train.py`、能对任意检查点评测的 `evaluate.py`、能导出并在无训练代码环境下补全的 `export.py` 与 `complete.py`，磁盘上有级 0 的真实检查点和日志。级 1、级 2、任务阶段、对照组各需 1 到 11 小时 GPU，由用户在夜间按本计划 Task 14 报告里给出的命令运行，不在计划执行期内完成。计划四（窗口）只读本计划写出的 jsonl 与检查点，依赖 `logging_io.py` 的记录定义、`checkpoint.load_model`、`completion.complete_middle` 和 `Transformer.generate`。

## 已核实的事实（2026-11-09）

- HellaSwag 验证集：GitHub 上 `rowanz/hellaswag` 的原始 jsonl 链接已失效（404）。改用 Hugging Face 数据集 `Rowan/hellaswag`（不需要权限）的 `data/validation-00000-of-00001.parquet`，6.3 MB，10,042 行；列 `ctx`（字符串）、`endings`（4 个字符串的列表）、`label`（**字符串**，如 `"3"`，要转 int）。
- 分词器把换行编成独立 token（`\n`、`\n\n`、`\n    ` 等都是单个 token，不与前面的词合并），所以"生成到行尾"可以靠解码文本里出现 `\n` 判断。多字节字符被截断时解码得到 U+FFFD，不抛异常。
- 代码验证集前 400 万 token 含约 1,800 篇文档，其中 98% 有 ≥ 3 个非空行，全部解码耗时 0.6 秒：1000 个 FIM 样本从前 400 万 token 里抽就够。
- PyYAML：`300_000_000` 解析为整数；`2e-3` 解析为**字符串**，`2.0e-3` 才是浮点数。配置文件里所有科学计数法必须带小数点，加载器必须对字段做类型检查。
- `torch.load` 在 torch 2.11 默认 `weights_only=True`：检查点里只能放张量、数字、字符串、列表、字典，配置以 dict 保存。
- 实测吞吐（`docs/benchmarks/throughput.md`）：级 0 微批 16 为 228k token/s、7.2 GB；级 1 微批 16 为 133k、10.5 GB；级 2 微批 8 为 52.5k、9.9 GB。显存超限在 Windows 上不报错而是溢出到系统内存，`minids/memguard.py` 的 reserved 守卫沿用。
- 当前（写计划时）GPU 被一个游戏进程占用 13.6 GB：Task 11 的 GPU 烟雾测试和 Task 14 开跑前要先确认 `nvidia-smi` 显示的已用显存低于 2 GB。

## Global Constraints

- 优化器 AdamW，β1 0.9、β2 0.95，权重衰减 0.1，**词嵌入与 RMSNorm 缩放参数不衰减**，梯度裁剪 1.0（spec 8.2）。
- 每个优化步 524,288 个 token；微批次由各级配置给出（16 / 16 / 8 / 8），不足部分用梯度累积（32 / 32 / 64 / 64 个微步）。
- 学习率：线性预热到峰值，余弦衰减到峰值的 10%；级 0 预热占本级步数 2%，其余级 5%（spec 8.2）。
- 阶梯（spec 8.1）：`stage0` 从零、8 层 d384、0.3B token、峰值 2e-3；`stage1` 加深到 16 层、0.5B、1.5e-3；`stage2` 加宽到 d768、2.0B、6e-4；`task` 结构不变、0.7B、2e-4、配比 web 0.20 / code 0.80、FIM 0.5。对照 `baseline` 为级 2 结构从零 2.0B、6e-4。
- 每级开始时优化器状态重置；数据读取位置（加载器的 step 计数）跨级累计不回头（spec 7.4）。
- 增长顺序（spec 7.2、7.3）：加深后 fp32 logits 必须 `torch.equal`；加宽先以 `noise_std_rel=0` 复制并要求相对误差 ≤ 1e-4，再 `break_symmetry`，再要求相对误差 ≤ 该级 `rel_tol`（默认 1e-2）。验证失败中止训练，不进入下一级。
- bf16 自动混合精度，主权重 fp32；CPU 上不开 autocast。
- 日志（spec 9.2）：`metrics.jsonl` 每 10 步、`eval.jsonl` 每 500 步及每级结束、`samples.jsonl` 每 500 步、`events.jsonl` 八种事件类型，字段名与 spec 完全一致，定义只在 `logging_io.py`。
- 检查点（spec 9.3）：`runs/<run>/ckpt/<stage>_step<N>.pt` 与 `<stage>_final.pt`，每级保留最近 2 个中间检查点；内容含模型、优化器、训练状态、配置快照、git 提交号。
- 评测（spec 10）：验证损失 web 与 code 各 200 个固定窗口；HellaSwag 只在每级结束算；FIM 精确匹配 1000 个样本，任务阶段每次评测算，其他级只在级末算一次。
- NaN/Inf 损失立即停止并写 `nan_abort`；显存溢出写 `oom` 并退出，提示微批次减半；`torch.compile` 失败写 `compile_fallback` 并继续（spec 14）。
- 随机种子固定于配置并写入检查点；同一配置中断后恢复，取到的数据序列与未中断时一致。
- 单元测试在 CPU 上运行、不访问网络、不读 `data/` 下的真实文件；只有 Task 13 用 GPU 和真实数据。注释中文、标识符英文；`runs/`、`data/`、`models/` 在 git 之外。
- 每个 commit 信息用英文 conventional commits 格式，末尾加一行 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。

## Review Focus

1. 配置里 `lr: 2e-3` 被 YAML 读成字符串：`load_ladder` 必须在加载时抛出指明字段名的 `ValueError`，而不是让训练跑到第一步才在优化器里报 TypeError。测试在 Task 1。
2. 窗口在训练进程写 jsonl 的同时读取：最后一行可能只写了一半，`read_jsonl` 必须跳过不完整的末行而不是抛 JSON 错误。测试在 Task 2。
3. 中断后恢复：恢复后每一步的损失必须与未中断的同一配置逐位相同（数据顺序、学习率、步数计数、跨级累计量全部对齐）；恢复时若该级的模型结构配置与检查点不一致，必须拒绝而不是把权重硬塞进去。测试在 Task 10。
4. 已完成的级（存在 `<stage>_final.pt`）再次运行时必须跳过且不改写任何文件；`--stage` 指定的级若前一级未完成，必须给出说明缺哪个文件的错误。测试在 Task 10。
5. 检查点写到一半断电：半成品不能被当成检查点读取。保存先写 `.tmp` 再原子替换；读取损坏文件抛出含文件名的 `CheckpointError`。测试在 Task 4。

---

## 文件结构

```
configs/ladder.yaml               # 四级阶梯（Task 1）
configs/baseline105m.yaml         # 对照组（Task 1）
configs/tiny_smoke.yaml           # 烟雾测试（Task 10）
minids/train/__init__.py
minids/train/config.py            # DataPaths / EvalConfig / Prompt / StageConfig / LadderConfig，yaml 加载与校验
minids/train/schedule.py          # warmup_steps / lr_at
minids/train/checkpoint.py        # TrainState、保存/加载/最新/清理、load_model
minids/train/optim.py             # build_optimizer（分组权重衰减）
minids/train/grow.py              # grow_for_stage：按级配置增长、验证、加噪
minids/train/loop.py              # run_ladder：逐级训练、恢复、日志、评测、检查点
minids/logging_io.py              # 四种记录、append/read、RunLogger
minids/gpu.py                     # gpu_temperature_c（nvidia-smi）
minids/completion.py              # fim_prompt_ids / greedy_line / complete_middle
minids/eval/__init__.py
minids/eval/val_loss.py           # fixed_val_windows / val_loss
minids/eval/fim_em.py             # FimCase、抽样、fim_exact_match
minids/eval/samples.py            # generate_samples（固定提示）
minids/eval/hellaswag.py          # 下载、读取、choose_ending、hellaswag_accuracy
minids/eval/suite.py              # EvalSuite：预加载评测资源，训练循环与 evaluate.py 共用
minids/export.py                  # export_model / load_exported（safetensors）
minids/data/loader.py             # 修改：ShardSet 增加 sequential_windows / read_prefix
train.py  evaluate.py  sample.py  export.py  complete.py   # 入口
scripts/make_smoke_data.py        # 合成 token 分片
docs/training/stage0-report.md    # Task 13 产出
tests/helpers_train.py            # tiny_ladder、合成数据、两个桩模型
tests/test_train_config.py  test_logging_io.py  test_gpu.py  test_schedule.py  test_checkpoint.py
tests/test_val_loss.py  test_completion.py  test_fim_em.py  test_samples.py  test_hellaswag.py  test_eval_suite.py
tests/test_optim.py  test_grow_step.py  test_loop.py  test_train_cli.py  test_smoke_data.py
tests/test_eval_cli.py  test_export_complete.py
```

与 spec 第 4 节的差异：训练配置放在 `minids/train/config.py` 而不是 `minids/config.py`，让模型配置不依赖数据模块；检查点、优化器、增长步骤从 `loop.py` 拆成独立文件；`keepawake.py` 计划二已放在 `minids/`，不再移动；新增 `completion.py`、`gpu.py`、`eval/suite.py`、`eval/samples.py`、`export.py`。

---

### Task 1: 训练配置与阶梯文件

**Files:**
- Create: `minids/train/__init__.py`、`minids/train/config.py`、`configs/ladder.yaml`、`configs/baseline105m.yaml`
- Test: `tests/test_train_config.py`

**Interfaces:**
- Consumes: `minids.config.ModelConfig`（`from_dict`、`grown_depth`、`grown_width`、`to_dict`）、`minids.data.loader.MixConfig`
- Produces（`minids/train/config.py`）：
  ```python
  GROWTH_KINDS = ("none", "depth", "width")

  @dataclass(frozen=True)
  class DataPaths:
      index: str = "data/tokens/index.json"
      tokenizer: str = "data/tokenizer/heimi-16k.json"
      hellaswag: str = "data/eval/hellaswag_val.parquet"

  @dataclass(frozen=True)
  class EvalConfig:
      interval: int = 500          # 每多少步评测
      val_windows: int = 200       # 每来源固定窗口数
      hellaswag: bool = True       # 级末是否跑 HellaSwag
      fim_em_samples: int = 1000
      sample_interval: int = 500
      sample_tokens: int = 64

  @dataclass(frozen=True)
  class Prompt:
      id: str
      prefix: str
      suffix: str | None = None    # 非 None 即 FIM 填空提示

  @dataclass(frozen=True)
  class StageConfig:
      name: str
      growth: str                  # GROWTH_KINDS 之一
      model: ModelConfig
      tokens: int
      lr: float
      warmup_frac: float
      micro_batch: int
      mix: MixConfig
      noise_std_rel: float = 1e-3
      rel_tol: float = 1e-2

  @dataclass(frozen=True)
  class LadderConfig:
      seed: int
      seq_len: int
      tokens_per_step: int
      stages: tuple[StageConfig, ...]
      prompts: tuple[Prompt, ...]
      data: DataPaths = DataPaths()
      eval: EvalConfig = EvalConfig()
      log_interval: int = 10
      ckpt_interval: int = 500
      keep_checkpoints: int = 2
      compile: bool = False
      max_mem_frac: float = 0.8
      def validate(self) -> None            # __post_init__ 调用
      def stage(self, name: str) -> StageConfig
      def stage_index(self, name: str) -> int
      def steps_for(self, stage: StageConfig) -> int   # stage.tokens // tokens_per_step
      def accum_steps(self, stage: StageConfig) -> int # tokens_per_step // (micro_batch × seq_len)
      def to_dict(self) -> dict             # 可 yaml.safe_dump，load 回来相等

  def ladder_from_dict(d: dict) -> LadderConfig
  def load_ladder(path: str | Path) -> LadderConfig
  ```
- yaml 结构：顶层 `seed`、`seq_len`、`tokens_per_step`、`log_interval`、`ckpt_interval`、`keep_checkpoints`、`compile`、`max_mem_frac`、`data`、`eval`、`model`（各级共用的 ModelConfig 字段）、`prompts`、`stages`。每级的 `model` 映射覆盖在顶层 `model` 之上再构造 `ModelConfig`。

校验规则（`validate`，全部抛 `ValueError`，消息含字段名或级名）：`stages` 非空且名字唯一；第一级 `growth == "none"`；`depth` 级的 `model` 必须等于前一级 `model.grown_depth()`，`width` 级等于 `grown_width()`，`none` 级（非首级）等于前一级；每级 `model.seq_len == seq_len` 且 `vocab_size` 一致；`tokens_per_step % (micro_batch × seq_len) == 0`；`tokens >= tokens_per_step`；`0 ≤ warmup_frac < 1`；`growth in GROWTH_KINDS`。类型检查：`lr`、`warmup_frac`、`noise_std_rel`、`rel_tol`、`max_mem_frac` 必须是 `int | float`（不是 `bool`、不是 `str`），`tokens`、`micro_batch` 等必须是 `int`。

六条固定提示（spec 9.2：3 英文、2 代码前缀、1 FIM）写进 `ladder.yaml` 的 `prompts`：
`en_history` "The history of the Roman Empire began when"；`en_science` "Water boils at a lower temperature on a mountain because"；`en_story` "Once upon a time, a small black cat named Heimi"；`py_func` "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"；`sh_find` "# list all python files modified in the last day\nfind . -name"；`fim_py` prefix "def is_even(n):\n    " suffix "\n\nprint(is_even(4))\n"。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_train_config.py
import pytest
import yaml
from minids.config import ModelConfig
from minids.data.loader import MixConfig
from minids.train.config import LadderConfig, Prompt, StageConfig, ladder_from_dict, load_ladder

STAGE2 = dict(n_layer=16, d_model=768, n_head=12, kv_lora_rank=256, ffn_hidden=2048)


def test_ladder_yaml_matches_spec_8_1():
    cfg = load_ladder("configs/ladder.yaml")
    assert [s.name for s in cfg.stages] == ["stage0", "stage1", "stage2", "task"]
    assert [s.growth for s in cfg.stages] == ["none", "depth", "width", "none"]
    assert [s.tokens for s in cfg.stages] == [300_000_000, 500_000_000, 2_000_000_000, 700_000_000]
    assert [s.lr for s in cfg.stages] == [2e-3, 1.5e-3, 6e-4, 2e-4]
    assert [s.warmup_frac for s in cfg.stages] == [0.02, 0.05, 0.05, 0.05]
    assert [s.micro_batch for s in cfg.stages] == [16, 16, 8, 8]
    assert cfg.stages[0].model == ModelConfig() and cfg.stages[2].model == ModelConfig(**STAGE2)
    assert cfg.stages[3].mix == MixConfig(web=0.2, code=0.8, fim_rate=0.5)
    assert [cfg.steps_for(s) for s in cfg.stages] == [572, 953, 3814, 1335]
    assert [cfg.accum_steps(s) for s in cfg.stages] == [32, 32, 64, 64]
    assert cfg.tokens_per_step == 524_288 and cfg.eval.interval == 500 and cfg.ckpt_interval == 500
    assert len(cfg.prompts) == 6 and sum(p.suffix is not None for p in cfg.prompts) == 1


def test_baseline_yaml_is_single_stage2_from_scratch():
    cfg = load_ladder("configs/baseline105m.yaml")
    assert len(cfg.stages) == 1 and cfg.stages[0].name == "baseline" and cfg.stages[0].growth == "none"
    assert cfg.stages[0].model == ModelConfig(**STAGE2) and cfg.stages[0].tokens == 2_000_000_000 and cfg.stages[0].lr == 6e-4


def _tiny_dict(**over):
    d = {
        "seed": 1, "seq_len": 32, "tokens_per_step": 256,
        "model": {"vocab_size": 300, "seq_len": 32, "n_layer": 2, "d_model": 64, "n_head": 2,
                  "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 32, "kv_lora_rank": 32, "ffn_hidden": 128},
        "prompts": [{"id": "a", "prefix": "x"}],
        "stages": [
            {"name": "s0", "growth": "none", "model": {}, "tokens": 768, "lr": 1.0e-3, "warmup_frac": 0.1,
             "micro_batch": 4, "mix": {"web": 0.5, "code": 0.5, "fim_rate": 0.0}},
            {"name": "s1", "growth": "depth", "model": {"n_layer": 4}, "tokens": 512, "lr": 1.0e-3,
             "warmup_frac": 0.1, "micro_batch": 4, "mix": {"web": 0.5, "code": 0.5, "fim_rate": 0.0}},
        ],
    }
    d.update(over)
    return d


def test_tiny_dict_round_trips_through_yaml(tmp_path):
    cfg = ladder_from_dict(_tiny_dict())
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert load_ladder(p) == cfg


def test_string_lr_from_yaml_scientific_notation_is_rejected():
    # PyYAML 把 2e-3 读成字符串；这里必须当场报错而不是在优化器里炸
    d = _tiny_dict()
    d["stages"][0]["lr"] = "2e-3"
    with pytest.raises(ValueError, match="lr"):
        ladder_from_dict(d)


def test_growth_chain_mismatch_names_stage():
    d = _tiny_dict()
    d["stages"][1]["model"] = {"n_layer": 3}
    with pytest.raises(ValueError, match="s1"):
        ladder_from_dict(d)


def test_first_stage_must_not_grow():
    d = _tiny_dict()
    d["stages"][0]["growth"] = "depth"
    with pytest.raises(ValueError, match="s0"):
        ladder_from_dict(d)


def test_tokens_per_step_must_divide_by_micro_tokens():
    with pytest.raises(ValueError, match="tokens_per_step"):
        ladder_from_dict(_tiny_dict(tokens_per_step=200))


def test_duplicate_stage_names_rejected():
    d = _tiny_dict()
    d["stages"][1]["name"] = "s0"
    d["stages"][1]["growth"] = "none"
    d["stages"][1]["model"] = {}
    with pytest.raises(ValueError, match="s0"):
        ladder_from_dict(d)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_config.py -q`
Expected: 全部 FAIL，`ModuleNotFoundError: minids.train`

- [ ] **Step 3: 实现 `minids/train/config.py`，写 `configs/ladder.yaml` 与 `configs/baseline105m.yaml`**

yaml 里所有科学计数法带小数点（`2.0e-3`、`6.0e-4`、`1.0e-3`、`1.0e-2`）；级 token 数写成纯数字。`baseline105m.yaml` 的唯一一级 `warmup_frac: 0.05`（与级 2 一致，让对照与级 2 只差初始化与此前阶梯花掉的 token，这是本计划的选择，spec 未规定）。`ladder.yaml` 顶层 `model` 只写 `vocab_size: 16384` 和 `seq_len: 1024`，每级写全 `n_layer`、`d_model`、`n_head`、`kv_lora_rank`、`ffn_hidden`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_config.py -q`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add minids/train/__init__.py minids/train/config.py configs/ladder.yaml configs/baseline105m.yaml tests/test_train_config.py
git commit -m "feat(train): ladder configuration with growth-chain validation and spec 8.1 configs"
```

---

### Task 2: 日志格式与 GPU 温度

**Files:**
- Create: `minids/logging_io.py`、`minids/gpu.py`
- Test: `tests/test_logging_io.py`、`tests/test_gpu.py`

**Interfaces:**
- Produces（`minids/logging_io.py`）：
  ```python
  METRICS_FILE, EVAL_FILE, EVENTS_FILE, SAMPLES_FILE = "metrics.jsonl", "eval.jsonl", "events.jsonl", "samples.jsonl"
  EVENT_TYPES = ("stage_start", "stage_end", "growth", "checkpoint", "nan_abort", "oom", "compile_fallback", "resume")

  @dataclass(frozen=True)
  class MetricsRecord:   # spec 9.2 字段，顺序一致
      stage: str; step: int; global_step: int; tokens_stage: int; tokens_total: int
      loss: float; lr: float; grad_norm: float; tok_per_s: float; mem_gb: float
      gpu_temp_c: float | None; time: str

  @dataclass(frozen=True)
  class EvalRecord:
      stage: str; global_step: int; tokens_total: int
      val_loss_web: float; val_loss_code: float
      hellaswag_acc: float | None; fim_em: float | None; time: str

  @dataclass(frozen=True)
  class EventRecord:     # __post_init__：type 不在 EVENT_TYPES → ValueError
      type: str; stage: str; global_step: int; tokens_total: int; time: str
      data: dict = field(default_factory=dict)

  @dataclass(frozen=True)
  class SampleRecord:
      stage: str; global_step: int; tokens_total: int
      prompt_id: str; prompt: str; completion: str; time: str

  def now_iso() -> str                                   # 本地时间，秒精度，带时区
  def append_jsonl(path: Path, record) -> None           # dataclass → dict，ensure_ascii=False，一行，写后 flush
  def read_jsonl(path: Path, since_global_step: int | None = None) -> list[dict]
      # 文件不存在 → []；末行不是完整 JSON（写了一半）→ 跳过；since 给出时只返回 global_step > since 的记录

  class RunLogger:
      def __init__(self, run_dir: Path)                  # 建目录
      def metrics(self, rec: MetricsRecord) -> None
      def eval(self, rec: EvalRecord) -> None
      def event(self, type: str, stage: str, global_step: int, tokens_total: int, **data) -> None
      def sample(self, rec: SampleRecord) -> None
      def path(self, name: str) -> Path                  # name ∈ 四个文件常量
  ```
- Produces（`minids/gpu.py`）：`def gpu_temperature_c(timeout_s: float = 2.0) -> float | None` —— 调 `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits`，取第一行转 float；找不到命令、超时、输出解析失败都返回 `None`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_logging_io.py
import pytest
from minids.logging_io import (EVENTS_FILE, METRICS_FILE, EvalRecord, EventRecord, MetricsRecord, RunLogger,
                               SampleRecord, append_jsonl, now_iso, read_jsonl)


def _metrics(gs, loss=1.0):
    return MetricsRecord(stage="s0", step=gs, global_step=gs, tokens_stage=gs * 10, tokens_total=gs * 10,
                         loss=loss, lr=1e-3, grad_norm=0.5, tok_per_s=1000.0, mem_gb=1.5, gpu_temp_c=None, time=now_iso())


def test_round_trip_all_record_types(tmp_path):
    log = RunLogger(tmp_path)
    log.metrics(_metrics(1))
    log.eval(EvalRecord("s0", 1, 10, 2.0, 3.0, None, 0.1, now_iso()))
    log.event("growth", "s1", 1, 10, params_before=5, params_after=9)
    log.sample(SampleRecord("s0", 1, 10, "p", "The", " cat", now_iso()))
    m, e, ev, s = (read_jsonl(log.path(n)) for n in ("metrics.jsonl", "eval.jsonl", "events.jsonl", "samples.jsonl"))
    assert m[0]["loss"] == 1.0 and set(m[0]) == set(MetricsRecord.__dataclass_fields__)
    assert e[0]["hellaswag_acc"] is None and e[0]["fim_em"] == 0.1
    assert ev[0]["type"] == "growth" and ev[0]["data"] == {"params_before": 5, "params_after": 9}
    assert s[0]["completion"] == " cat"


def test_since_filters_strictly_greater(tmp_path):
    p = tmp_path / METRICS_FILE
    for gs in (10, 20, 30):
        append_jsonl(p, _metrics(gs))
    assert [r["global_step"] for r in read_jsonl(p, since_global_step=20)] == [30]
    assert len(read_jsonl(p)) == 3


def test_missing_file_reads_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_half_written_last_line_is_skipped(tmp_path):
    # 窗口读的时候训练进程可能正写到一半
    p = tmp_path / METRICS_FILE
    append_jsonl(p, _metrics(1))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"stage": "s0", "step": 2, "glo')
    assert [r["step"] for r in read_jsonl(p)] == [1]


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError, match="type"):
        EventRecord("exploded", "s0", 0, 0, now_iso())


def test_unicode_kept_readable(tmp_path):
    p = tmp_path / EVENTS_FILE
    append_jsonl(p, EventRecord("stage_start", "级0", 0, 0, now_iso(), {"note": "黑米"}))
    assert "黑米" in p.read_text(encoding="utf-8") and read_jsonl(p)[0]["stage"] == "级0"
```

```python
# tests/test_gpu.py
import subprocess
from minids import gpu


def test_parses_first_gpu_temperature(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="55\n", stderr=""))
    assert gpu.gpu_temperature_c() == 55.0


def test_missing_nvidia_smi_returns_none(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.gpu_temperature_c() is None


def test_garbage_output_returns_none(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="N/A\n", stderr=""))
    assert gpu.gpu_temperature_c() is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logging_io.py tests/test_gpu.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `minids/logging_io.py` 与 `minids/gpu.py`**

`read_jsonl` 逐行 `json.loads`，只对**最后一行**的 `JSONDecodeError` 容忍（中间行坏了说明文件损坏，照常抛出）。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logging_io.py tests/test_gpu.py -q`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add minids/logging_io.py minids/gpu.py tests/test_logging_io.py tests/test_gpu.py
git commit -m "feat: jsonl run logging records shared with the dashboard; GPU temperature probe"
```

---

### Task 3: 学习率调度

**Files:**
- Create: `minids/train/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Produces：
  ```python
  def warmup_steps(total_steps: int, warmup_frac: float) -> int   # max(1, round(total × frac))
  def lr_at(step: int, total_steps: int, peak_lr: float, warmup: int, min_ratio: float = 0.1) -> float
  ```
  `step < warmup`：`peak × (step + 1) / warmup`；否则 `progress = min(1, (step − warmup) / max(1, total − warmup))`，`lr = min + 0.5 × (peak − min) × (1 + cos(π × progress))`，`min = min_ratio × peak`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_schedule.py
import math
from minids.train.schedule import lr_at, warmup_steps


def test_warmup_steps_from_spec_fractions():
    assert warmup_steps(572, 0.02) == 11 and warmup_steps(953, 0.05) == 48
    assert warmup_steps(3814, 0.05) == 191 and warmup_steps(10, 0.0) == 1


def test_linear_warmup_reaches_peak_on_last_warmup_step():
    assert math.isclose(lr_at(0, 1000, 1e-3, 50), 1e-3 / 50)
    assert math.isclose(lr_at(49, 1000, 1e-3, 50), 1e-3)
    assert math.isclose(lr_at(50, 1000, 1e-3, 50), 1e-3)


def test_cosine_ends_at_ten_percent_and_clamps_beyond_total():
    assert math.isclose(lr_at(1000, 1000, 1e-3, 50), 1e-4)
    assert math.isclose(lr_at(1500, 1000, 1e-3, 50), 1e-4)
    mid = lr_at(525, 1000, 1e-3, 50)  # 衰减段正中
    assert math.isclose(mid, 1e-4 + 0.5 * 9e-4, rel_tol=1e-6)


def test_non_increasing_after_warmup():
    vals = [lr_at(s, 1000, 1e-3, 50) for s in range(50, 1001)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schedule.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `minids/train/schedule.py`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schedule.py -q`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add minids/train/schedule.py tests/test_schedule.py
git commit -m "feat(train): warmup plus cosine learning-rate schedule"
```

---

### Task 4: 检查点

**Files:**
- Create: `minids/train/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes：`Transformer`、`ModelConfig.from_dict / to_dict`
- Produces：
  ```python
  class CheckpointError(RuntimeError): ...

  @dataclass
  class TrainState:
      stage: str
      step: int = 0            # 本级内已完成的优化步
      global_step: int = 0     # 跨级累计优化步
      micro_step: int = 0      # 加载器 get_batch 的 step 参数，跨级累计
      tokens_stage: int = 0
      tokens_total: int = 0
      def to_dict(self) -> dict
      @classmethod
      def from_dict(cls, d: dict) -> "TrainState"

  def git_commit() -> str | None                           # git rev-parse HEAD，失败 None
  def checkpoint_name(stage: str, step: int | None) -> str  # f"{stage}_step{step}.pt"；step 为 None → f"{stage}_final.pt"
  def save_checkpoint(path: Path, model: Transformer, optimizer: torch.optim.Optimizer | None,
                      state: TrainState, ladder_cfg: dict) -> None
      # payload: model（state_dict）、optimizer（state_dict 或 None）、state、model_config、ladder_config、
      #          git_commit、torch_rng（CPU 状态张量）、cuda_rng（张量或 None）、saved_at
      # 先写 path.with_suffix(".tmp") 再 os.replace 到 path
  def load_checkpoint(path: Path, device: str) -> dict     # torch.load(weights_only=True, map_location=device)；任何异常 → CheckpointError(含 path.name)
  def load_model(path: Path, device: str) -> tuple[Transformer, dict]
      # 按 model_config 重建、加载权重、eval()；第二项是 payload 去掉 model 与 optimizer
  def latest_step_checkpoint(ckpt_dir: Path, stage: str) -> Path | None   # 按 N 的数值最大
  def prune_step_checkpoints(ckpt_dir: Path, stage: str, keep: int) -> list[Path]  # 删除该级最旧的，只留 keep 个；不碰 _final 与其他级；返回删掉的
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_checkpoint.py
import pytest
import torch
from minids.model.transformer import Transformer
from minids.train.checkpoint import (CheckpointError, TrainState, checkpoint_name, latest_step_checkpoint, load_checkpoint,
                                     load_model, prune_step_checkpoints, save_checkpoint)


def _trained(tiny_cfg):
    m = Transformer(tiny_cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (2, 16))
    m(x, x)[1].backward()
    opt.step()
    return m, opt


def test_round_trip_weights_optimizer_state(tmp_path, tiny_cfg):
    m, opt = _trained(tiny_cfg)
    state = TrainState("s0", step=3, global_step=7, micro_step=14, tokens_stage=300, tokens_total=700)
    p = tmp_path / checkpoint_name("s0", 3)
    save_checkpoint(p, m, opt, state, {"seed": 1})
    assert p.name == "s0_step3.pt" and not list(tmp_path.glob("*.tmp"))
    payload = load_checkpoint(p, "cpu")
    m2 = Transformer(tiny_cfg)
    m2.load_state_dict(payload["model"])
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(payload["optimizer"])
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
    assert TrainState.from_dict(payload["state"]) == state
    assert payload["ladder_config"] == {"seed": 1} and payload["model_config"] == tiny_cfg.to_dict()
    assert "git_commit" in payload and "saved_at" in payload


def test_load_model_rebuilds_from_config(tmp_path, tiny_cfg):
    m, _ = _trained(tiny_cfg)
    p = tmp_path / checkpoint_name("s0", None)
    save_checkpoint(p, m, None, TrainState("s0"), {})
    m2, meta = load_model(p, "cpu")
    x = torch.randint(0, 256, (1, 8))
    assert torch.equal(m.eval()(x)[0], m2(x)[0]) and not m2.training
    assert "model" not in meta and "optimizer" not in meta and meta["state"]["stage"] == "s0"


def test_corrupt_file_raises_named_error(tmp_path):
    p = tmp_path / "s0_step1.pt"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="s0_step1.pt"):
        load_checkpoint(p, "cpu")


def test_latest_is_numeric_not_lexical(tmp_path):
    for n in (300, 1200, 50):
        (tmp_path / f"s0_step{n}.pt").write_bytes(b"x")
    (tmp_path / "s1_step9999.pt").write_bytes(b"x")
    (tmp_path / "s0_final.pt").write_bytes(b"x")
    assert latest_step_checkpoint(tmp_path, "s0").name == "s0_step1200.pt"
    assert latest_step_checkpoint(tmp_path, "s2") is None


def test_prune_keeps_newest_and_final_and_other_stages(tmp_path):
    for n in (100, 200, 300, 400):
        (tmp_path / f"s0_step{n}.pt").write_bytes(b"x")
    (tmp_path / "s0_final.pt").write_bytes(b"x")
    (tmp_path / "s1_step50.pt").write_bytes(b"x")
    deleted = prune_step_checkpoints(tmp_path, "s0", keep=2)
    assert sorted(p.name for p in deleted) == ["s0_step100.pt", "s0_step200.pt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s0_final.pt", "s0_step300.pt", "s0_step400.pt", "s1_step50.pt"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_checkpoint.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `minids/train/checkpoint.py`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_checkpoint.py -q`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add minids/train/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(train): atomic checkpoints with train state, latest lookup and pruning"
```

---

### Task 5: 顺序读分片与验证损失

**Files:**
- Modify: `minids/data/loader.py`（`ShardSet` 增加两个方法）
- Create: `minids/eval/__init__.py`、`minids/eval/val_loss.py`
- Test: `tests/test_loader.py`（追加）、`tests/test_val_loss.py`

**Interfaces:**
- Consumes：`ShardSet.from_index(index_path, source, split, min_tokens)`、`write_shard`（tests/helpers_data.py）
- Produces（`ShardSet`）：
  ```python
  def sequential_windows(self, length: int, n: int) -> np.ndarray
      # (n, length) int64：从第一片开头起按顺序切不重叠窗口，一片剩余不足一个窗口就跳到下一片开头；凑不够 n 个 → ValueError
  def read_prefix(self, max_tokens: int) -> np.ndarray
      # 按片顺序拼接，最多 max_tokens 个 token，int64
  ```
- Produces（`minids/eval/val_loss.py`）：
  ```python
  def fixed_val_windows(index_path: Path, source: str, n_windows: int, seq_len: int) -> torch.LongTensor  # (n, seq_len + 1)
  @torch.no_grad()
  def val_loss(model: Transformer, windows: torch.LongTensor, device: str, micro_batch: int) -> float
      # 按 micro_batch 分批，cuda 上 bf16 autocast；所有窗口所有位置的平均交叉熵；结束后恢复 model.training
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_loader.py（追加）
def test_sequential_windows_follow_file_order_across_shards(tmp_path):
    a = write_shard(tmp_path / "web_0.bin", list(range(20, 31)))   # 11 个 token：两个长 5 窗口，余 1 丢弃
    b = write_shard(tmp_path / "web_1.bin", list(range(100, 110)))  # 10 个：两个窗口
    s = ShardSet([a, b], [11, 10], 5)
    w = s.sequential_windows(5, 4)
    assert w.tolist() == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [100, 101, 102, 103, 104], [105, 106, 107, 108, 109]]
    assert w.dtype == np.int64
    with pytest.raises(ValueError, match="5"):
        s.sequential_windows(5, 5)


def test_read_prefix_concatenates_shards(tmp_path):
    a = write_shard(tmp_path / "web_0.bin", list(range(20, 25)))
    b = write_shard(tmp_path / "web_1.bin", list(range(100, 105)))
    assert ShardSet([a, b], [5, 5], 5).read_prefix(7).tolist() == [20, 21, 22, 23, 24, 100, 101]
```

```python
# tests/test_val_loss.py
import json
import torch
import torch.nn.functional as F
from minids.eval.val_loss import fixed_val_windows, val_loss
from minids.model.transformer import Transformer
from tests.helpers_data import write_shard


def _index(tmp_path, tokens):
    write_shard(tmp_path / "web" / "val_0000.bin", tokens)
    idx = {"vocab_size": 256, "dtype": "uint16", "eot_id": 0,
           "sources": {"web": {"train": [], "val": [{"file": "val_0000.bin", "tokens": len(tokens)}]}}}
    p = tmp_path / "index.json"
    p.write_text(json.dumps(idx), encoding="utf-8")
    return p


def test_fixed_windows_are_stable_and_shaped(tmp_path):
    p = _index(tmp_path, list(range(16, 16 + 200)))
    w1, w2 = fixed_val_windows(p, "web", 3, 16), fixed_val_windows(p, "web", 3, 16)
    assert w1.shape == (3, 17) and w1.dtype == torch.int64 and torch.equal(w1, w2)
    assert w1[1, 0].item() == 16 + 17


def test_val_loss_matches_manual_mean_over_uneven_batches(tiny_cfg):
    torch.manual_seed(0)
    m = Transformer(tiny_cfg).train()
    windows = torch.randint(16, 256, (5, 9))
    got = val_loss(m, windows, "cpu", micro_batch=2)  # 5 个窗口分成 2+2+1
    assert m.training is True  # 评测后恢复原来的训练模式
    m.eval()
    logits, _ = m(windows[:, :-1])
    want = F.cross_entropy(logits.reshape(-1, 256), windows[:, 1:].reshape(-1)).item()
    assert abs(got - want) < 1e-5
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_loader.py tests/test_val_loss.py -q`
Expected: 新增 4 个 FAIL（`AttributeError: sequential_windows` 与 `ModuleNotFoundError: minids.eval`），原有 test_loader 用例通过

- [ ] **Step 3: 实现 `ShardSet.sequential_windows`、`ShardSet.read_prefix` 与 `minids/eval/val_loss.py`**

`fixed_val_windows` 用 `ShardSet.from_index(index_path, source, "val", seq_len + 1).sequential_windows(seq_len + 1, n_windows)`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_loader.py tests/test_val_loss.py -q`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add minids/data/loader.py minids/eval/__init__.py minids/eval/val_loss.py tests/test_loader.py tests/test_val_loss.py
git commit -m "feat(eval): sequential shard reads and fixed-window validation loss"
```

---

### Task 6: FIM 补全、单行精确匹配与固定提示采样

**Files:**
- Create: `minids/completion.py`、`minids/eval/fim_em.py`、`minids/eval/samples.py`、`tests/helpers_train.py`（本任务只放两个桩模型；Task 9 再补数据与配置构造）
- Test: `tests/test_completion.py`、`tests/test_fim_em.py`、`tests/test_samples.py`

**Interfaces:**
- Consumes：`HeimiTokenizer`（`encode`、`decode`）、`FimIds`、`Transformer.generate(idx, max_new_tokens, temperature, top_k)`、`Prompt`（Task 1）、`tiny_tokenizer`（tests/helpers_data.py）
- Produces（`tests/helpers_train.py`）：
  ```python
  class ScriptedModel(nn.Module):
      """generate 每次追加脚本里的下一个 token；用于测试停止规则。有 cfg.seq_len。"""
      def __init__(self, script: list[int], seq_len: int = 64)
      def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None) -> LongTensor
  class CountingModel(nn.Module):
      """forward 返回 (logits, None)，logits 强烈偏向 下一个 token = 当前 token + 1；用于 HellaSwag 测试。"""
      def __init__(self, vocab_size: int, seq_len: int = 64)
  ```
- Produces（`minids/completion.py`）：
  ```python
  FIM_HOLE = "<FILL>"   # samples.jsonl 里展示 FIM 提示时用它标出空位
  def fim_prompt_ids(tok: HeimiTokenizer, prefix: str, suffix: str, max_prefix_tokens: int, max_suffix_tokens: int,
                     ids: FimIds = FimIds()) -> list[int]
      # [ids.prefix] + prefix 编码的最后 max_prefix_tokens 个 + [ids.suffix] + suffix 编码的前 max_suffix_tokens 个 + [ids.middle]
  @torch.no_grad()
  def greedy_line(model, tok: HeimiTokenizer, prompt_ids: list[int], max_new_tokens: int, device: str, eot: int = 0) -> str
      # 每步 model.generate(idx, 1, temperature=0.0)；生成 eot 即停；解码已生成部分含 "\n" 即截到第一个换行并停；返回文本
  def complete_middle(model, tok, prefix: str, suffix: str, max_new_tokens: int, device: str,
                      max_prefix_tokens: int = 512, max_suffix_tokens: int = 256) -> str
  ```
  调用方保证 `max_prefix_tokens + max_suffix_tokens + 3 + max_new_tokens ≤ model.cfg.seq_len`（默认值配 seq_len 1024 时留 253 个 token 生成余量）。
- Produces（`minids/eval/fim_em.py`）：
  ```python
  @dataclass(frozen=True)
  class FimCase:
      prefix: str; middle: str; suffix: str     # prefix + middle + suffix == 原文
  def split_documents(tokens: np.ndarray, eot: int = 0) -> list[np.ndarray]   # 按 eot 切，丢弃空段
  def make_fim_cases(docs: list[str], n: int, rng: np.random.Generator, min_lines: int = 3) -> list[FimCase]
      # 只用 ≥ min_lines 个非空行的文档；不放回抽 n 篇（不足全取）；每篇均匀随机挑一个非空行做 middle（不含行尾换行）
  def cases_from_index(index_path: Path, tok: HeimiTokenizer, n: int, seed: int, max_tokens: int = 4_000_000) -> list[FimCase]
      # ShardSet.from_index(..., "code", "val", 1).read_prefix(max_tokens) → split → decode → make_fim_cases(rng=default_rng(seed))
  def fim_exact_match(model, tok, cases: Sequence[FimCase], device: str, max_new_tokens: int = 64,
                      max_prefix_tokens: int = 512, max_suffix_tokens: int = 256) -> float
      # complete_middle(...).strip() == case.middle.strip() 的比例；cases 为空 → ValueError
  ```
- Produces（`minids/eval/samples.py`）：
  ```python
  def generate_samples(model, tok, prompts: Sequence[Prompt], max_new_tokens: int, device: str, seed: int,
                       max_prefix_tokens: int = 512, max_suffix_tokens: int = 256) -> list[tuple[str, str, str]]
      # 每条提示返回 (prompt_id, 展示文本, completion)。普通提示：torch.random.fork_rng 内 manual_seed(seed)，
      # generate(temperature=0.8, top_k=50)，completion 为新 token 解码；FIM 提示：complete_middle，展示文本为 prefix + FIM_HOLE + suffix
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_completion.py
import numpy as np
from minids.completion import complete_middle, fim_prompt_ids, greedy_line
from tests.helpers_data import tiny_tokenizer
from tests.helpers_train import ScriptedModel


def test_prompt_layout_and_truncation(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    prefix, suffix = "abc def ghi", "jkl mno pqr"
    ids = fim_prompt_ids(tok, prefix, suffix, max_prefix_tokens=2, max_suffix_tokens=1)
    p_ids, s_ids = tok.encode(prefix), tok.encode(suffix)
    assert ids == [1] + p_ids[-2:] + [3] + s_ids[:1] + [2]


def test_greedy_line_stops_at_newline_and_strips_it(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    script = tok.encode("return a + b") + tok.encode("\n") + tok.encode("garbage")
    assert greedy_line(ScriptedModel(script), tok, [1, 2], 32, "cpu") == "return a + b"


def test_greedy_line_stops_at_eot_and_at_budget(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    body = tok.encode("x = 1")
    assert greedy_line(ScriptedModel(body + [0] + body), tok, [1], 32, "cpu") == "x = 1"
    assert greedy_line(ScriptedModel(body * 10), tok, [1], len(body), "cpu") == "x = 1"


def test_complete_middle_wires_prompt_and_generation(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    out = complete_middle(ScriptedModel(tok.encode("pass") + tok.encode("\n")), tok, "def f():\n    ", "\n", 8, "cpu", 4, 4)
    assert out == "pass"
```

```python
# tests/test_fim_em.py
import numpy as np
import pytest
from minids.eval.fim_em import FimCase, fim_exact_match, make_fim_cases, split_documents
from tests.helpers_data import tiny_tokenizer
from tests.helpers_train import ScriptedModel

DOCS = ["a = 1\nb = 2\nc = 3\n", "one\n\ntwo\nthree", "short\n", "x\ny\nz\nw\n"]


def test_split_documents_on_eot_drops_empty():
    assert [d.tolist() for d in split_documents(np.array([5, 6, 0, 0, 7, 0]))] == [[5, 6], [7]]


def test_cases_reconstruct_doc_and_pick_nonblank_line():
    cases = make_fim_cases(DOCS, n=10, rng=np.random.default_rng(0), min_lines=3)
    assert len(cases) == 3  # "short\n" 只有一行，被排除
    assert {c.prefix + c.middle + c.suffix for c in cases} == set(DOCS) - {"short\n"}
    for c in cases:
        assert c.middle.strip() and "\n" not in c.middle


def test_cases_deterministic_per_seed():
    a = make_fim_cases(DOCS, 3, np.random.default_rng(7))
    b = make_fim_cases(DOCS, 3, np.random.default_rng(7))
    assert a == b


def test_exact_match_rate(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    cases = [FimCase("a\n", "b = 2", "\nc\n"), FimCase("a\n", "zzz", "\nc\n")]
    model = ScriptedModel(tok.encode("b = 2") + tok.encode("\n"))
    assert fim_exact_match(model, tok, cases, "cpu", 8, 4, 4) == 0.5
    with pytest.raises(ValueError):
        fim_exact_match(model, tok, [], "cpu")
```

```python
# tests/test_samples.py
from dataclasses import replace
from minids.completion import FIM_HOLE
from minids.eval.samples import generate_samples
from minids.model.transformer import Transformer
from minids.train.config import Prompt
from tests.helpers_data import tiny_tokenizer

PROMPTS = [Prompt("en", "The cat"), Prompt("fim", "def f():\n    ", "\nprint(f())\n")]


def test_one_sample_per_prompt_with_display_text(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    m = Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size))
    out = generate_samples(m, tok, PROMPTS, 4, "cpu", seed=0, max_prefix_tokens=8, max_suffix_tokens=8)
    assert [o[0] for o in out] == ["en", "fim"]
    assert out[0][1] == "The cat" and out[1][1] == "def f():\n    " + FIM_HOLE + "\nprint(f())\n"
    assert "\n" not in out[1][2]


def test_samples_reproducible_for_same_seed(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    m = Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size))
    a = generate_samples(m, tok, PROMPTS[:1], 6, "cpu", seed=3)
    b = generate_samples(m, tok, PROMPTS[:1], 6, "cpu", seed=3)
    assert a == b
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_completion.py tests/test_fim_em.py tests/test_samples.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `tests/helpers_train.py` 的两个桩、`minids/completion.py`、`minids/eval/fim_em.py`、`minids/eval/samples.py`**

`ScriptedModel.generate` 忽略 `temperature`/`top_k`，按调用次数从脚本取下一个 token 追加到 `idx` 末尾（脚本用尽后重复最后一个）。`make_fim_cases` 的 `prefix` 是该行起点之前的全部文本，`suffix` 从该行末尾的换行（若有）开始。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_completion.py tests/test_fim_em.py tests/test_samples.py -q`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add minids/completion.py minids/eval/fim_em.py minids/eval/samples.py tests/helpers_train.py tests/test_completion.py tests/test_fim_em.py tests/test_samples.py
git commit -m "feat(eval): FIM line completion, single-line exact-match metric and fixed-prompt sampling"
```

---

### Task 7: HellaSwag

**Files:**
- Create: `minids/eval/hellaswag.py`
- Test: `tests/test_hellaswag.py`

**Interfaces:**
- Consumes：`huggingface_hub.hf_hub_download`、`pyarrow.parquet`、`write_parquet`（tests/helpers_data.py）、`CountingModel`（Task 6）
- Produces：
  ```python
  HELLASWAG_REPO = "Rowan/hellaswag"
  HELLASWAG_FILE = "data/validation-00000-of-00001.parquet"   # 6.3 MB，10,042 行
  def download_hellaswag(dest: Path) -> Path
      # dest 已存在直接返回；否则 hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=临时目录) 后移到 dest
  def load_hellaswag(path: Path) -> list[dict]       # [{"ctx": str, "endings": [str × 4], "label": int}]，label 列是字符串要转 int
  @torch.no_grad()
  def choose_ending(model, ctx_ids: list[int], endings_ids: list[list[int]], device: str, max_len: int, pad_id: int = 4) -> int
      # 每个候选 = ctx + ending，超过 max_len 从左截断；右填充 pad_id 成一批前向一次；
      # 对 ending 各 token 位置的 log_softmax 取均值（归一化对数似然）；返回最大者的下标
  def hellaswag_accuracy(model, tok, rows: Sequence[dict], device: str, max_len: int, limit: int | None = None) -> float
      # ending 前补一个空格再编码：tok.encode(" " + ending)
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hellaswag.py
import pyarrow as pa
import pyarrow.parquet as pq
from minids.eval.hellaswag import choose_ending, hellaswag_accuracy, load_hellaswag
from minids.model.transformer import Transformer
from tests.helpers_data import tiny_tokenizer
from tests.helpers_train import CountingModel


def test_choose_ending_prefers_most_likely_continuation():
    m = CountingModel(vocab_size=128)
    ctx = [10, 11, 12]
    endings = [[13, 14], [20, 21, 22], [13, 99], [5]]
    assert choose_ending(m, ctx, endings, "cpu", max_len=64) == 0


def test_choose_ending_normalizes_by_length():
    # 两个候选都完全正确，长的不该因为累加更多项而输给短的，也不该赢：均值相等时取第一个
    m = CountingModel(vocab_size=128)
    assert choose_ending(m, [10, 11], [[12, 13, 14], [12]], "cpu", max_len=64) == 0


def test_load_hellaswag_casts_string_label(tmp_path):
    p = tmp_path / "hs.parquet"
    pq.write_table(pa.table({"ctx": ["A man is", "She"], "endings": [["a", "b", "c", "d"]] * 2,
                             "label": ["3", "0"], "ind": [1, 2]}), p)
    rows = load_hellaswag(p)
    assert rows[0] == {"ctx": "A man is", "endings": ["a", "b", "c", "d"], "label": 3} and rows[1]["label"] == 0


def test_accuracy_in_unit_interval_with_real_tiny_model(tmp_path, tiny_cfg):
    from dataclasses import replace
    tok = tiny_tokenizer(tmp_path)
    m = Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size))
    rows = [{"ctx": "The quick brown", "endings": ["fox", "dog", "cat", "x"], "label": 0}] * 3
    acc = hellaswag_accuracy(m, tok, rows, "cpu", max_len=64, limit=2)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hellaswag.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `minids/eval/hellaswag.py`**

`CountingModel.forward` 的 logits 在 `token + 1` 处为 10.0、其余 0.0（`CountingModel` 已在 Task 6 建好，若其 logits 规则不够尖锐就在这里调整）。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hellaswag.py -q`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add minids/eval/hellaswag.py tests/test_hellaswag.py
git commit -m "feat(eval): HellaSwag download and normalized log-likelihood accuracy"
```

---

### Task 8: 评测套件

**Files:**
- Create: `minids/eval/suite.py`
- Modify: `tests/helpers_train.py`（加 `write_tiny_data`、`tiny_ladder`）
- Test: `tests/test_eval_suite.py`

**Interfaces:**
- Consumes：Task 5、6、7 的函数；`LadderConfig`（Task 1）；`HeimiTokenizer.load`
- Produces（`tests/helpers_train.py`）：
  ```python
  CODE_DOCS: list[str]   # ≥ 6 篇、每篇 ≥ 3 个非空行的小代码片段
  def write_tiny_data(root: Path, tok: HeimiTokenizer, seed: int = 0, train_tokens: int = 20_000) -> Path
      # root/tokens/{web,code}/train_0000.bin：[16, vocab) 的随机 token；val_0000.bin：CODE_DOCS 编码后以 0 分隔、重复到 ≥ 3000 token；
      # 用 minids.data.prepare.write_index 写 root/tokens/index.json；返回 index 路径
  def tiny_ladder(root: Path, stage_steps: tuple[int, ...] = (3, 2, 2),
                  ckpt_interval: int = 2, log_interval: int = 1, eval_interval: int = 2) -> LadderConfig
      # 先 tiny_tokenizer(root)（vocab 300）再 write_tiny_data；seq_len 32、micro_batch 4、tokens_per_step 256（累积 2 微步）；
      # 级结构同 conftest.tiny_cfg 但 vocab_size 300、seq_len 32；stage_steps 给 1 到 4 个数，依次对应
      # smoke0 none / smoke1 depth / smoke2 width / smoke_task none（mix 0.2/0.8，fim 0.5）；每级 tokens = steps × 256，lr 1e-3，warmup 0.2；
      # eval: val_windows 4、hellaswag False、fim_em_samples 3、sample_interval = eval_interval、sample_tokens 4；
      # prompts：一条普通 Prompt("en", "The cat") 与一条 FIM Prompt("fim", "def f():\n    ", "\n")；data 路径全指向 root 下
  ```
- Produces（`minids/eval/suite.py`）：
  ```python
  class EvalSuite:
      def __init__(self, cfg: LadderConfig, tok: HeimiTokenizer, device: str)
          # 预加载：web/code 固定窗口（cfg.eval.val_windows × (seq_len+1)）、FIM 样例（cases_from_index，seed=cfg.seed）、
          # HellaSwag 行（cfg.eval.hellaswag 为 True 时 load_hellaswag(cfg.data.hellaswag)，文件缺失 → FileNotFoundError 说明如何下载）
      def run(self, model, *, micro_batch: int, hellaswag: bool, fim: bool) -> dict
          # {"val_loss_web", "val_loss_code", "hellaswag_acc" (None 若不跑), "fim_em" (None 若不跑)}
          # HellaSwag 的 max_len = cfg.seq_len；FIM 生成预算：max_prefix_tokens = min(512, seq_len // 2)，max_suffix_tokens = min(256, seq_len // 4)，
          # max_new_tokens = min(64, seq_len - 前两者 - 3)
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_eval_suite.py
import pytest
from minids.data.tokenizer import HeimiTokenizer
from minids.eval.suite import EvalSuite
from minids.model.transformer import Transformer
from tests.helpers_train import tiny_ladder


def test_suite_runs_everything_but_hellaswag_when_disabled(tmp_path):
    cfg = tiny_ladder(tmp_path)
    tok = HeimiTokenizer.load(cfg.data.tokenizer)
    suite = EvalSuite(cfg, tok, "cpu")
    out = suite.run(Transformer(cfg.stages[0].model), micro_batch=4, hellaswag=True, fim=True)
    assert set(out) == {"val_loss_web", "val_loss_code", "hellaswag_acc", "fim_em"}
    assert out["val_loss_web"] > 0 and out["val_loss_code"] > 0
    assert out["hellaswag_acc"] is None and 0.0 <= out["fim_em"] <= 1.0


def test_fim_skipped_when_not_requested(tmp_path):
    cfg = tiny_ladder(tmp_path)
    tok = HeimiTokenizer.load(cfg.data.tokenizer)
    out = EvalSuite(cfg, tok, "cpu").run(Transformer(cfg.stages[0].model), micro_batch=4, hellaswag=False, fim=False)
    assert out["fim_em"] is None


def test_missing_hellaswag_file_is_a_clear_error(tmp_path):
    from dataclasses import replace
    cfg = tiny_ladder(tmp_path)
    cfg = replace(cfg, eval=replace(cfg.eval, hellaswag=True))
    with pytest.raises(FileNotFoundError, match="hellaswag"):
        EvalSuite(cfg, HeimiTokenizer.load(cfg.data.tokenizer), "cpu")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_suite.py -q`
Expected: FAIL，`ImportError: tiny_ladder` 或 `ModuleNotFoundError: minids.eval.suite`

- [ ] **Step 3: 实现 `write_tiny_data`、`tiny_ladder` 与 `minids/eval/suite.py`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_suite.py -q`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add minids/eval/suite.py tests/helpers_train.py tests/test_eval_suite.py
git commit -m "feat(eval): preloaded evaluation suite shared by the training loop and evaluate.py"
```

---

### Task 9: 优化器分组与进入新一级

**Files:**
- Create: `minids/train/optim.py`、`minids/train/grow.py`
- Test: `tests/test_optim.py`、`tests/test_grow_step.py`

**Interfaces:**
- Consumes：`grow_depth`、`grow_width`、`break_symmetry`、`verify_growth`、`GrowthReport.to_dict`、`StageConfig`
- Produces（`minids/train/optim.py`）：
  ```python
  def build_optimizer(model: Transformer, lr: float, weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW
      # 两个参数组：decay = ndim ≥ 2 且不是 model.tok_emb.weight；no_decay = 其余（RMSNorm 缩放、词嵌入）；参数在 cuda 上时 fused=True
  ```
- Produces（`minids/train/grow.py`）：
  ```python
  @dataclass(frozen=True)
  class GrowthOutcome:
      kind: str; params_before: int; params_after: int
      exact_report: dict | None      # none 时 None
      noisy_report: dict | None      # 只有 width 有
      seconds: float
  def grow_for_stage(prev: Transformer, stage: StageConfig, verify_tokens: torch.LongTensor, seed: int) -> tuple[Transformer, GrowthOutcome]
      # none：要求 stage.model == prev.cfg，原样返回 prev；
      # depth：grow_depth → verify_growth(rel_tol=0.0)，且 report.exact 必须 True；
      # width：grow_width(noise_std_rel=0.0) → verify_growth(rel_tol=1e-4) → break_symmetry(stage.noise_std_rel, Generator().manual_seed(seed))
      #        → verify_growth(rel_tol=stage.rel_tol)；
      # 最后断言新模型 cfg == stage.model，否则 ValueError（配置与算子不一致）
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_optim.py
from minids.model.transformer import Transformer
from minids.train.optim import build_optimizer


def test_groups_partition_params_and_exclude_embedding_and_norms(tiny_cfg):
    m = Transformer(tiny_cfg)
    opt = build_optimizer(m, lr=1e-3)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))
    ids_no = {id(p) for p in no_decay["params"]}
    assert id(m.tok_emb.weight) in ids_no and id(m.final_norm.weight) in ids_no and id(m.blocks[0].attn.kv_norm.weight) in ids_no
    assert id(m.lm_head.weight) not in ids_no and id(m.blocks[0].attn.wq.weight) not in ids_no
    assert opt.defaults["betas"] == (0.9, 0.95)
```

```python
# tests/test_grow_step.py
import pytest
import torch
from dataclasses import replace
from minids.data.loader import MixConfig
from minids.model.transformer import Transformer
from minids.train.config import StageConfig
from minids.train.grow import grow_for_stage


def _stage(model_cfg, growth):
    return StageConfig("s", growth, model_cfg, 256, 1e-3, 0.1, 4, MixConfig(), noise_std_rel=1e-3, rel_tol=1e-2)


def test_depth_is_exact(tiny_cfg):
    prev = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (2, 16))
    new, out = grow_for_stage(prev, _stage(tiny_cfg.grown_depth(), "depth"), tokens, seed=0)
    assert new.cfg == tiny_cfg.grown_depth() and out.kind == "depth"
    assert out.exact_report["exact"] is True and out.noisy_report is None
    assert out.params_after == Transformer(tiny_cfg.grown_depth()).count_params()


def test_width_verifies_twice_and_breaks_symmetry(tiny_cfg):
    prev = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (2, 16))
    new, out = grow_for_stage(prev, _stage(tiny_cfg.grown_width(), "width"), tokens, seed=0)
    assert out.exact_report["max_rel_err"] <= 1e-4 and out.noisy_report["max_rel_err"] <= 1e-2
    half = new.blocks[0].mlp.w_up.weight.shape[0] // 2
    w = new.blocks[0].mlp.w_up.weight
    assert not torch.equal(w[:half], w[half:])  # 对称已打破


def test_none_returns_same_object_or_rejects_mismatch(tiny_cfg):
    prev = Transformer(tiny_cfg)
    tokens = torch.randint(0, 256, (1, 8))
    same, out = grow_for_stage(prev, _stage(tiny_cfg, "none"), tokens, seed=0)
    assert same is prev and out.kind == "none" and out.exact_report is None
    with pytest.raises(ValueError):
        grow_for_stage(prev, _stage(replace(tiny_cfg, n_layer=3), "none"), tokens, seed=0)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_optim.py tests/test_grow_step.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `minids/train/optim.py` 与 `minids/train/grow.py`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_optim.py tests/test_grow_step.py -q`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add minids/train/optim.py minids/train/grow.py tests/test_optim.py tests/test_grow_step.py
git commit -m "feat(train): grouped AdamW and verified stage-entry growth with symmetry breaking"
```

---

### Task 10: 训练主循环

**Files:**
- Create: `minids/train/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes：Task 1 到 9 的全部接口；`MixedLoader`、`ShardSet.from_index`；`cuda_memory_snapshot`、`exceeds_memory_budget`（memguard）；`gpu_temperature_c`
- Produces：
  ```python
  class TrainingAborted(RuntimeError): ...   # nan_abort / oom / 显存超预算，消息含处理建议
  def run_ladder(cfg: LadderConfig, run_dir: Path, device: str, only_stage: str | None = None, max_steps: int | None = None) -> None
  ```

逐级流程（签名决定不了的部分）：

1. 读 `index.json` 的 `vocab_size`，与每级 `model.vocab_size` 不等 → `ValueError`。加载分词器（`HeimiTokenizer.load`）、`RunLogger(run_dir)`、`EvalSuite(cfg, tok, device)`。`ckpt_dir = run_dir / "ckpt"`。
2. 对每级（`only_stage` 给出时只处理它，但它的前一级必须有 `_final.pt`，否则 `FileNotFoundError` 写明缺的文件名）：
   - `<stage>_final.pt` 存在 → 跳过，不写任何文件。
   - 否则 `latest_step_checkpoint` 存在 → 核对检查点 `model_config == stage.model.to_dict()`（不等 → `ValueError`，消息含 "model_config"；`ladder_config` 不同只记一条 warning 日志），载入模型、优化器、`TrainState`，事件 `resume`（data：`checkpoint` 文件名）。
   - 否则进入新一级：首级 `torch.manual_seed(cfg.seed)` 后 `Transformer(stage.model)`；非首级 `load_model(前一级 _final.pt)`，取其 `state` 的 `global_step`、`micro_step`、`tokens_total` 作为起点，`grow_for_stage(prev, stage, verify_tokens=EvalSuite 的 web 固定窗口前 4 个[:, :-1], seed)`；`growth` 为 depth/width 时写事件 `growth`（data：`GrowthOutcome` 的字段展开），为 none 时不写（窗口按 `growth` 事件画竖线，数据切换不是增长）。`build_optimizer(model, stage.lr)`。事件 `stage_start`（data：`params_non_embedding`、`steps`、`accum_steps`、`micro_batch`、`growth`）。
   - `MixedLoader({web, code: ShardSet.from_index(index, 来源, "train", seq_len + 1)}, stage.mix, batch_size=stage.micro_batch, seq_len, seed=cfg.seed, vocab_size)`。
   - `compile` 开启时 `torch.compile(model)`；第一个微步前向抛任何异常 → 事件 `compile_fallback`，换回原模型重做该微步。
3. 每个优化步：`lr = lr_at(state.step, steps, stage.lr, warmup_steps(steps, stage.warmup_frac))` 写进两个参数组；累积 `accum_steps` 个微步：`x, y = loader.get_batch(state.micro_step)` → `to(device, non_blocking=True)` → cuda 上 `torch.autocast("cuda", dtype=torch.bfloat16)` → `(loss / accum).backward()` → `state.micro_step += 1`、`tokens_stage/tokens_total += micro_batch × seq_len`；`grad_norm = clip_grad_norm_(1.0)`；累计 loss 非有限 → 事件 `nan_abort` → `TrainingAborted`（消息含 "nan_abort"，提示从最近检查点恢复并考虑调低学习率）；`opt.step(); opt.zero_grad(set_to_none=True)`；`state.step += 1; state.global_step += 1`。`torch.cuda.OutOfMemoryError` → 事件 `oom` → `TrainingAborted`（提示微批次减半）。
4. 第一个优化步后（cuda）：`cuda_memory_snapshot()`，`exceeds_memory_budget(reserved, total, cfg.max_mem_frac)` 为真 → 事件 `oom`（data：`reserved_gb`、`total_gb`、`hint`）→ `TrainingAborted`。
5. 每 `log_interval` 步及本级最后一步：`MetricsRecord`，`tok_per_s` 为上次记录以来的 token 数除以耗时，`mem_gb` 为 `torch.cuda.max_memory_reserved()/1e9`（随后 `reset_peak_memory_stats`，CPU 上 0.0），`gpu_temp_c = gpu_temperature_c()`（device 不是 cuda 时不调用，直接 None）。
6. 每 `eval.interval` 步及本级最后一步：`suite.run(model, micro_batch, hellaswag=是最后一步 and cfg.eval.hellaswag, fim=是最后一步 or stage.mix.fim_rate > 0)` → `EvalRecord`。每 `eval.sample_interval` 步及最后一步：`generate_samples(..., seed=cfg.seed + state.global_step)` → 每条一个 `SampleRecord`。评测与采样在 `model.eval()` 下做，之后 `model.train()`。
7. 每 `ckpt_interval` 步：`save_checkpoint(ckpt_dir / checkpoint_name(stage, state.step), ...)`，事件 `checkpoint`，`prune_step_checkpoints(keep=cfg.keep_checkpoints)`。
8. `max_steps`：本次调用累计执行的优化步数达到它 → 保存步检查点、事件 `checkpoint`（data：`reason: "max_steps"`）、返回。
9. 本级结束：保存 `_final.pt`（含优化器），事件 `stage_end`（data：`loss_last`、`params_non_embedding`、`seconds`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_loop.py
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
    from dataclasses import replace
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
    from dataclasses import replace
    cfg = tiny_ladder(tmp_path, stage_steps=(1,))
    bad = replace(cfg.stages[0], model=replace(cfg.stages[0].model, vocab_size=512))
    cfg = replace(cfg, stages=(bad,))
    with pytest.raises(ValueError, match="vocab"):
        run_ladder(cfg, tmp_path / "run", "cpu")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_loop.py -q`
Expected: FAIL，`ModuleNotFoundError: minids.train.loop`

- [ ] **Step 3: 实现 `minids/train/loop.py`**

把上面的九条流程拆成私有函数（进入新一级、训练一级、记指标、做评测与采样、存检查点），`run_ladder` 只做编排。评测/采样期间 `model.eval()`，之后恢复 `train()`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_loop.py -q`
Expected: 7 passed（CPU 上 1 分钟内）

- [ ] **Step 5: 跑全套，确认没有新的 warning**

Run: `.venv/Scripts/python.exe -m pytest -q -W error`
Expected: 全部 passed

- [ ] **Step 6: 提交**

```bash
git add minids/train/loop.py tests/test_loop.py
git commit -m "feat(train): resumable ladder training loop with growth, logging, eval and checkpoints"
```

---

### Task 11: 入口 `train.py`、烟雾数据与烟雾配置

**Files:**
- Create: `train.py`、`scripts/make_smoke_data.py`、`configs/tiny_smoke.yaml`
- Modify: `.gitignore`（加 `/models/`）
- Test: `tests/test_train_cli.py`、`tests/test_smoke_data.py`

**Interfaces:**
- Consumes：`run_ladder`、`load_ladder`、`keep_awake`、`download_hellaswag`、`write_index`（prepare.py）、`write_shard`
- Produces（`train.py`）：`def main(argv: Sequence[str] | None = None) -> int`，参数 `--config`（必填）、`--run`（必填，run 名）、`--runs-dir`（默认 `runs`）、`--stage`、`--device`（默认有 cuda 则 cuda 否则 cpu）、`--max-steps`。流程：`logging.basicConfig(INFO)` → `load_ladder` → `cfg.eval.hellaswag` 且文件不存在时 `download_hellaswag` → `with keep_awake(): run_ladder(...)`；`TrainingAborted` 打印消息返回 3；正常返回 0。
- Produces（`scripts/make_smoke_data.py`）：`def main(argv) -> int`，参数 `--out`（默认 `data/smoke`）、`--vocab`（默认 16384）、`--tokens`（默认 400000，每来源每 split）、`--seed`；`main` 调用 `write_smoke_data(out: Path, vocab: int, tokens: int, seed: int) -> Path`：web/code 的 train/val 各一片随机 token（范围 `[16, vocab)`，每 300 个 token 插一个 0），`write_index(out, per_source, vocab)`，`per_source` 用 `prepare_source` 返回值同样的键（`docs`、`skipped`、`tokens_train`、`tokens_val`、`val_rows`、`train`、`val`）。
- `configs/tiny_smoke.yaml`（spec 13.2）：seed 0、seq_len 128、tokens_per_step 2048、log_interval 5、ckpt_interval 25、keep 2；data index `data/smoke/index.json`、tokenizer `data/tokenizer/heimi-16k.json`；eval interval 25、val_windows 8、hellaswag false、fim_em_samples 20、sample_interval 25、sample_tokens 16；顶层 model vocab 16384、seq_len 128；级 `smoke0` none（2 层 d64 2 头 qk_nope 16 rope 8 v 32 kv 32 ffn 128，61440 token = 30 步，lr 1.0e-3，warmup 0.1，micro 8，mix 0.85/0.15）→ `smoke1` depth（4 层，40960 = 20 步）→ `smoke2` width（d128 4 头 kv 64 ffn 256，40960 = 20 步）→ `smoke_task` none（20480 = 10 步，mix 0.2/0.8，fim 0.5，lr 5.0e-4）；prompts 与 `ladder.yaml` 相同。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_smoke_data.py
import json
from scripts.make_smoke_data import write_smoke_data


def test_smoke_data_layout_matches_index(tmp_path):
    idx = write_smoke_data(tmp_path, vocab=300, tokens=5000, seed=0)
    d = json.loads(idx.read_text(encoding="utf-8"))
    assert d["vocab_size"] == 300 and set(d["sources"]) == {"web", "code"}
    for src in ("web", "code"):
        for split in ("train", "val"):
            e = d["sources"][src][split][0]
            assert (tmp_path / src / e["file"]).stat().st_size == 2 * e["tokens"] == 2 * 5000
```

```python
# tests/test_train_cli.py
import yaml
import train
from minids.logging_io import read_jsonl
from tests.helpers_train import tiny_ladder


def test_cli_trains_with_max_steps_and_exits_zero(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(3,))
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    rc = train.main(["--config", str(p), "--run", "t", "--runs-dir", str(tmp_path / "runs"), "--device", "cpu", "--max-steps", "2"])
    assert rc == 0
    assert (tmp_path / "runs" / "t" / "ckpt" / "smoke0_step2.pt").exists()
    assert len(read_jsonl(tmp_path / "runs" / "t" / "metrics.jsonl")) == 2


def test_tiny_smoke_yaml_parses():
    from minids.train.config import load_ladder
    cfg = load_ladder("configs/tiny_smoke.yaml")
    assert [s.name for s in cfg.stages] == ["smoke0", "smoke1", "smoke2", "smoke_task"]
    assert [cfg.steps_for(s) for s in cfg.stages] == [30, 20, 20, 10] and cfg.eval.hellaswag is False
```

`tests/test_smoke_data.py` 以 `scripts.make_smoke_data` 导入，所以要新建空的 `scripts/__init__.py`（`tests/` 已是包，pytest 会把仓库根目录放进 `sys.path`，`import train` 同理）。

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke_data.py tests/test_train_cli.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 `train.py`、`scripts/make_smoke_data.py`、`configs/tiny_smoke.yaml`，`.gitignore` 加 `/models/`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke_data.py tests/test_train_cli.py -q`
Expected: 3 passed

- [ ] **Step 5: GPU 烟雾测试（spec 13.2）**

先确认 GPU 空闲：`nvidia-smi --query-gpu=memory.used --format=csv,noheader` 低于 2000 MiB，否则等待。

Run:
```
.venv/Scripts/python.exe scripts/make_smoke_data.py
set PYTHONUTF8=1 && .venv/Scripts/python.exe train.py --config configs/tiny_smoke.yaml --run smoke --device cuda
```
Expected: 1 分钟内结束；`runs/smoke/` 下四个 jsonl 非空；`events.jsonl` 含 2 条 `growth`（depth `exact: true`，width `max_rel_err ≤ 1e-2`）与 4 条 `stage_end`；`metrics.jsonl` 最后一条 `smoke0` 的 loss 低于第一条；`ckpt/` 下四个 `_final.pt`。

再测恢复：
```
.venv/Scripts/python.exe train.py --config configs/tiny_smoke.yaml --run smoke2 --device cuda --max-steps 12
.venv/Scripts/python.exe train.py --config configs/tiny_smoke.yaml --run smoke2 --device cuda
```
Expected: 第二次运行 `events.jsonl` 有一条 `resume`，最终也到 `smoke_task_final.pt`。把两次运行的用时写进提交信息。

- [ ] **Step 6: 提交**

```bash
git add train.py scripts/make_smoke_data.py scripts/__init__.py configs/tiny_smoke.yaml .gitignore tests/test_smoke_data.py tests/test_train_cli.py
git commit -m "feat: train.py entry, synthetic smoke data and tiny smoke ladder (GPU smoke run passes)"
```

---

### Task 12: `evaluate.py` 与 `sample.py`

**Files:**
- Create: `evaluate.py`、`sample.py`
- Test: `tests/test_eval_cli.py`

**Interfaces:**
- Consumes：`load_model`、`EvalSuite`、`load_ladder`、`RunLogger`、`EvalRecord`、`HeimiTokenizer.load`、`Transformer.generate`
- Produces（`evaluate.py`）：`main(argv) -> int`，参数 `--ckpt`（必填）、`--config`（默认 `configs/ladder.yaml`，提供数据路径与评测设置）、`--device`、`--no-hellaswag`、`--fim-samples`（覆盖 `eval.fim_em_samples`）、`--out`（默认 ckpt 所在 run 目录即 `ckpt_path.parent.parent` 的 `eval.jsonl`）。用检查点 `state` 里的 `stage`、`global_step`、`tokens_total` 构造 `EvalRecord`，追加并把 JSON 打印到标准输出。
- Produces（`sample.py`）：`main(argv) -> int`，参数 `--ckpt`（必填）、`--prompt`（必填）、`--tokenizer`（默认 `data/tokenizer/heimi-16k.json`）、`--max-tokens 64`、`--temperature 0.8`、`--top-k 50`、`--seed 0`、`--device`。打印提示加续写。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_eval_cli.py
import json
import yaml
import evaluate
import sample
from minids.logging_io import read_jsonl
from minids.train.loop import run_ladder
from tests.helpers_train import tiny_ladder


def _trained(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(2,))
    run_ladder(cfg, tmp_path / "run", "cpu")
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg, tmp_path / "run" / "ckpt" / "smoke0_final.pt", p


def test_evaluate_appends_record_and_prints_json(tmp_path, capsys):
    cfg, ckpt, yml = _trained(tmp_path)
    n_before = len(read_jsonl(tmp_path / "run" / "eval.jsonl"))
    assert evaluate.main(["--ckpt", str(ckpt), "--config", str(yml), "--device", "cpu", "--fim-samples", "2"]) == 0
    recs = read_jsonl(tmp_path / "run" / "eval.jsonl")
    assert len(recs) == n_before + 1 and recs[-1]["stage"] == "smoke0" and recs[-1]["global_step"] == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["val_loss_web"] == recs[-1]["val_loss_web"]


def test_sample_prints_prompt_and_continuation(tmp_path, capsys):
    cfg, ckpt, _ = _trained(tmp_path)
    assert sample.main(["--ckpt", str(ckpt), "--tokenizer", cfg.data.tokenizer, "--prompt", "The cat", "--max-tokens", "3", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("The cat") and len(out) > len("The cat")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_cli.py -q`
Expected: FAIL，`ModuleNotFoundError: evaluate`

- [ ] **Step 3: 实现 `evaluate.py` 与 `sample.py`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_cli.py -q`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add evaluate.py sample.py tests/test_eval_cli.py
git commit -m "feat: evaluate.py and sample.py entry points"
```

---

### Task 13: 导出与命令行补全

**Files:**
- Create: `minids/export.py`、`export.py`、`complete.py`
- Modify: `pyproject.toml`（dependencies 加 `safetensors>=0.4`；执行前 `uv pip install safetensors`）
- Test: `tests/test_export_complete.py`

**Interfaces:**
- Consumes：`load_model`、`ModelConfig`、`HeimiTokenizer.load`、`complete_middle`、`safetensors.torch.save_file / load_file`
- Produces（`minids/export.py`）：
  ```python
  def export_model(ckpt_path: Path, tokenizer_path: Path, out_dir: Path) -> Path
      # out_dir/model.safetensors（state_dict）、out_dir/config.json（{"name": "heimi", "model": ModelConfig dict, "stage", "global_step",
      # "tokens_total", "git_commit", "exported_at"}）、out_dir/tokenizer.json（原样复制）；返回 out_dir
  def load_exported(model_dir: Path, device: str) -> tuple[Transformer, HeimiTokenizer]   # eval 模式
  ```
- Produces（`export.py`）：`main(argv) -> int`，参数 `--ckpt`（必填）、`--out`（必填）、`--tokenizer`（默认 `data/tokenizer/heimi-16k.json`）。
- Produces（`complete.py`，spec 11.3）：`main(argv) -> int`，参数 `--model DIR`（导出目录）或 `--ckpt PATH`（二选一，必有其一；用 `--ckpt` 时另需 `--tokenizer`）、`--prefix-file`、`--suffix-file`、`--max-tokens 64`、`--device cpu`。没有 `--prefix-file` 就从标准输入读前缀；没有后缀文件则后缀为空串。把 `complete_middle(...)` 的结果打印到标准输出（一行）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_export_complete.py
import json
import torch
import complete
import export as export_cli
from minids.export import export_model, load_exported
from minids.train.loop import run_ladder
from tests.helpers_train import tiny_ladder


def _ckpt(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(1,))
    run_ladder(cfg, tmp_path / "run", "cpu")
    return cfg, tmp_path / "run" / "ckpt" / "smoke0_final.pt"


def test_export_then_load_gives_identical_logits(tmp_path):
    cfg, ckpt = _ckpt(tmp_path)
    out = export_model(ckpt, cfg.data.tokenizer, tmp_path / "exported")
    assert {p.name for p in out.iterdir()} == {"model.safetensors", "config.json", "tokenizer.json"}
    meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert meta["name"] == "heimi" and meta["stage"] == "smoke0" and meta["model"]["vocab_size"] == 300
    m, tok = load_exported(out, "cpu")
    from minids.train.checkpoint import load_model
    ref, _ = load_model(ckpt, "cpu")
    x = torch.randint(16, 300, (1, 8))
    assert torch.equal(m(x)[0], ref(x)[0]) and tok.vocab_size == 300 and not m.training


def test_export_cli_and_complete_cli_round_trip(tmp_path, capsys, monkeypatch):
    cfg, ckpt = _ckpt(tmp_path)
    assert export_cli.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "exp"), "--tokenizer", cfg.data.tokenizer]) == 0
    (tmp_path / "p.txt").write_text("def f():\n    ", encoding="utf-8")
    (tmp_path / "s.txt").write_text("\nprint(f())\n", encoding="utf-8")
    rc = complete.main(["--model", str(tmp_path / "exp"), "--prefix-file", str(tmp_path / "p.txt"),
                        "--suffix-file", str(tmp_path / "s.txt"), "--max-tokens", "4", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0 and out.endswith("\n") and out.count("\n") == 1


def test_complete_reads_prefix_from_stdin(tmp_path, capsys, monkeypatch):
    import io
    cfg, ckpt = _ckpt(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("x = "))
    rc = complete.main(["--ckpt", str(ckpt), "--tokenizer", cfg.data.tokenizer, "--max-tokens", "3", "--device", "cpu"])
    assert rc == 0 and capsys.readouterr().out.count("\n") == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_export_complete.py -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 装依赖并实现 `minids/export.py`、`export.py`、`complete.py`**

Run: `uv pip install --python .venv/Scripts/python.exe safetensors` 后写代码。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_export_complete.py -q`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add minids/export.py export.py complete.py pyproject.toml tests/test_export_complete.py
git commit -m "feat: safetensors export and command-line FIM completion"
```

---

### Task 14: 级 0 实跑、级 1 入口试跑与报告

**Files:**
- Create: `docs/training/stage0-report.md`

**Interfaces:**
- Consumes：Task 11 到 13 的全部入口；真实数据 `data/tokens/`、`data/tokenizer/heimi-16k.json`

- [ ] **Step 1: 确认 GPU 空闲与磁盘**

Run: `nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader`
Expected: 已用显存低于 2000 MiB（写计划时有游戏占 13.6 GB，若仍被占用则等用户让出 GPU 再继续）。`runs/` 预计写入约 3 个级 0 检查点（每个约 160 MB）。

- [ ] **Step 2: 下载 HellaSwag 并跑级 0**

Run（后台执行，用 `PYTHONUTF8=1` 防止日志乱码）：
```
set PYTHONUTF8=1 && .venv/Scripts/python.exe train.py --config configs/ladder.yaml --run heimi --stage stage0 --device cuda
```
Expected: `data/eval/hellaswag_val.parquet` 出现（6.3 MB）；572 步约 25 到 30 分钟；`metrics.jsonl` 的 `tok_per_s` 在 180k 到 230k 之间、`mem_gb` 低于 9；`eval.jsonl` 有 2 条（step 500 与级末），级末 `hellaswag_acc` 在 0.24 到 0.30 之间、`fim_em` 接近 0；`ckpt/stage0_final.pt` 存在，`stage0_step*.pt` 最多 2 个。期间每 5 分钟看一次 `metrics.jsonl` 尾部确认损失下降、无 `nan_abort`/`oom` 事件。

- [ ] **Step 3: 级 1 入口试跑（真实模型上的加深与恢复）**

Run:
```
set PYTHONUTF8=1 && .venv/Scripts/python.exe train.py --config configs/ladder.yaml --run heimi --device cuda --max-steps 20
```
Expected: 跳过 stage0，`events.jsonl` 新增 `growth`（`kind: depth`，`exact_report.exact: true`，`params_before` 12,934,528、`params_after` 25,869,056 左右）、`stage_start`，20 步后 `checkpoint`（`reason: max_steps`），`ckpt/stage1_step20.pt` 存在；`metrics.jsonl` 里 stage1 第一条 loss 与 stage0 最后一条相差不超过 0.1（增长点不跳升）。记录 stage1 的 `tok_per_s`（预期 120k 到 135k）。

- [ ] **Step 4: 评测、采样、导出、补全各跑一次**

Run:
```
.venv/Scripts/python.exe evaluate.py --ckpt runs/heimi/ckpt/stage0_final.pt --device cuda
.venv/Scripts/python.exe sample.py --ckpt runs/heimi/ckpt/stage0_final.pt --prompt "The history of the Roman Empire began when" --device cuda
.venv/Scripts/python.exe export.py --ckpt runs/heimi/ckpt/stage0_final.pt --out models/heimi-stage0
echo def add(a, b):>prefix.txt && .venv/Scripts/python.exe complete.py --model models/heimi-stage0 --prefix-file prefix.txt --max-tokens 32
```
Expected: `evaluate.py` 打印的 `val_loss_web`/`val_loss_code` 与 `eval.jsonl` 级末那条相差不超过 0.01（HellaSwag 与 FIM 随机种子相同应完全一致）；`sample.py` 输出英文片段；`models/heimi-stage0/` 三个文件；`complete.py` 输出一行。删掉临时 `prefix.txt`。

- [ ] **Step 5: 写 `docs/training/stage0-report.md`**

内容：运行命令；级 0 的步数、用时、平均 tok/s、峰值显存、GPU 温度范围；训练损失首末值与两次验证损失（web、code）；HellaSwag、FIM 精确匹配；级 1 入口的增长事件数字与增长点前后损失；与 `docs/benchmarks/throughput.md` 预测的对比；观察到的问题；剩余各级的运行命令与预计用时（stage1 约 1 小时、stage2 约 11 小时、task 约 4 小时、baseline 约 11 小时，均可用 `train.py --config ... --run heimi` 一条命令续跑，中断后重跑同一命令即恢复）；以及 spec 7.3 留下的校准项：进入 stage2 的加宽验证若因 `max_rel_err > rel_tol` 中止，调小 `ladder.yaml` 里 stage2 的 `noise_std_rel` 或放宽 `rel_tol` 后重跑同一命令即可（增长在进入该级时重新执行）。

- [ ] **Step 6: 提交**

```bash
git add docs/training/stage0-report.md
git commit -m "docs(training): stage 0 real-data run and stage 1 entry report"
```

---

## 完成标准

- `.venv/Scripts/python.exe -m pytest -q -W error` 全部通过（计划二后的 89 个加本计划新增的约 65 个），CPU 上 3 分钟内。
- `configs/tiny_smoke.yaml` 在 GPU 上 1 分钟内跑完四级，日志四个文件齐全，两次增长验证通过，中断后能恢复。
- `runs/heimi/ckpt/stage0_final.pt` 与 `stage1_step20.pt` 存在；`runs/heimi/` 四个 jsonl 能被计划四直接读取。
- `evaluate.py`、`sample.py`、`export.py`、`complete.py` 都在真实检查点上跑通。
- `docs/training/stage0-report.md` 给出实测数字与剩余各级的运行命令。
