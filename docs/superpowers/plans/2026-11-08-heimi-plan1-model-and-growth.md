# 黑米 计划一：环境、模型与增长算子 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭好 CUDA 训练环境，手写 DeepSeek 风格的 MLA 稠密模型，实现并用精确测试验证"加深"和"加宽"两种保函数增长算子，并实测三级结构在本机的训练吞吐。

**Architecture:** `minids/model/` 下每个文件一个组件（RMSNorm、RoPE、MLA、SwiGLU、Block、Transformer），`minids/growth/` 只依赖模型的公开属性名完成权重变换，`verify.py` 用真实前向比较增长前后的 logits。全部单元测试在 CPU fp32 上用迷你配置运行，GPU 只用于环境检查和吞吐基准。

**Tech Stack:** Python 3.12、uv、PyTorch（Windows CUDA 版，cu128 或 cu130）、numpy、pyyaml、pytest。

**Spec:** `docs/superpowers/specs/2026-11-08-heimi-growth-lm-design.md`（本计划实现其第 3、4、5、7 节及第 13.1 节中对应的测试；第 16 节"吞吐低于估计"的应对在 Task 12 落地）

## 本计划在整体中的位置

四份计划顺序执行：① 环境、模型与增长算子（本份）→ ② 数据管线（spec 第 6 节）→ ③ 训练、评测与任务阶段（spec 第 8 到 11、14 节）→ ④ 窗口（spec 第 12 节）。本份完成时，仓库里有可实例化的三级模型、通过精确测试的增长算子和一份吞吐基准报告，但还不能用真实数据训练。

## Global Constraints

- Python 3.12；虚拟环境 `.venv` 由 `uv` 创建。
- PyTorch 必须是 Windows CUDA 版，`torch.cuda.is_available()` 为 True 且 `torch.cuda.get_device_capability(0)[0] == 12`（Blackwell）。
- 不使用 `transformers`、Megatron 等框架；模型与算子全部手写。
- 全模型无偏置项：所有 `nn.Linear` 使用 `bias=False`。
- 注意力只用 `torch.nn.functional.scaled_dot_product_attention`，不装 flash-attn。
- 默认结构参数来自 spec 5.3：vocab 16384、seq_len 1024、RoPE base 10000、RMSNorm eps 1e-6、初始化标准差 0.02，残差输出投影再乘 `1/sqrt(2·n_layer)`。
- 词嵌入与输出头不共享权重。模型大小按"非嵌入参数"计（总参数减去 `tok_emb` 和 `lm_head`）。
- 单元测试在 CPU、fp32 上运行，不依赖 GPU；只有 `tests/test_env.py` 和 `scripts/bench_throughput.py` 需要 GPU。
- 注释用中文，标识符用英文。
- 每个 commit 信息用英文 conventional commits 格式，末尾加一行 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。

## Review Focus

1. 输入序列长度超过 `seq_len`：`Transformer.forward` 应抛出带有实际长度与上限的 `ValueError`，而不是在 RoPE 缓存切片处报形状错误。测试在 Task 8。
2. token id 超出词表（数据分片损坏时会发生）：`Transformer.forward` 应抛出说明最大 id 与词表大小的 `ValueError`，而不是 CUDA 端的 device-side assert。测试在 Task 8。
3. `generate` 的 `temperature=0`：应退化为贪心解码，不能出现除零或 NaN。测试在 Task 8。
4. `generate` 的提示长度超过 `seq_len`：应只保留最后 `seq_len` 个 token 作为条件继续生成，而不是报错。测试在 Task 8。
5. RMSNorm 输入全零：输出必须是有限值（eps 起作用），不能 NaN。测试在 Task 3。

---

## 文件结构

```
pyproject.toml                 # 项目元数据、依赖、pytest 配置
.gitignore                     # .venv/ runs/ data/ __pycache__/ 等
.python-version                # 3.12
minids/__init__.py
minids/config.py               # ModelConfig：字段、校验、yaml 读写、增长后的配置
minids/model/__init__.py
minids/model/norm.py           # RMSNorm
minids/model/rope.py           # RoPE 缓存与应用
minids/model/mla.py            # MLA 注意力
minids/model/mlp.py            # SwiGLU
minids/model/block.py          # pre-norm Transformer 块
minids/model/transformer.py    # 整模型、初始化、生成、参数统计
minids/growth/__init__.py
minids/growth/depth.py         # grow_depth
minids/growth/width.py         # grow_width
minids/growth/verify.py        # verify_growth / GrowthReport
scripts/bench_throughput.py    # GPU 吞吐基准
docs/benchmarks/throughput.md  # 基准结果（Task 12 产出）
tests/conftest.py              # tiny_cfg、stage0_cfg、stage2_cfg 夹具
tests/test_env.py
tests/test_config.py
tests/test_norm.py
tests/test_rope.py
tests/test_mla.py
tests/test_mlp.py
tests/test_block.py
tests/test_transformer.py
tests/test_growth_depth.py
tests/test_growth_width.py
tests/test_growth_verify.py
```

---

### Task 1: 项目骨架与 CUDA 环境

**Files:**
- Create: `pyproject.toml`、`.gitignore`、`.python-version`、`minids/__init__.py`、`minids/model/__init__.py`、`minids/growth/__init__.py`、`tests/__init__.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Produces: 可运行的 `.venv`，`pytest` 可用，`import minids` 成功。

- [ ] **Step 1: 写失败的环境测试**

```python
# tests/test_env.py
import torch

def test_cuda_available():
    assert torch.cuda.is_available()

def test_blackwell_capability():
    major, _ = torch.cuda.get_device_capability(0)
    assert major == 12

def test_bf16_supported():
    assert torch.cuda.is_bf16_supported()
```

- [ ] **Step 2: 创建 pyproject 与骨架文件**

`pyproject.toml`：`[build-system] requires = ["setuptools>=68"]`、`build-backend = "setuptools.build_meta"`；`[project] name = "minids"`，`version = "0.1.0"`，`requires-python = ">=3.12,<3.13"`，`dependencies = ["numpy>=1.26", "pyyaml>=6"]`，`[project.optional-dependencies] dev = ["pytest>=8"]`，`[tool.pytest.ini_options] testpaths = ["tests"]`，`[tool.setuptools.packages.find] include = ["minids*"]`。torch 不写进依赖（由 PyTorch 索引单独安装）。
`.gitignore`：`.venv/`、`runs/`、`data/`、`__pycache__/`、`*.pyc`、`.pytest_cache/`、`*.egg-info/`、`docs/benchmarks/*.jsonl`。
`.python-version`：`3.12`。四个 `__init__.py` 为空。

- [ ] **Step 3: 建环境并安装 CUDA 版 torch**

```bash
cd F:/mini-deepseek
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

若 cu128 索引没有 cp312 的 Windows 轮子，改用 `https://download.pytorch.org/whl/cu130`。

- [ ] **Step 4: 运行环境测试**

Run: `.venv/Scripts/python -m pytest tests/test_env.py -v`
Expected: 3 passed，且 `python -c "import torch;print(torch.__version__)"` 输出带 `+cu128` 或 `+cu130`。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .python-version minids tests/__init__.py tests/test_env.py
git commit -m "chore: project skeleton and CUDA torch environment"
```

---

### Task 2: ModelConfig

**Files:**
- Create: `minids/config.py`
- Test: `tests/test_config.py`、`tests/conftest.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ModelConfig:
      vocab_size: int = 16384
      seq_len: int = 1024
      n_layer: int = 8
      d_model: int = 384
      n_head: int = 6
      qk_nope_head_dim: int = 48
      qk_rope_head_dim: int = 16
      v_head_dim: int = 64
      kv_lora_rank: int = 128
      ffn_hidden: int = 1024
      rope_base: float = 10000.0
      norm_eps: float = 1e-6
      init_std: float = 0.02
      @property
      def qk_head_dim(self) -> int: ...          # qk_nope_head_dim + qk_rope_head_dim
      def validate(self) -> None: ...            # 非法时 raise ValueError
      def to_dict(self) -> dict: ...
      @classmethod
      def from_dict(cls, d: dict) -> "ModelConfig": ...
      def to_yaml(self, path: str | Path) -> None: ...
      @classmethod
      def from_yaml(cls, path: str | Path) -> "ModelConfig": ...
      def grown_depth(self) -> "ModelConfig": ...   # n_layer * 2
      def grown_width(self) -> "ModelConfig": ...   # d_model, n_head, kv_lora_rank, ffn_hidden 各 * 2
  ```
  `__post_init__` 调用 `validate()`。校验规则：所有整数字段 > 0；`qk_rope_head_dim` 为偶数；`vocab_size >= 16`。
- conftest 夹具：
  ```python
  tiny_cfg  = ModelConfig(vocab_size=256, seq_len=64, n_layer=2, d_model=64, n_head=2,
                          qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=32,
                          kv_lora_rank=32, ffn_hidden=128)
  stage0_cfg = ModelConfig()                       # 默认值即 spec 级 0
  stage2_cfg = ModelConfig(n_layer=16, d_model=768, n_head=12, kv_lora_rank=256, ffn_hidden=2048)
  ```

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_config.py
def test_defaults_match_spec_stage0(stage0_cfg):
    assert (stage0_cfg.n_layer, stage0_cfg.d_model, stage0_cfg.n_head) == (8, 384, 6)
    assert stage0_cfg.qk_head_dim == 64

def test_grown_depth_doubles_layers(stage0_cfg):
    g = stage0_cfg.grown_depth()
    assert g.n_layer == 16 and g.d_model == 384

def test_grown_width_doubles_width_only(stage0_cfg):
    g = stage0_cfg.grown_width()
    assert (g.d_model, g.n_head, g.kv_lora_rank, g.ffn_hidden) == (768, 12, 256, 2048)
    assert (g.n_layer, g.qk_nope_head_dim, g.qk_rope_head_dim, g.v_head_dim) == (8, 48, 16, 64)

def test_yaml_round_trip(tmp_path, stage2_cfg):
    p = tmp_path / "c.yaml"; stage2_cfg.to_yaml(p)
    assert ModelConfig.from_yaml(p) == stage2_cfg

def test_invalid_rope_dim_raises():
    with pytest.raises(ValueError):
        ModelConfig(qk_rope_head_dim=15)
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: minids.config`

- [ ] **Step 3: 实现 `minids/config.py` 与 `tests/conftest.py`**

按 Interfaces 的签名实现；yaml 用 `yaml.safe_dump` / `yaml.safe_load`。

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add minids/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: ModelConfig with validation, yaml io and growth derivations"
```

---

### Task 3: RMSNorm

**Files:**
- Create: `minids/model/norm.py`
- Test: `tests/test_norm.py`

**Interfaces:**
- Produces: `class RMSNorm(nn.Module)`，`__init__(self, dim: int, eps: float = 1e-6)`，参数 `self.weight = nn.Parameter(torch.ones(dim))`，`forward(self, x: Tensor) -> Tensor`。内部在 fp32 计算 `x * rsqrt(mean(x²) + eps)`，再乘 `weight`，返回值转回输入 dtype。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_norm.py
def test_output_rms_is_one():
    norm = RMSNorm(64); x = torch.randn(4, 10, 64) * 3
    y = norm(x)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4, 10), atol=1e-4)

def test_weight_scales_output():
    norm = RMSNorm(8); norm.weight.data.fill_(2.0)
    x = torch.randn(2, 8)
    assert torch.allclose(norm(x), 2.0 * RMSNorm(8)(x))

def test_zero_input_is_finite():
    assert torch.isfinite(RMSNorm(8)(torch.zeros(2, 8))).all()

def test_preserves_dtype():
    assert RMSNorm(8)(torch.randn(2, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_norm.py -v` → FAIL（模块不存在）

- [ ] **Step 3: 实现 `minids/model/norm.py`**

- [ ] **Step 4: 运行，确认通过** → 4 passed

- [ ] **Step 5: Commit**：`git commit -m "feat: RMSNorm"`

---

### Task 4: RoPE

**Files:**
- Create: `minids/model/rope.py`
- Test: `tests/test_rope.py`

**Interfaces:**
- Produces:
  ```python
  RopeCache = tuple[Tensor, Tensor]   # (cos, sin)，形状各为 (seq_len, dim)
  def precompute_rope(dim: int, seq_len: int, base: float = 10000.0,
                      device=None, dtype=torch.float32) -> RopeCache
  def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor
      # x: (B, h, T, dim)；cos/sin 传入前已切成 (T, dim)；rotate-half 约定（GPT-NeoX 式）：
      # x * cos + rotate_half(x) * sin，rotate_half(x) = cat(-x2, x1)，x1/x2 为前后两半
  ```
  `dim` 必须为偶数，否则 `ValueError`。频率 `inv_freq[i] = base^(-2i/dim)`，`i in [0, dim/2)`，`cos/sin` 的后半与前半相同（`cat(freqs, freqs)`）。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_rope.py
def test_shapes():
    cos, sin = precompute_rope(16, 32)
    assert cos.shape == sin.shape == (32, 16)

def test_relative_position_property():
    dim, T = 16, 40
    cos, sin = precompute_rope(dim, T)
    q = torch.randn(1, 1, 1, dim); k = torch.randn(1, 1, 1, dim)
    def score(m, n):
        qm = apply_rope(q, cos[m:m+1], sin[m:m+1]); kn = apply_rope(k, cos[n:n+1], sin[n:n+1])
        return (qm * kn).sum()
    assert torch.allclose(score(3, 1), score(13, 11), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)

def test_position_zero_is_identity():
    cos, sin = precompute_rope(8, 4); x = torch.randn(1, 2, 1, 8)
    assert torch.allclose(apply_rope(x, cos[0:1], sin[0:1]), x)

def test_odd_dim_raises():
    with pytest.raises(ValueError):
        precompute_rope(7, 4)
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `minids/model/rope.py`**
- [ ] **Step 4: 运行，确认通过** → 4 passed
- [ ] **Step 5: Commit**：`git commit -m "feat: rotary position embedding"`

---

### Task 5: MLA 注意力

**Files:**
- Create: `minids/model/mla.py`
- Test: `tests/test_mla.py`

**Interfaces:**
- Consumes: `ModelConfig`、`RMSNorm`、`apply_rope`。
- Produces: `class MLA(nn.Module)`，`__init__(self, cfg: ModelConfig)`，子模块与形状（`nn.Linear(in, out, bias=False)` 的 `weight` 为 `(out, in)`）：
  | 属性 | in | out |
  |---|---|---|
  | `wq` | `d_model` | `n_head * qk_head_dim` |
  | `wdkv` | `d_model` | `kv_lora_rank` |
  | `kv_norm` | `RMSNorm(kv_lora_rank, eps)` | |
  | `wukv` | `kv_lora_rank` | `n_head * (qk_nope_head_dim + v_head_dim)` |
  | `wkpe` | `d_model` | `qk_rope_head_dim` |
  | `wo` | `n_head * v_head_dim` | `d_model` |

  `forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor`，`x: (B, T, d_model)`，`cos/sin: (T, qk_rope_head_dim)`。头的排布必须是 head-major：`wq(x).view(B, T, n_head, qk_head_dim)`，`wukv(...)` 同理 `view(B, T, n_head, nope + v)`，`wo` 的输入是各头 `v` 按 head-major 拼接。`k_pe` 由 `wkpe(x)` 得到后 `unsqueeze(1)` 广播到全部头。SDPA 参数：`is_causal=True`，`scale=1/sqrt(qk_head_dim)`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_mla.py
def _run(cfg, x):
    cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.seq_len)
    return MLA(cfg)(x, cos[:x.shape[1]], sin[:x.shape[1]])

def test_output_shape(tiny_cfg):
    x = torch.randn(2, 16, tiny_cfg.d_model)
    assert _run(tiny_cfg, x).shape == (2, 16, tiny_cfg.d_model)

def test_causal(tiny_cfg):
    torch.manual_seed(0); m = MLA(tiny_cfg)
    cos, sin = precompute_rope(tiny_cfg.qk_rope_head_dim, tiny_cfg.seq_len)
    x = torch.randn(1, 12, tiny_cfg.d_model); x2 = x.clone(); x2[:, 8:] += 5.0
    y, y2 = m(x, cos[:12], sin[:12]), m(x2, cos[:12], sin[:12])
    assert torch.allclose(y[:, :8], y2[:, :8], atol=1e-5)
    assert not torch.allclose(y[:, 8:], y2[:, 8:], atol=1e-3)

def test_parameter_shapes(stage0_cfg):
    m = MLA(stage0_cfg)
    assert m.wq.weight.shape == (384, 384)
    assert m.wdkv.weight.shape == (128, 384)
    assert m.wukv.weight.shape == (672, 128)
    assert m.wkpe.weight.shape == (16, 384)
    assert m.wo.weight.shape == (384, 384)
    assert all(mod.bias is None for mod in [m.wq, m.wdkv, m.wukv, m.wkpe, m.wo])

def test_backward_flows(tiny_cfg):
    x = torch.randn(2, 8, tiny_cfg.d_model, requires_grad=True)
    _run(tiny_cfg, x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `minids/model/mla.py`**
- [ ] **Step 4: 运行，确认通过** → 4 passed
- [ ] **Step 5: Commit**：`git commit -m "feat: multi-head latent attention"`

---

### Task 6: SwiGLU

**Files:**
- Create: `minids/model/mlp.py`
- Test: `tests/test_mlp.py`

**Interfaces:**
- Produces: `class SwiGLU(nn.Module)`，`__init__(self, d_model: int, hidden: int)`，`w_gate: Linear(d_model, hidden)`，`w_up: Linear(d_model, hidden)`，`w_down: Linear(hidden, d_model)`，全部 `bias=False`；`forward(x) = w_down(F.silu(w_gate(x)) * w_up(x))`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_mlp.py
def test_shape():
    assert SwiGLU(64, 128)(torch.randn(2, 5, 64)).shape == (2, 5, 64)

def test_zero_down_proj_gives_zero_output():
    m = SwiGLU(64, 128); m.w_down.weight.data.zero_()
    assert torch.equal(m(torch.randn(2, 5, 64)), torch.zeros(2, 5, 64))

def test_no_bias():
    m = SwiGLU(8, 16)
    assert m.w_gate.bias is None and m.w_up.bias is None and m.w_down.bias is None
```

- [ ] **Step 2–5**：运行失败 → 实现 `minids/model/mlp.py` → 3 passed → `git commit -m "feat: SwiGLU feed-forward"`

---

### Task 7: Transformer 块

**Files:**
- Create: `minids/model/block.py`
- Test: `tests/test_block.py`

**Interfaces:**
- Consumes: `MLA`、`SwiGLU`、`RMSNorm`。
- Produces: `class Block(nn.Module)`，`__init__(self, cfg: ModelConfig)`，子模块 `attn_norm: RMSNorm(d_model)`、`attn: MLA`、`mlp_norm: RMSNorm(d_model)`、`mlp: SwiGLU(d_model, ffn_hidden)`；`forward(self, x, cos, sin) -> Tensor`：`x = x + attn(attn_norm(x), cos, sin)`；`x = x + mlp(mlp_norm(x))`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_block.py
def test_shape(tiny_cfg):
    cos, sin = precompute_rope(tiny_cfg.qk_rope_head_dim, 16)
    x = torch.randn(2, 16, tiny_cfg.d_model)
    assert Block(tiny_cfg)(x, cos, sin).shape == x.shape

def test_identity_when_residual_projections_are_zero(tiny_cfg):
    b = Block(tiny_cfg); b.attn.wo.weight.data.zero_(); b.mlp.w_down.weight.data.zero_()
    cos, sin = precompute_rope(tiny_cfg.qk_rope_head_dim, 16)
    x = torch.randn(2, 16, tiny_cfg.d_model)
    assert torch.equal(b(x, cos, sin), x)
```

- [ ] **Step 2–5**：运行失败 → 实现 `minids/model/block.py` → 2 passed → `git commit -m "feat: pre-norm transformer block"`

---

### Task 8: Transformer 整模型

**Files:**
- Create: `minids/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `Block`、`RMSNorm`、`precompute_rope`、`ModelConfig`。
- Produces:
  ```python
  class Transformer(nn.Module):
      cfg: ModelConfig
      tok_emb: nn.Embedding            # (vocab_size, d_model)
      blocks: nn.ModuleList[Block]
      final_norm: RMSNorm
      lm_head: nn.Linear               # (d_model -> vocab_size), bias=False
      # register_buffer("rope_cos"/"rope_sin", persistent=False)，形状 (seq_len, qk_rope_head_dim)
      def __init__(self, cfg: ModelConfig): ...
      def forward(self, idx: LongTensor, targets: LongTensor | None = None) -> tuple[Tensor, Tensor | None]
          # 返回 (logits (B,T,vocab), loss 或 None)；loss 为 F.cross_entropy(logits.view(-1,V), targets.view(-1))
      @torch.no_grad()
      def generate(self, idx: LongTensor, max_new_tokens: int, temperature: float = 1.0,
                   top_k: int | None = None) -> LongTensor
      def count_params(self, non_embedding: bool = True) -> int
  ```
  初始化（`_init_weights`，在 `__init__` 末尾用 `self.apply`，再对残差投影二次处理）：所有 `nn.Linear` 与 `nn.Embedding` 的 weight 用 `normal_(0, init_std)`；每个 block 的 `attn.wo.weight` 与 `mlp.w_down.weight` 用 `normal_(0, init_std / sqrt(2 * n_layer))`。
  错误处理：`T > seq_len` 抛 `ValueError(f"sequence length {T} exceeds seq_len {cfg.seq_len}")`；`idx.max() >= vocab_size` 抛 `ValueError`，信息含最大 id 与词表大小。
  `generate`：每步取最后 `seq_len` 个 token 做条件；`temperature == 0` 时取 argmax；`top_k` 时把非前 k 的 logits 设为 `-inf` 后再采样。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_transformer.py
def test_forward_shapes_and_loss(tiny_cfg):
    m = Transformer(tiny_cfg); idx = torch.randint(0, 256, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, 256) and loss.ndim == 0 and torch.isfinite(loss)

def test_param_counts_match_spec(stage0_cfg, stage2_cfg):
    assert Transformer(stage0_cfg).count_params() == 12_934_528
    assert Transformer(stage0_cfg).count_params(non_embedding=False) == 12_934_528 + 2 * 16384 * 384
    assert Transformer(stage2_cfg).count_params() == 103_247_872 + 768

def test_residual_proj_init_is_scaled(stage0_cfg):
    torch.manual_seed(0); m = Transformer(stage0_cfg)
    assert abs(m.blocks[0].mlp.w_up.weight.std().item() - 0.02) < 0.002
    assert abs(m.blocks[0].mlp.w_down.weight.std().item() - 0.02 / 4) < 0.001   # sqrt(2*8)=4

def test_too_long_sequence_raises(tiny_cfg):
    with pytest.raises(ValueError, match="exceeds"):
        Transformer(tiny_cfg)(torch.zeros(1, 65, dtype=torch.long))

def test_out_of_range_token_raises(tiny_cfg):
    with pytest.raises(ValueError):
        Transformer(tiny_cfg)(torch.tensor([[0, 256]]))

def test_generate_length_and_greedy(tiny_cfg):
    m = Transformer(tiny_cfg); idx = torch.randint(0, 256, (1, 5))
    out = m.generate(idx, 7, temperature=0.0)
    assert out.shape == (1, 12) and torch.equal(out[:, :5], idx)
    assert torch.equal(out, m.generate(idx, 7, temperature=0.0))   # 贪心可复现

def test_generate_crops_long_prompt(tiny_cfg):
    m = Transformer(tiny_cfg); idx = torch.randint(0, 256, (1, 100))
    assert m.generate(idx, 3, temperature=0.0).shape == (1, 103)
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `minids/model/transformer.py`**

`count_params(non_embedding=True)` = 所有参数元素数 − `tok_emb.weight.numel()` − `lm_head.weight.numel()`。

- [ ] **Step 4: 运行，确认通过** → 7 passed
- [ ] **Step 5: Commit**：`git commit -m "feat: Transformer model with init, generation and param counting"`

---

### Task 9: 加深算子

**Files:**
- Create: `minids/growth/depth.py`
- Test: `tests/test_growth_depth.py`

**Interfaces:**
- Consumes: `Transformer`、`ModelConfig.grown_depth()`。
- Produces: `def grow_depth(model: Transformer) -> Transformer`。新模型按 `model.cfg.grown_depth()` 构造，然后对每个 i：`new.blocks[2*i]` 载入 `old.blocks[i]` 的 state_dict；`new.blocks[2*i+1]` 同样载入后把 `attn.wo.weight` 与 `mlp.w_down.weight` 置零；`tok_emb`、`final_norm`、`lm_head` 直接拷贝。返回的新模型与旧模型在同一 device、同一 dtype，且处于 eval/train 状态与旧模型一致。不修改旧模型。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_growth_depth.py
def test_doubles_layers_and_params(tiny_cfg):
    old = Transformer(tiny_cfg); new = grow_depth(old)
    assert new.cfg.n_layer == 4 and len(new.blocks) == 4
    per_layer = sum(p.numel() for p in old.blocks[0].parameters())
    assert new.count_params() == old.count_params() + 2 * per_layer

def test_function_preserved_exactly(tiny_cfg):
    torch.manual_seed(0); old = Transformer(tiny_cfg).eval(); new = grow_depth(old).eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        assert torch.equal(old(idx)[0], new(idx)[0])

def test_new_blocks_have_zero_residual_projections(tiny_cfg):
    new = grow_depth(Transformer(tiny_cfg))
    for j in (1, 3):
        assert not new.blocks[j].attn.wo.weight.any()
        assert not new.blocks[j].mlp.w_down.weight.any()
    assert new.blocks[0].attn.wo.weight.any()

def test_old_model_untouched(tiny_cfg):
    old = Transformer(tiny_cfg); before = {k: v.clone() for k, v in old.state_dict().items()}
    grow_depth(old)
    assert all(torch.equal(before[k], v) for k, v in old.state_dict().items())
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `minids/growth/depth.py`**
- [ ] **Step 4: 运行，确认通过** → 4 passed
- [ ] **Step 5: Commit**：`git commit -m "feat: function-preserving depth growth (block expansion)"`

---

### Task 10: 加宽算子

**Files:**
- Create: `minids/growth/width.py`
- Test: `tests/test_growth_width.py`

**Interfaces:**
- Consumes: `Transformer`、`ModelConfig.grown_width()`。
- Produces: `def grow_width(model: Transformer, noise_std_rel: float = 1e-3, generator: torch.Generator | None = None) -> Transformer`。新模型按 `model.cfg.grown_width()` 构造，再按下表用旧权重 `W`（nn.Linear 的 weight 形状为 `(out, in)`）填充；`noise_std_rel == 0` 时不加噪。三个基本操作：
  - `both(W) = 0.5 * cat([cat([W, W], dim=1), cat([W, W], dim=1)], dim=0)`
  - `in_only(W) = 0.5 * cat([W, W], dim=1)`
  - `vec(g) = cat([g, g])`

  | 参数 | 规则 | 加噪位置 |
  |---|---|---|
  | `tok_emb.weight` (V, d) | `cat([E, E], dim=1)`（不乘 0.5） | 后半列 `[:, d:]` |
  | `blocks[i].attn_norm.weight`、`mlp_norm.weight` | `vec` | 不加 |
  | `attn.wq.weight` | `both` | 后半行 |
  | `attn.wdkv.weight` | `both` | 后半行 |
  | `attn.kv_norm.weight` | `vec` | 不加 |
  | `attn.wukv.weight` | `both` | 后半行 |
  | `attn.wkpe.weight` | `in_only` | 不加 |
  | `attn.wo.weight` | `both` | 后半行 |
  | `mlp.w_gate.weight`、`w_up.weight`、`w_down.weight` | `both` | 后半行 |
  | `final_norm.weight` | `vec` | 不加 |
  | `lm_head.weight` (V, d) | `in_only` | 不加 |

  噪声：对标记位置加 `normal(0, noise_std_rel * W.std())`，用传入的 `generator`。实现时先用 `noise_std_rel=0` 的规则构造全部张量，再统一加噪，便于 Task 11 分别验证。不修改旧模型。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_growth_width.py
def test_shapes_doubled(tiny_cfg):
    new = grow_width(Transformer(tiny_cfg), noise_std_rel=0.0)
    c = new.cfg
    assert (c.d_model, c.n_head, c.kv_lora_rank, c.ffn_hidden) == (128, 4, 64, 256)
    assert new.blocks[0].attn.wkpe.weight.shape == (8, 128)
    assert new.lm_head.weight.shape == (256, 128)

def test_function_preserved_without_noise(tiny_cfg):
    torch.manual_seed(0); old = Transformer(tiny_cfg).eval()
    new = grow_width(old, noise_std_rel=0.0).eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        a, b = old(idx)[0], new(idx)[0]
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)

def test_noise_keeps_relative_error_within_tolerance(tiny_cfg):
    torch.manual_seed(0); old = Transformer(tiny_cfg).eval()
    g = torch.Generator().manual_seed(1)
    new = grow_width(old, noise_std_rel=1e-3, generator=g).eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        a, b = old(idx)[0], new(idx)[0]
    assert ((a - b).abs().max() / a.abs().max()).item() <= 1e-2

def test_noise_breaks_symmetry(tiny_cfg):
    g = torch.Generator().manual_seed(1)
    new = grow_width(Transformer(tiny_cfg), noise_std_rel=1e-3, generator=g)
    W = new.blocks[0].mlp.w_gate.weight; h = W.shape[0] // 2
    assert not torch.equal(W[:h], W[h:])
    E = new.tok_emb.weight; d = E.shape[1] // 2
    assert not torch.equal(E[:, :d], E[:, d:])

def test_old_model_untouched(tiny_cfg):
    old = Transformer(tiny_cfg); before = {k: v.clone() for k, v in old.state_dict().items()}
    grow_width(old)
    assert all(torch.equal(before[k], v) for k, v in old.state_dict().items())
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `minids/growth/width.py`**
- [ ] **Step 4: 运行，确认通过** → 5 passed
- [ ] **Step 5: Commit**：`git commit -m "feat: function-preserving width growth (HyperCloning)"`

---

### Task 11: 增长验证

**Files:**
- Create: `minids/growth/verify.py`
- Test: `tests/test_growth_verify.py`

**Interfaces:**
- Consumes: `Transformer`。
- Produces:
  ```python
  class GrowthVerificationError(RuntimeError): ...

  @dataclass(frozen=True)
  class GrowthReport:
      params_before: int
      params_after: int
      max_abs_diff: float
      max_rel_err: float          # max|Δlogit| / max|logit_before|
      exact: bool                 # torch.equal 为 True
      def to_dict(self) -> dict: ...

  def verify_growth(old: Transformer, new: Transformer, tokens: LongTensor,
                    rel_tol: float = 1e-2) -> GrowthReport
  ```
  两个模型都在 `eval()` 与 `torch.no_grad()` 下、以 fp32 权重拷贝（`copy.deepcopy(m).float()`，不改原模型）前向 `tokens`；`max_rel_err > rel_tol` 时抛 `GrowthVerificationError`，错误信息含两个数值。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_growth_verify.py
def test_depth_growth_reports_exact(tiny_cfg):
    old = Transformer(tiny_cfg); tokens = torch.randint(0, 256, (2, 32))
    r = verify_growth(old, grow_depth(old), tokens)
    assert r.exact and r.max_abs_diff == 0.0 and r.params_after > r.params_before

def test_width_growth_with_noise_passes(tiny_cfg):
    old = Transformer(tiny_cfg); tokens = torch.randint(0, 256, (2, 32))
    r = verify_growth(old, grow_width(old, generator=torch.Generator().manual_seed(1)), tokens)
    assert r.max_rel_err <= 1e-2

def test_broken_growth_raises(tiny_cfg):
    old = Transformer(tiny_cfg); bad = grow_depth(old)
    bad.blocks[1].attn.wo.weight.data.normal_()          # 破坏恒等性
    with pytest.raises(GrowthVerificationError):
        verify_growth(old, bad, torch.randint(0, 256, (2, 32)))

def test_report_to_dict_keys(tiny_cfg):
    old = Transformer(tiny_cfg)
    d = verify_growth(old, grow_depth(old), torch.randint(0, 256, (1, 8))).to_dict()
    assert set(d) == {"params_before", "params_after", "max_abs_diff", "max_rel_err", "exact"}
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `minids/growth/verify.py`**
- [ ] **Step 4: 运行，确认通过** → 4 passed；再跑全量 `.venv/Scripts/python -m pytest -q` → 全部通过
- [ ] **Step 5: Commit**：`git commit -m "feat: growth verification report"`

---

### Task 12: 吞吐基准

**Files:**
- Create: `scripts/bench_throughput.py`、`docs/benchmarks/throughput.md`

**Interfaces:**
- Consumes: `Transformer`、`ModelConfig`、三级配置（与 conftest 相同的数值，脚本内按 `--stage 0|1|2` 构造）。
- Produces: 命令行 `python scripts/bench_throughput.py --stage 2 --micro-batch 16 [--compile] [--steps 30]`：在 GPU 上用随机 token、bf16 autocast、AdamW 做 `steps` 步完整训练步（前向、反向、优化器），丢弃前 5 步，输出 `tokens/s`、`peak_mem_gb`、`ms/step`，并以 JSON 行追加到 `docs/benchmarks/throughput.jsonl`（git 忽略）。`--compile` 时尝试 `torch.compile(model)`，失败则打印警告并以普通模式继续，输出中标明 `compiled: false`。

- [ ] **Step 1: 实现脚本**

- [ ] **Step 2: 运行基准矩阵**

对每级跑不开 compile 的基准，微批次从 8 开始翻倍直到显存溢出前的最大值；级 2 再跑一次 `--compile`。记录每次的命令与输出。

Run 示例：`.venv/Scripts/python scripts/bench_throughput.py --stage 0 --micro-batch 32`
Expected: 打印 `tokens/s` 为正数，`peak_mem_gb < 16`。

- [ ] **Step 3: 写 `docs/benchmarks/throughput.md`**

表格：级、微批次、是否 compile、tokens/s、峰值显存、每步毫秒。末尾一段结论：按实测 tokens/s 换算 spec 8.1 各级 token 预算的小时数；若级 2 的 2.0B token 超过 14 小时，写明建议把级 2 与对照组预算降到 1.5B（spec 第 16 节的应对）。同时给出计划三要用的推荐微批次（每级显存占用不超过 12 GB 的最大值）。

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_throughput.py docs/benchmarks/throughput.md
git commit -m "perf: training throughput benchmark for ladder stages"
```

---

## 完成标准

- `.venv/Scripts/python -m pytest -q` 全部通过（含 GPU 环境测试）。
- `grow_depth` 与 `grow_width(noise_std_rel=0)` 对迷你配置保函数，精度满足 spec 7.3。
- `docs/benchmarks/throughput.md` 给出三级实测吞吐、推荐微批次，以及对 spec 8.1 时间预算的修正结论。
- 计划二（数据管线）可以在此基础上开始。
