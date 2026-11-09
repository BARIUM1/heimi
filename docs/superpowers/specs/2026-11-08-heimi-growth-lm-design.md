# 黑米（Heimi）设计文档：可渐进增长的迷你 DeepSeek 风格语言模型

日期：2026-11-08
状态：待用户审阅
名称：黑米 / Heimi，以用户家的猫命名

## 1. 目标与范围

### 1.1 目标

在一台 Windows 11 笔记本（RTX 5080 笔记本版，16 GB 显存，32 GB 内存）上，用自己写的 PyTorch 代码：

1. 从零训练一个采用 DeepSeek 核心设计（MLA 注意力、RMSNorm、SwiGLU、pre-norm）的小型稠密语言模型。
2. 让模型分级"长大"：每次增长在数学上保持函数不变，训练继续进行，前一级学到的东西不丢。
3. 阶梯长完后，把模型训练成一个本地离线的代码与命令行单行补全模型（支持 Fill-in-the-Middle），并提供最简使用入口。
4. 提供一个属于这个模型的本地网页窗口，直观展示训练过程、每次增长的效果，以及不同级别模型对同一输入的表现。

### 1.2 非目标（本期不做）

- MoE、多 token 预测（MTP）、Q 的低秩压缩、KV 缓存优化。
- 中文能力。数据只用英文文本和代码。
- 开放式聊天或问答能力。
- 编辑器插件。本期只交付命令行补全工具。
- 超过约 105M 非嵌入参数的模型。
- 实验跟踪平台（wandb 等）。日志为本地文件。

### 1.3 成功标准

1. 每次增长后，在一批真实数据上，增长前后模型的 logits 在不加对称破缺噪声时逐位相等；加噪声后最大相对误差不超过 1e-2。增长后的训练损失曲线在增长点不出现跳升。
2. 增长到 105M 的模型，达到从零训练的 105M 对照模型最终验证损失所用的累计 token 数，少于对照模型的 2.0B。具体省多少不预设，由实验给出。
3. 任务阶段结束后，模型在留出代码集上的单行 FIM 精确匹配率可测量且明显高于阶梯结束时的水平；对固定的英文提示能生成语法通顺的句子。
4. 窗口能在训练进行中实时展示四个区域的内容，且训练不依赖窗口。
5. 全部单元测试和烟雾测试通过。

### 1.4 约束

- 总 GPU 时间 1 到 2 天（含对照组），主要靠夜间运行。
- 只用英文数据：通用文本用 FineWeb-Edu，代码用公开且许可宽松的数据集。
- 不依赖大框架：不用 transformers、不用 Megatron 等，模型与训练循环手写。
- Windows 原生运行，不装 WSL。

## 2. 总体架构

```
数据流：
  FineWeb-Edu parquet + 代码 parquet
      → tokenizer（自训 16k BPE）
      → uint16 token 分片（train/val，web/code 分开）
      → loader（按配比随机取窗口；任务阶段对代码做 FIM 改写）
      → train.py（按 ladder.yaml 逐级训练，级间调用 growth 算子）
      → runs/<run>/ 下的检查点与 jsonl 日志
      → dashboard.py 读取日志并调用检查点生成
      → complete.py 用最终检查点做命令行补全
```

模块之间只通过文件（配置、分片、检查点、日志）和少数明确的函数接口通信。训练脚本不知道窗口的存在；窗口只读文件，不写训练状态。

## 3. 环境

- 位置：`F:\mini-deepseek`，独立 git 仓库。
- Python 3.12，使用已安装的 `uv` 创建虚拟环境 `.venv`。
- PyTorch：从 PyTorch 官方 cu128 或 cu130 索引安装 Windows CUDA 版本（当前机器上的 `torch 2.13.0+cpu` 不可用于训练）。安装后以 `torch.cuda.is_available()` 和 `get_device_capability()` 返回 Blackwell（12.x）为准。
- 依赖最小集：`torch`、`numpy`、`tokenizers`、`huggingface_hub`、`pyarrow`、`pyyaml`、`fastapi`、`uvicorn`、`matplotlib`、`pytest`。
- 注意力使用 `torch.nn.functional.scaled_dot_product_attention`，不安装 flash-attn。
- `torch.compile` 为可选加速：启动时尝试，失败则记录警告并回退普通模式。正确性与可复现性不依赖它。
- GPU 状态（温度、显存）通过调用 `nvidia-smi` 读取。

## 4. 项目结构

```
mini-deepseek/
  configs/
    ladder.yaml          # 阶梯：每级的模型结构、token 预算、学习率、增长类型
    baseline105m.yaml    # 对照组：105M 从零训练
    tiny_smoke.yaml      # 烟雾测试用迷你配置
  minids/
    config.py            # ModelConfig / TrainConfig 数据类与 yaml 读写
    model/
      norm.py            # RMSNorm
      rope.py            # 旋转位置编码（只作用于 rope 维）
      mla.py             # MLA 注意力
      mlp.py             # SwiGLU 前馈
      block.py           # 一个 Transformer 块（pre-norm）
      transformer.py     # 整模型：嵌入、块堆叠、最终 norm、输出头、生成
    growth/
      depth.py           # 加深算子（块扩展）
      width.py           # 加宽算子（HyperCloning）
      verify.py          # 增长前后一致性检查
    data/
      download.py        # 下载 FineWeb-Edu 与代码数据分片，断点续传与校验
      tokenizer.py       # 训练与加载 16k 字节级 BPE
      prepare.py         # parquet → uint16 token 分片
      loader.py          # 按配比取 batch；FIM 改写
      fim.py             # FIM 格式转换（独立出来便于测试）
    train/
      loop.py            # 训练循环、检查点、恢复、日志写入
      schedule.py        # 学习率调度
      keepawake.py       # Windows 防休眠
    eval/
      val_loss.py        # 分 web/code 的验证损失
      hellaswag.py       # HellaSwag 准确率
      fim_em.py          # 单行 FIM 精确匹配率
    logging_io.py        # jsonl 读写（训练端与窗口端共用的格式定义）
  dashboard/
    server.py            # FastAPI 后端
    worker.py            # 生成工作进程
    static/index.html    # 单页前端（Chart.js）
  train.py               # 入口：按配置逐级训练
  evaluate.py            # 入口：对检查点跑评测
  sample.py              # 入口：文本生成
  complete.py            # 入口：命令行 FIM 补全
  dashboard.py           # 入口：启动窗口
  tests/
  runs/                  # 检查点与日志（git 忽略）
  data/                  # 原始 parquet、分词器、token 分片（git 忽略）
  docs/
```

每个文件只管一件事。`model/` 下每个文件都能单独导入并测试；`growth/` 只依赖 `model/` 的公开结构（`ModelConfig` 和模块的命名参数），不依赖训练代码。

## 5. 模型架构

### 5.1 骨架

DeepSeek-V2-Lite 去掉 MoE 的稠密版本：

- 词嵌入（不与输出头共享权重）
- N 个 pre-norm 块：`x = x + Attn(RMSNorm(x))`，`x = x + MLP(RMSNorm(x))`
- 最终 RMSNorm
- 输出头（线性层，无偏置）

全模型无偏置项。

### 5.2 MLA 注意力

记模型维度 `d`，头数 `h`，每头维度：`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`，潜向量维度 `kv_lora_rank`。

- Q：`W_q: d → h × (qk_nope + qk_rope)`，拆成 `q_nope` 和 `q_pe`，`q_pe` 加 RoPE。
- KV 压缩：`W_dkv: d → kv_lora_rank`，过 RMSNorm，得潜向量 `c_kv`。
- KV 升维：`W_ukv: kv_lora_rank → h × (qk_nope + v)`，拆成 `k_nope` 和 `v`。
- 位置键：`W_kpe: d → qk_rope`，所有头共用，加 RoPE，广播到每个头。
- 拼接：`q = [q_nope | q_pe]`，`k = [k_nope | k_pe]`，交给 SDPA 做因果注意力，缩放系数为 `1/sqrt(qk_nope + qk_rope)`。
- 输出：`W_o: h × v → d`。

不做 Q 的低秩压缩，不做推理期的权重吸收（实验不需要）。

### 5.3 各级结构参数

| 参数 | 级 0 | 级 1 | 级 2 与任务阶段 |
|---|---|---|---|
| n_layer | 8 | 16 | 16 |
| d_model | 384 | 384 | 768 |
| n_head | 6 | 6 | 12 |
| qk_nope_head_dim | 48 | 48 | 48 |
| qk_rope_head_dim | 16 | 16 | 16 |
| v_head_dim | 64 | 64 | 64 |
| kv_lora_rank | 128 | 128 | 256 |
| ffn_hidden | 1024 | 1024 | 2048 |
| vocab_size | 16384 | 16384 | 16384 |
| seq_len | 1024 | 1024 | 1024 |
| 非嵌入参数（约） | 13M | 26M | 105M |
| 嵌入 + 输出头参数（约） | 12.6M | 12.6M | 25.2M |

对照组与级 2 结构完全相同。

### 5.4 初始化与数值

- 线性层与嵌入：正态分布，标准差 0.02。
- 两个残差输出投影（`W_o`、MLP 下投影）：标准差再乘 `1/sqrt(2 × n_layer)`。
- RMSNorm eps = 1e-6，缩放参数初始为 1。
- 训练用 bf16 自动混合精度，主权重 fp32。
- RoPE base = 10000。

### 5.5 生成

朴素实现：每生成一个 token 全量前向一次。支持温度与 top-k。105M 模型在 CPU 上生成 64 个 token 约数秒，满足窗口与命令行工具的需要。

## 6. 分词器与数据

### 6.1 分词器

- 字节级 BPE，用 `tokenizers` 库训练，总词表 16384。
- 训练语料：FineWeb-Edu 与代码数据按约 7:3 抽样，合计约 2 GB 文本，保证代码缩进、常见符号序列有高效编码。
- 特殊 token（占词表前 16 个位置，未用的保留）：`<|endoftext|>`、`<|fim_prefix|>`、`<|fim_middle|>`、`<|fim_suffix|>`、`<|pad|>`。
- 词表一经确定不再更改。所有阶段共用。

### 6.2 数据来源

- 通用文本：`HuggingFaceFW/fineweb-edu` 的 `sample/10BT` 目录，14 个分片各 2.15 GB，取前 8 个（约 17 GB，约 5.5B 个 GPT-2 等价 token；在 16k 词表下 token 数会多出约 10% 到 20%）。
- 代码：一个公开、许可宽松、无需申请权限的代码数据集，取 Python、Shell、JavaScript/TypeScript 三类，目标约 1B token。实施时按可下载性在候选中选定（如 `codeparrot/github-code-clean`、The Stack 系列的开放子集）。若首选需要权限，换下一个候选，不阻塞。
- 验证集：从 web 与 code 各留出约 2000 万 token，不参与训练与分词器训练。

### 6.3 预处理

- `download.py`：按 Hugging Face 提供的文件大小与校验值核对，支持断点续传。
- `prepare.py`：逐文档分词，文档之间插入 `<|endoftext|>`，写成 uint16 二进制分片，每片 1 亿 token，文件名 `data/tokens/{web,code}/{train,val}_{NNNN}.bin`，附带一个 `index.json` 记录每片 token 数。损坏或空的 parquet 行跳过并计数写入日志。

### 6.4 加载与配比

- `loader.py` 从指定分片集合中随机取长度 `seq_len + 1` 的连续窗口，构成 `(input, target)`。
- 两个数据源按配比独立采样后拼成一个 batch。
- 配比由配置给出：阶梯阶段 web 0.85 / code 0.15；任务阶段 web 0.20 / code 0.80。
- 随机数由 `seed + step` 决定，恢复训练后取到的数据序列与未中断时一致。

### 6.5 FIM 改写

仅在任务阶段对代码窗口启用，比例 50%。采用 DeepSeek-Coder 的文档级 PSM 格式：

```
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>
```

切分点在 token 序列上均匀随机选两处。`fim.py` 提供正向改写和逆向还原两个函数，用于测试可逆性。

## 7. 增长算子

### 7.1 加深：块扩展

输入 N 层模型，输出 2N 层模型。对每个原有块 i，在其后插入一个新块：

- 新块所有权重从块 i 深拷贝。
- 新块的注意力输出投影 `W_o` 和 MLP 下投影全部置零。
- 新块的两个 RMSNorm 缩放参数照常拷贝。

置零后新块的残差分支输出恒为零，整体为恒等映射，模型 logits 与增长前逐位相等。

### 7.2 加宽：HyperCloning

输入维度 `d` 的模型，输出维度 `2d` 的模型。头数、`ffn_hidden`、`kv_lora_rank` 同步翻倍；`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim` 不变。

记原线性层权重 `W`（形状 out × in）。约定增长后残差流中每个向量是原向量的两份拼接 `[x; x]`。

- 输入输出都翻倍的层（`W_q`、`W_dkv`、`W_ukv`、`W_o`、MLP 三个投影）：`W' = 1/2 · [[W, W], [W, W]]`。输入 `[x; x]` 时输出为 `[Wx; Wx]`。
- 只有输入翻倍的层（`W_kpe`，输出维仍是 `qk_rope`）：`W' = 1/2 · [W, W]`。输出与原来相同。
- 词嵌入（vocab × d）：按列复制 `E' = [E, E]`，使残差流起点就是 `[x; x]`。
- 输出头（vocab × d）：`H' = 1/2 · [H, H]`，logits 与原来相同。
- RMSNorm 缩放参数：复制一份拼接。`rms([x; x]) = rms(x)`，归一化结果为 `[x̂; x̂]`。
- 注意力：头数翻倍，新头与原头两两相同，每头内的 `q·k` 不变，softmax 不变。`W_ukv` 的输出维按头拼接，复制规则同上。
- RoPE：作用于每头内的 `qk_rope` 维，不受影响。

对称破缺：复制完成并通过精确验证后，对所有输出维翻倍的矩阵，给对应第二份输出的那一半行加入均值 0、标准差为该矩阵元素标准差 1e-3 倍的高斯噪声；词嵌入按列复制的第二份同样加噪。只复制输入维的 `W_kpe` 和输出头不加噪，RMSNorm 缩放参数不加噪。这样残差流的两半从第一层起就不再完全相同，训练中会各自分化。

### 7.3 验证

`verify.py` 接收增长前后两个模型和一批真实 token：

1. 噪声关闭时，断言两个模型 fp32 前向的 logits `torch.equal` 或 `allclose(atol=1e-5)`（bf16 下允许后者）。
2. 噪声开启后，断言最大相对误差 `max|Δlogit| / max|logit| ≤ 1e-2`。
3. 结果写入 `events.jsonl`，类型 `growth`，包含前后参数量、验证误差、耗时。

验证失败则中止训练，不进入下一级。

### 7.4 增长后的训练

- 优化器状态重置（AdamW 的一阶、二阶矩不跨级迁移）。
- 学习率先用该级总步数 5% 的线性预热到该级峰值，再余弦衰减。
- 数据读取位置继续累计，不回头。

## 8. 阶梯与训练预算

### 8.1 阶梯

| 级 | 动作 | 结构 | 非嵌入参数 | token 预算 | 峰值学习率 | 预计耗时 |
|---|---|---|---|---|---|---|
| 0 | 从零初始化 | 8 层，d=384 | 13M | 0.3B | 2e-3 | 约 1 小时 |
| 1 | 加深 8 → 16 | 16 层，d=384 | 26M | 0.5B | 1.5e-3 | 约 1.5 小时 |
| 2 | 加宽 384 → 768 | 16 层，d=768 | 105M | 2.0B | 6e-4 | 约 10 小时 |
| 任务 | 结构不变，数据切换，启用 FIM | 同级 2 | 105M | 0.7B | 2e-4 | 约 4 小时 |
| 对照 | 从零初始化 | 同级 2 | 105M | 2.0B | 6e-4 | 约 10 小时 |

耗时按约 50k token/s（105M 模型）估算，实施第一步会实测吞吐并修正。对照组可最后跑。

### 8.2 通用训练超参数

- 优化器：AdamW，β1 0.9，β2 0.95，权重衰减 0.1（嵌入与 norm 参数不衰减），梯度裁剪 1.0。
- 每步约 524,288 token（512 条 × 1024），微批次大小由配置给出，不足部分用梯度累积。
- 学习率：级 0 预热占本级 2%，其余级预热占 5%；余弦衰减到峰值的 10%。
- 随机种子固定，写入配置与日志。

### 8.3 配置文件

`configs/ladder.yaml` 一个文件描述整条阶梯：全局项（种子、seq_len、batch token 数、数据配比、评测间隔、检查点间隔、样本提示）和 `stages` 列表，每项含 `name`、`growth`（`none` / `depth` / `width`）、目标结构参数、`tokens`、`lr`、`data_mix`、`fim_rate`。对照组用 `baseline105m.yaml`，只有一个 `growth: none` 的阶段。

## 9. 训练循环、日志与检查点

### 9.1 运行方式

- `python train.py --config configs/ladder.yaml --run heimi`：按顺序执行全部阶段。存在检查点时自动续训，包括从上一级末尾自动执行增长进入下一级。
- `--stage <name>` 只跑指定阶段（要求前一阶段的最终检查点存在）。
- `python train.py --config configs/baseline105m.yaml --run baseline105m`。

### 9.2 日志格式（`runs/<run>/`）

所有 jsonl 每行一个 JSON 对象，字段定义集中在 `logging_io.py`，训练端写、窗口端读。

- `metrics.jsonl`（每 10 步一行）：`stage`、`step`（本级内）、`global_step`、`tokens_stage`、`tokens_total`（跨级累计，窗口的横轴）、`loss`、`lr`、`grad_norm`、`tok_per_s`、`mem_gb`、`gpu_temp_c`、`time`。
- `eval.jsonl`（每 500 步及每级结束）：`stage`、`global_step`、`tokens_total`、`val_loss_web`、`val_loss_code`、`hellaswag_acc`（仅每级结束与任务阶段结束时算）、`fim_em`（任务阶段每次评测都算；其他阶段每级结束算一次作为基线）。
- `events.jsonl`：`type` 取 `stage_start`、`stage_end`、`growth`、`checkpoint`、`nan_abort`、`oom`、`compile_fallback`、`resume`，带时间与相关字段。
- `samples.jsonl`（每 500 步）：对配置中的固定提示生成 64 个 token，字段 `stage`、`global_step`、`tokens_total`、`prompt_id`、`prompt`、`completion`。固定提示 6 条：3 条英文开头、2 条代码前缀续写、1 条 FIM 填空。

### 9.3 检查点

- 路径 `runs/<run>/ckpt/<stage>_step<N>.pt` 与 `runs/<run>/ckpt/<stage>_final.pt`；每级保留最近 2 个中间检查点与最终检查点。
- 内容：模型权重、优化器状态、调度器状态、`global_step`、`tokens_total`、数据随机数状态、配置快照、代码 git 提交号。
- 恢复：读最近检查点，校验配置一致，继续。`events.jsonl` 记一条 `resume`。

### 9.4 防休眠

`keepawake.py` 在训练进程内调用 Windows `SetThreadExecutionState`，训练期间阻止系统待机（不阻止关屏），进程退出自动恢复。

## 10. 评测

- 验证损失：web 与 code 各取固定的 200 个窗口（约 20 万 token），分别报告。
- HellaSwag：标准的 10,042 条验证集，按续写的归一化对数似然选项，报告准确率。只在每级结束和任务阶段结束跑（约几分钟）。
- 单行 FIM 精确匹配率：从代码验证集抽 1000 个样本，每个随机挖掉一整行作为 middle，贪心生成到换行，报告与原行去首尾空白后完全相同的比例。
- `evaluate.py --ckpt <path>` 可对任意检查点单独跑全部评测，结果追加到该 run 的 `eval.jsonl` 并打印。

## 11. 任务阶段与交付物

- 数据配比 web 0.20 / code 0.80，FIM 比例 50%，峰值学习率 2e-4，0.7B token。
- 交付：
  1. `runs/heimi/ckpt/task_final.pt`。
  2. `python export.py --ckpt ... --out models/heimi-105m/`：导出 `model.safetensors`、`config.json`、分词器文件，供不带训练代码的推理使用。
  3. `complete.py`：`--prefix-file`、`--suffix-file`（可选）、`--max-tokens`，输出中间内容到标准输出；也接受标准输入。默认 CPU，`--device cuda` 可切 GPU。

## 12. 窗口（Dashboard）

### 12.1 形态

`python dashboard.py --runs runs/` 启动 FastAPI 服务于 `http://127.0.0.1:8787` 并自动打开浏览器。单页应用，全中文，每 5 秒拉取一次最新数据。页面标题与顶栏显示模型名字。

### 12.2 后端接口

- `GET /api/runs`：可用 run 列表及其阶段、最新步数、检查点列表。
- `GET /api/runs/{run}/metrics?since=<global_step>`：增量读取 `metrics.jsonl`。
- `GET /api/runs/{run}/eval`、`/events`、`/samples`：对应 jsonl 全量。
- `GET /api/status`：训练是否在跑（看 `metrics.jsonl` 最后一行时间）、当前 GPU 温度与显存（`nvidia-smi`）、预计剩余时间（按最近 100 步吞吐与本级剩余 token 算）。
- `POST /api/generate`：`{run, checkpoint, prefix, suffix?, max_new_tokens, temperature, device}`，返回生成文本与耗时。

### 12.3 生成工作进程

`worker.py` 为独立子进程，按需加载检查点，最多缓存 2 个模型；默认 CPU，请求可指定 `cuda`。超时 60 秒返回错误。训练进程与之无共享状态。

### 12.4 四个区域

1. **成长总览**：主图横轴 `tokens_total`、纵轴验证损失（web、code 两条线），每条 `growth` 事件画竖线并标注"13M → 26M 加深"；若 `baseline105m` 存在，叠加虚线。副图：非嵌入参数量随 `tokens_total` 的阶梯图。
2. **当前状态**：run、阶段、步数、损失、学习率、tok/s、显存、GPU 温度、预计剩余时间、最近一次评测结果。
3. **成长样本墙**：按 `prompt_id` 分列，按时间从上到下排列每次采样的续写，默认显示最近 8 次，可展开全部。
4. **试玩台**：输入前缀与可选后缀，选择 run 与检查点（下拉列出各级 final），生成并显示；可勾选"并排比较"同时选第二个检查点。

### 12.5 前端实现

一个 `index.html`，内联 CSS 与 JS，Chart.js 从本地 `static/` 加载（不依赖外网）。不用框架、不用构建步骤。

## 13. 测试策略

### 13.1 单元测试（pytest，`tests/`）

- `test_norm.py`：RMSNorm 输出均方根为 1；缩放参数生效。
- `test_rope.py`：相对位置性质，`q_m · k_n` 只依赖 `m - n`。
- `test_mla.py`：输出形状；因果性：改动位置 t 之后的输入不改变位置 ≤ t 的输出；`kv_lora_rank` 等于 `h × (qk_nope + v)` 且升维矩阵为单位阵时退化为普通注意力（可选）。
- `test_block.py`、`test_transformer.py`：形状、前向可反传、生成长度正确。
- `test_growth_depth.py`：加深后 logits 与加深前精确相等；层数、参数量符合预期。
- `test_growth_width.py`：加宽后不加噪 logits 精确相等（fp32）；加噪后相对误差 ≤ 1e-2；各参数形状符合预期。
- `test_checkpoint.py`：保存再加载，权重、优化器、步数、随机数状态一致；损坏文件报清晰错误。
- `test_loader.py`：同一种子同一步取到同一 batch；配比在大样本下接近配置值；窗口不越界。
- `test_fim.py`：改写后还原等于原文；比例在大样本下接近配置值；特殊 token 位置正确。
- `test_tokenizer.py`：任意字节串编码再解码还原；特殊 token 不被拆分。
- `test_schedule.py`：预热、峰值、余弦终点数值正确。
- `test_logging_io.py`：写入再读取字段一致；`since` 增量读取正确。

### 13.2 烟雾测试

`configs/tiny_smoke.yaml`：2 层、d=64、2 头、词表沿用、seq_len 128，用一个几百 KB 的合成 token 文件；跑 30 步 → 加深 → 20 步 → 加宽 → 20 步，检查损失下降、增长验证通过、检查点可恢复、日志四个文件都有内容。目标在 GPU 上 1 分钟内完成，CPU 上 5 分钟内完成。每次改代码都跑。

### 13.3 窗口测试

- 后端接口用 FastAPI 的 TestClient 对一份样例 `runs/` 快照做读取测试。
- 生成接口用烟雾测试产出的迷你检查点测一次往返。
- 前端不做自动化测试，用样例数据人工看一遍。

## 14. 容错与运维

- NaN 或 Inf 损失：立即停止，写 `nan_abort` 事件，保留最近检查点；重启后从该检查点继续，并在日志中提示可能需要调低学习率。
- 显存溢出：捕获后写 `oom` 事件并退出，提示把微批次减半、用梯度累积补回。
- `torch.compile` 失败：写 `compile_fallback` 事件，普通模式继续。
- 下载：断点续传，校验失败删除重下，最多 3 次。
- 数据损坏：单行解析失败跳过并计数，计数写入 `prepare` 的日志。
- 笔记本散热：`gpu_temp_c` 进指标，窗口显示；不做自动降频，由用户判断。
- 磁盘：数据约 20 GB 原始加约 12 GB token 分片，检查点每个约 1.3 GB（105M 含优化器状态），按保留策略最多约 10 GB。F 盘当前 282 GB 可用。

## 15. 时间与资源预算

- 实施（写代码与测试）：按实施计划分步进行，与训练交错。
- GPU 时间：阶梯约 12.5 小时，任务阶段约 4 小时，对照约 10 小时，合计约 27 小时，分两到三个夜晚。
- 下载：约 20 GB，视网速 1 到 3 小时。
- 分词器训练与数据预处理：约 1 到 2 小时 CPU。

## 16. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 吞吐低于估计 | 时间翻倍 | 实施第一步实测；必要时把级 2 预算降到 1.5B，对照同步降 |
| `torch.compile` 在 Windows 不可用 | 慢 20% 到 40% | 设计上已可选；优先保证普通模式 |
| 首选代码数据集需要权限 | 下载受阻 | 候选列表顺延，不阻塞 |
| 笔记本持续满载降频 | 吞吐下降、时间拉长 | 温度进指标；可在配置中限制微批次降低功耗 |
| 加宽后对称破缺噪声过大或过小 | 要么偏离太多，要么两半长期相同 | 1e-3 为起点，验证阈值把关；烟雾测试观察两半权重是否分化 |
| HellaSwag 在 13M、26M 接近随机 | 指标无区分度 | 预期之内，主要看验证损失；HellaSwag 用于级 2 与任务阶段 |

## 17. 未来扩展（不在本期）

- 第四级增长：把 MLP 升级为 DeepSeekMoE（稀疏升级），复现 DeepSeek 真实架构。
- 中文持续预训练，作为另一种"增长"。
- 编辑器插件，把 `complete.py` 接到 VS Code。
- KV 缓存与权重吸收，让生成变快。
- 多 token 预测。
