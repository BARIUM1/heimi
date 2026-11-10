# 黑米官方评测设计说明

日期：2026-11-09。本文是主设计文档 `2026-11-08-heimi-growth-lm-design.md` 的补充，只管一件事：用公开的标准评测、在公开的标准评测框架下给黑米的每个阶段打分，并且有一条同机同法测出来的参照线。主文档第 10 节的自有评测（验证损失、自写的 HellaSwag、单行 FIM 精确匹配）保持不变，继续在训练中途使用。

## 1. 为什么要做

自有评测回答"这一级比上一级好了多少"，但回答不了"黑米放在别的模型旁边是什么水平"。原因有三个：验证集是自己切的；HellaSwag 是自己写的打分，预处理和归一化与公开数字用的不一样；FIM 样本是从自己的语料里挖的。要和外面的数字放在一起，评测数据、提示构造、打分公式都得是别人定的。

## 2. 评什么

| 评测 | 数据 | 怎么跑 | 报告的数 | 性质 |
| --- | --- | --- | --- | --- |
| BLiMP | `nyu-mll/blimp`，67 个子任务各 1,000 对句子 | lm-evaluation-harness 的 `blimp` 组 | 准确率（随机水平 0.5） | 官方 |
| HellaSwag | `Rowan/hellaswag` 验证集 10,042 条 | harness 的 `hellaswag` | `acc` 与 `acc_norm`（随机水平 0.25） | 官方 |
| WikiText-2 | `EleutherAI/wikitext_document_level` 测试集 62 篇 | harness 的 `wikitext` | bits per byte、word perplexity | 官方 |
| RepoBench 下一行 | `tianyang/repobench_python_v1.1` 的 `in_file` 子集 7,910 条 | 自己的运行器，照官方脚本构造提示、取行、打分 | EM（0–1）、ES（0–100） | 官方打分，设置有两处不同（见第 4 节） |
| HumanEval 单行填空 | `loubnabnl/humaneval_infilling` 的 SingleLine 子集 1,033 条 | 自己的运行器 | 与标准答案行精确匹配的比例 | 非官方指标（见第 3 节） |

三个 harness 任务的指标里，bits per byte 是唯一与分词器无关的：它按原文的字节数归一，词表 16k 的黑米和词表 50k 的 GPT-2 可以直接比。word perplexity 也按原文的词数归一，同样可比；token 级的困惑度不可比，不报告。

参照模型是 GPT-2 small（`openai-community/gpt2`，非嵌入参数约 85M，与黑米级 2 的 103M 同一量级）。它在同一台机器上、用同一版本的框架、同样的样本数跑同一套评测，结果写进仓库里的 `docs/official/reference.json`。黑米的数字只和这条线比，不直接和排行榜上的数字比。

## 3. 不做什么，以及原因

- 不执行模型写出来的代码。HumanEval 填空的官方指标是 pass@1：把补出来的代码拼回去跑单元测试。这等于在本机运行一个 100M 小模型随机写出的 Python；框架自带的 `humaneval_infilling` 任务为此要求显式确认 `confirm_run_unsafe_code`，它依赖的 `code_eval` 在 Windows 上也不能原生运行。所以这里只做精确匹配。补出的行和标准答案一样，拼回去就是标准答案，必然通过测试，因此精确匹配率是 pass@1 的下界。窗口和文档里都把它标成"非官方指标"。
- 不用 CodeXGLUE 的行补全。它的数据是预先分好词的（字面量被替换成占位符、token 之间加了空格），与黑米的训练分布差得太远，零样本的数字没有意义。上一轮对话里我提过它，这里更正：RepoBench 才是合适的选择。
- 不做 RepoBench 的跨文件设置。官方提示把其它文件的片段放在当前文件之前，总长按 15,800 token 截断；黑米的上下文只有 1,024，跨文件片段会被全部截掉。所以只跑 `in_file` 子集，提示只含当前文件。
- 不报 CodeBLEU（需要 tree-sitter 语法包）。不跑 SAFIM（需要执行单元测试）。
- 不改训练流程。`minids/train/`、`train.py`、`evaluate.py` 一行不动；官方评测是训练之外单独运行的命令。

## 4. 公平性规则

1. 同一台机器，同一个框架版本（`lm_eval[hf]==0.4.13`，写死在依赖里），同样的样本数。每条记录带上框架版本、`limit`（框架任务每个子任务取前多少条，空为全量）和 `code_limit`（代码评测抽多少条，空为全量）。两条记录的这两个数不一样，就不放在一起比：窗口把这样的行标成"抽样"。
2. 打分用 float32，不开混合精度。BLiMP 比的是两句话的对数似然谁大，差距常常很小，bf16 的舍入会翻转结果；GPT-2 在框架里也是 float32。
3. 解码一律贪心。RepoBench 官方脚本用 temperature 0.2、top_p 0.95 采样；这里改成贪心是为了结果可复现，GPT-2 参照线用同样的贪心重跑，所以两者之间公平，但都不能直接对排行榜。
4. 代码评测默认抽 500 条（固定种子 1337，取样后按原顺序）。黑米的生成没有 KV 缓存，每个新 token 都是一次完整前向，7,910 条全量在 105M 模型上要几个小时。`--code-all` 可以跑全量。
5. GPT-2 不会填空（没有见过 FIM 标记，也不能以后缀为条件）。单行填空因此报两个数：`fim`（前缀加后缀，用 FIM 提示）和 `prefix`（只给前缀，从左到右写一行）。GPT-2 只有 `prefix`。黑米任务阶段之前的检查点没训练过 FIM 标记，`fim` 这一栏预期接近 0；两栏的差距正好是 FIM 训练买到的东西。
6. 空上下文的条件 token 用文档分隔符 `<|endoftext|>`（id 0），与框架对 GPT-2 的做法相同，也与黑米训练数据里每篇文档前的分隔符一致。

## 5. 架构

```
minids/official/scoring.py     与框架无关的打分核心：给定上下文与续写的 token，返回续写的对数似然和"是否贪心"
minids/official/harness.py     框架适配：HeimiLM（lm_eval.api.model.LM 的子类）、结果摊平、调用 simple_evaluate
minids/official/codebench.py   代码评测的纯函数：取第一行代码、EM、ES、RepoBench 提示、抽样、两个计分器
minids/official/codedata.py    两份代码评测数据的下载与读取
minids/official/completers.py  黑米的两个补全函数（下一行、单行填空），带截断与提前停止
minids/official/reference.py   GPT-2 参照：加载、两个补全函数、整套评测
minids/official/results.py     记录的数据类、official.jsonl 读写、挑待评测的检查点、参照文件读写
minids/official/run.py         把以上拼成"评一个检查点 → 一条记录"
official_eval.py               命令行入口
```

依赖方向：`scoring`、`codebench`、`results` 只依赖 torch 或标准库；只有 `harness` 在模块顶层 import `lm_eval`，只有 `reference` 在函数内 import `transformers`。不装官方评测依赖时，训练、自有评测、窗口全部照常工作。

打分核心不依赖框架，是为了让它能被直接测试：上下文左截断、批内右填充、滚动窗口这三处最容易错，都用迷你模型对着直接计算的结果断言；滚动窗口另外对着框架自己的 `get_rolling_token_windows` 断言逐窗相等。

## 6. 结果放在哪里

- `runs/<run>/official.jsonl`：每评一个检查点追加一行。字段：`checkpoint`（文件名）、`stage`、`global_step`、`tokens_total`、`params_non_embedding`、`limit`、`code_limit`、`metrics`（指标名到数值的字典）、`harness_version`、`time`。
- `docs/official/reference.json`：`{"gpt2": {label, model_id, params_non_embedding, limit, code_limit, metrics, harness_version, time}}`。提交进仓库，窗口只读它。
- 指标名：`blimp_acc`、`hellaswag_acc`、`hellaswag_acc_norm`、`wikitext_bits_per_byte`、`wikitext_byte_perplexity`、`wikitext_word_perplexity`、`repobench_em`、`repobench_es`、`repobench_n`、`infill_em_fim`、`infill_em_prefix`、`infill_n`。

`limit` 为空表示框架任务全量。`code_limit` 为空表示代码评测全量，为 0 表示没跑代码评测，为正数表示抽样条数。

同一个检查点有多条记录时只展示一条：先要框架任务全量的，再要带代码评测的，都一样就要最后写入的。

## 7. 命令行

```
python official_eval.py --run heimi                 # 评 runs/heimi/ckpt 下每个还没评过的 *_final.pt
python official_eval.py --run heimi --ckpt stage1_step500.pt --limit 50 --code-limit 50
python official_eval.py --reference gpt2            # 重跑参照线，写 docs/official/reference.json
```

选项：`--device`（默认有 GPU 用 GPU）、`--limit N`、`--code-limit N`（默认 500）、`--code-all`（代码评测全量）、`--skip-code`、`--force`（已评过也重评）、`--allow-busy-gpu`。

显卡已被占用一半以上时（多半是训练在跑）拒绝在 GPU 上启动：Windows 上显存不够不会报错，而是溢出到系统内存，两边一起慢一到两个数量级。可以等训练停下、改用 `--device cpu`、或明确加 `--allow-busy-gpu`。

## 8. 窗口

主文档 12.4 节的四个区域之外加第五个："官方评测"。一张表：第一行是 GPT-2 参照，后面每个评过的检查点一行，按训练进度排；列是参数量、BLiMP、HellaSwag（`acc_norm`）、WikiText bits/byte、RepoBench EM 与 ES、单行填空的 `fim` 与 `prefix`。表下一句话说明随机水平和每列的方向。没有结果时显示怎么运行。新增接口 `GET /api/runs/{run}/official`，返回 `{rows, reference}`。窗口仍然只读。

## 9. 怎么算做完

1. `official_eval.py --reference gpt2` 在本机跑通，GPT-2 的数字落在公开报告的范围内（BLiMP 0.75 以上，HellaSwag `acc_norm` 0.28–0.34，WikiText word perplexity 30–45）。这一步验证的是数据、框架和本机环境。这三个范围来自我对公开报告的记忆，写文档这天没有逐一核实；它们的用途只是发现明显的环境或数据错误，不是验收线。
2. `official_eval.py --run heimi` 给级 0 的检查点产出一条全量记录，窗口的第五个区域里能看到它和 GPT-2 并排。
3. 之后每训完一级，用户运行一次同样的命令，表里多一行。
