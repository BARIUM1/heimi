# 级 0 实跑与级 1 入口试跑报告（2026-11-09）

机器：RTX 5080 笔记本版 16 GB，torch 2.11.0+cu128，Windows 11。代码：分支 `plan3-training`，计划三 Task 14。
数据：`data/tokens/`（计划二产出，web 6.58B / code 1.17B 训练 token），分词器 `data/tokenizer/heimi-16k.json`。
GPU 开跑前已用显存 3.6 GB（桌面程序），未被其他重负载占用。

## 命令

```
python train.py --config configs/ladder.yaml --run heimi --stage stage0 --device cuda      # 级 0，自动下载 HellaSwag
python train.py --config configs/ladder.yaml --run heimi --device cuda --max-steps 20       # 跳过级 0，进入级 1 跑 20 步后保存退出
python evaluate.py --ckpt runs/heimi/ckpt/stage0_final.pt --device cuda
python sample.py --ckpt runs/heimi/ckpt/stage0_final.pt --prompt "The history of the Roman Empire began when" --device cpu
python export.py --ckpt runs/heimi/ckpt/stage0_final.pt --out models/heimi-stage0
python complete.py --model models/heimi-stage0 --prefix-file prefix.txt --suffix-file suffix.txt --max-tokens 32
```

全部命令前加 `PYTHONUTF8=1`（GBK 控制台下中文日志会乱码）。

## 级 0：8 层 d384，12,934,528 非嵌入参数，0.3B token

| 项 | 值 |
|---|---|
| 优化步 | 572（每步 524,288 token = 16 条 × 1024 × 32 个微步） |
| 实际 token | 299,892,736 |
| 用时 | 2,326 秒 = 38.8 分钟（含第 500 步与级末两次评测；级末评测 7.8 分钟），纯训练约 31 分钟 |
| 吞吐 | 前 200 步 220k 到 235k token/s；第 250 步起 120k 到 170k；全程均值 171k |
| 峰值显存（reserved） | 8.18 GB |
| GPU 温度 | 51 到 60 °C，全程平稳 |
| 训练损失 | 7.72（第 10 步）→ 4.36（第 572 步），最低 4.33 |
| 验证损失 web | 4.523（第 500 步）→ 4.471（级末） |
| 验证损失 code | 3.556（第 500 步）→ 3.487（级末） |
| HellaSwag | 0.2366（随机为 0.25，spec 第 16 节预期 13M 无区分度） |
| FIM 单行精确匹配 | 0.0（阶梯阶段没见过 FIM 格式，作为基线） |
| 检查点 | `stage0_step500.pt`、`stage0_final.pt`，各 306 MB |
| 事件 | `stage_start`、`checkpoint`（第 500 步）、`stage_end`；没有 `nan_abort` / `oom` |

级末固定提示的续写（`samples.jsonl` 第 572 步）：

- `en_history`："The history of the Roman Empire began when" → " the modern Roman Empire began in the 1960s, according to the French Revolution of 1919. Since the 1960s, the"
- `en_science`："Water boils at a lower temperature on a mountain because" → " of snowy rain: warm water and winter, this is too cold.\nGiven the cold, dryer in the year, water is the same"
- `py_func`：产出以 `#` 开头的英文注释，不是代码
- `fim_py`：`def is_even(n):\n    ` 与 `\n\nprint(is_even(4))` 之间填出 "(n)"

英文已经成句、主题相关但事实混乱，符合 13M 模型训练 0.3B token 的预期。

## 级 1 入口试跑：真实模型上的加深与恢复

| 项 | 值 |
|---|---|
| 增长 | depth，12,934,528 → 25,868,672 非嵌入参数 |
| 验证 | `exact: true`，`max_abs_diff: 0.0`，耗时 1.4 秒 |
| 增长点前后损失 | 级 0 末步 4.3614 → 级 1 第 10 步 4.3397（第 20 步 4.3851，预热中学习率在升） |
| 吞吐 | 104k 到 109k token/s（基准 133k，见观察 1） |
| 峰值显存（reserved） | 11.35 GB |
| 计数连续性 | `global_step` 572 → 582，`tokens_total` 299,892,736 → 305,135,616 |
| 检查点 | `stage1_step20.pt`（`reason: max_steps`）；再跑 `train.py --run heimi` 会从它恢复 |

## 工具链

- `evaluate.py`（GPU，287 秒）：对 `stage0_final.pt` 重算得到 val web 4.4713 / code 3.4870 / HellaSwag 0.2366 / FIM 0.0，四项与训练时的级末记录逐位相同（固定窗口、固定样例、固定种子），已作为第 3 条追加到 `runs/heimi/eval.jsonl`
- `sample.py`（CPU，48 个 token，3.5 秒）："The history of the Roman Empire began when the Japanese military leader of the Nileo River from the French, the French government, was inquering the British culture where it was later formed under the establishment of Halachini to find the Kurata Peninsula. The"
- `export.py` → `models/heimi-stage0/`：`model.safetensors` 102 MB、`config.json`（结构、级名、步数、token 数、git 提交）、`tokenizer.json` 1.1 MB
- `complete.py`（CPU，导出目录）：前缀 `def add(a, b):\n    `、后缀 `\n\nprint(add(1, 2))\n` → "(1, 2)"，2.9 秒；标准输入形式 `echo import os | python complete.py --model ...` → "_path = os.path.path.path.path.join(path"

## 与基准预测的对照（`docs/benchmarks/throughput.md`）

| 项 | 预测 | 实测 |
|---|---|---|
| 级 0 吞吐（微批 16） | 228k token/s | 前 200 步 220k 到 235k，之后 120k 到 170k |
| 级 0 训练用时 | 约 22 分钟 | 约 31 分钟（加评测 38.8 分钟） |
| 级 0 峰值显存 | 7.23 GB（allocated） | 8.18 GB（reserved） |
| 级 1 吞吐（微批 16） | 133k token/s | 104k 到 109k |
| 级 1 峰值显存 | 10.45 GB（allocated） | 11.35 GB（reserved） |

显存与预测一致（reserved 比 allocated 高约 1 GB 属正常），两级都远低于 12.8 GB 的溢出判定线。吞吐前 200 步与基准一致，之后持续低约 40%。

## 观察

1. **吞吐在第 250 步附近掉到 120k 到 170k 并保持到结束，级 1 试跑也只有基准的 80%。** GPU 温度全程 51 到 60 °C，不是过热降频。更可能是别的程序开始用 GPU（开跑时桌面程序已占 3.6 GB）或笔记本电源模式变化。夜间跑之前：插电、电源计划设为高性能、关掉会用 GPU 的程序（游戏、壁纸引擎、浏览器硬件加速）。窗口（计划四）的 tok/s 曲线能直接看出这类下降。
2. **级末评测 7.8 分钟**，大头是 1000 个 FIM 样本逐 token 贪心生成（每个样本最多 64 次完整前向）。105M 模型预计 15 到 20 分钟，任务阶段每 500 步评测一次会花掉约 15% 的时间。可改成批量生成或把阶梯阶段的 FIM 样本数调小；本期不改。
3. **HellaSwag 0.2366**，随机水平，与 spec 第 16 节的预期一致；它要到级 2 与任务阶段才有区分度。
4. **13M 模型已经能补出结构**：FIM 填空 "(n)"、代码续写知道用 `os.path`，虽然内容不对。
5. 训练进程写出的 jsonl 四个文件格式与 `logging_io.py` 一致，窗口可直接读；`samples.jsonl` 每次 6 条。

## 剩余各级怎么跑

一条命令续跑整条阶梯，中断后重跑同一命令即从最近检查点恢复：

```
set PYTHONUTF8=1
python train.py --config configs/ladder.yaml --run heimi --device cuda
```

| 级 | 步数 | 基准吞吐下预计 | 若吞吐只有 80% |
|---|---|---|---|
| stage1（剩 933 步） | 953 | 约 1.0 小时 | 约 1.3 小时 |
| stage2 | 3,814 | 约 10.6 小时 | 约 13 小时 |
| task | 1,335 | 约 3.7 小时 | 约 4.7 小时 |
| baseline（另一条命令） | 3,814 | 约 10.6 小时 | 约 13 小时 |

每级结束还有级末评测（级 2 与任务阶段预计 15 到 20 分钟）。对照组：

```
python train.py --config configs/baseline105m.yaml --run baseline --device cuda
```

检查点占用：级 1 每个约 460 MB，级 2 / 任务 / 对照每个约 1.5 GB，每级最多 2 个中间加 1 个最终，总计约 15 GB；F 盘目前剩 241 GB。

**spec 7.3 留下的校准项**：进入 stage2 的加宽验证若因 `max_rel_err > rel_tol` 中止（日志里是 `GrowthVerificationError`），调小 `configs/ladder.yaml` 里 stage2 的 `noise_std_rel`（如 5.0e-4）或放宽 `rel_tol`（如 2.0e-2），重跑同一命令即可：增长在进入该级时重新执行，级 1 的最终检查点不受影响。
