# 黑米数据管线：真实数据跑通报告

日期：2026-11-09　分支：`plan2-data-pipeline`　机器：i7-14700K / 32 GB 内存 / NVMe，全程 CPU，不用 GPU。

本报告对应计划二 Task 6：用 `minids.data` 的四个命令行把 FineWeb-Edu 与 github-code-clean 的原始 parquet
变成训练可直接读取的 uint16 token 分片，并抽样核对结果。所有数字来自 `data/tokens/index.json`、
文件系统和一次全量分片扫描，没有估算值除非注明。

## 1. 执行的命令

```bash
.venv/Scripts/python.exe -m minids.data.download --source all --root data
.venv/Scripts/python.exe -m minids.data.prepare val-rows --root data --out data/tokens/val_rows.json
.venv/Scripts/python.exe -m minids.data.tokenizer train --root data --out data/tokenizer/heimi-16k.json --val-rows data/tokens/val_rows.json
.venv/Scripts/python.exe -m minids.data.prepare run --root data --tokenizer data/tokenizer/heimi-16k.json --val-rows data/tokens/val_rows.json --source all
```

## 2. 每个来源的统计

| 项目 | web（FineWeb-Edu sample/10BT 前 8 片） | code（github-code-clean 前 20 片，只取 Python/JavaScript/TypeScript/Shell） |
|---|---:|---:|
| 原始 parquet | 8 个文件，17.22 GB | 20 个文件，7.16 GB（含全部 30 种语言） |
| 文档数（非空） | 5,856,000 | 480,197 |
| 跳过（null 或空串） | 0 | 0 |
| 验证集行数 `val_rows.json` | 16,892（第 0 片前 16,892 行） | 51,310（第 0 片前 51,310 行，其中 10,113 行是目标语言） |
| 训练 token | 6,579,272,268 | 1,166,420,903 |
| 验证 token | 19,051,766 | 23,861,340 |
| 训练分片 | 66 片（65 片满 1 亿 + 1 片 79,272,268） | 12 片（11 片满 1 亿 + 1 片 66,420,903） |
| 验证分片 | 1 片 | 1 片 |
| 每 token 平均字节数 ¹ | 4.23 | 3.48 |
| 分词耗时 | 3,239 秒（54.0 分钟），2.04M token/s | 695 秒（11.6 分钟），1.71M token/s |

¹ 在各来源第 0 片验证行之后取 3,000 篇文档，UTF-8 字节数 ÷（token 数 + 每篇 1 个 `<|endoftext|>`）。
web 15.2 MB → 3.59M token，code 26.2 MB → 7.54M token。分词器训练时的单文档抽查是 web 4.68、Python 3.00、
JavaScript 3.46、Shell 2.24，和这里的整体均值一致。

## 3. 磁盘占用

| 目录 | 字节 | 说明 |
|---|---:|---|
| `data/raw/web` | 17,217,181,603 | 原始 parquet，可删 |
| `data/raw/code` | 7,155,285,967 | 原始 parquet，可删 |
| `data/tokens/web` | 13,196,648,068 | 67 个 `.bin`，= token 数 × 2 字节 |
| `data/tokens/code` | 2,380,564,486 | 13 个 `.bin` |
| `data/tokenizer/heimi-16k.json` | 1,128,247 | 词表 16,384，前 16 个 id 为特殊 token |
| 合计 | 39.95 GB | 删掉 raw 后 15.58 GB |

## 4. 各步骤耗时

| 步骤 | 耗时 | 备注 |
|---|---:|---|
| 下载 web 17.2 GB | ≈ 3 分钟 | 由各文件落盘时间推算（23:53:35 到 23:56:16 写完 7 片，每片约 23 秒，约 90 MB/s） |
| 下载 code 7.2 GB | ≈ 4 分钟 | 23:56:31 到 00:00:38，中间含换数据源的停顿 |
| `prepare val-rows` | ≈ 10 秒 | 顺序读两个第 0 片直到累计 8000 万字节 |
| `tokenizer train` | 111 秒 | 语料 366,465 篇、2.00 GB（web 1.4 GB + code 0.6 GB），跳过验证行 |
| `prepare run` web | 3,239 秒 | 00:03 到 00:57 |
| `prepare run` code | 695 秒 | 00:57 到 01:09 |
| 全量完整性扫描（见第 5 节） | 14 秒 | 读完 15.6 GB 分片 |

下载到分片写完总共约 1 小时 20 分钟，其中预处理 65.6 分钟。整个过程由 `keep_awake()` 保持机器不休眠。

## 5. 完整性核对

对全部 80 个分片做了一次独立扫描（numpy memmap，不经过加载器）：

| 来源/切分 | 分片 | token 数（与 index.json 一致） | `<|endoftext|>` 数 | 最大 token id | 文件大小 = token × 2 |
|---|---:|---:|---:|---:|:---:|
| web/train | 66 | 6,579,272,268 | 5,839,108 | 16,383 | 是 |
| web/val | 1 | 19,051,766 | 16,892 | 16,383 | 是 |
| code/train | 12 | 1,166,420,903 | 470,084 | 16,383 | 是 |
| code/val | 1 | 23,861,340 | 10,113 | 16,383 | 是 |

- web：5,839,108 + 16,892 = 5,856,000，等于文档数；code：470,084 + 10,113 = 480,197，等于文档数。
  每篇文档恰好一个结束符，没有多写也没有漏写，验证集行数与 `val_rows.json` 完全对应。
- 所有 token id ≤ 16,383 < 16,384，没有越界值；每个分片的字节数都等于 index 里的 token 数 × 2。

## 6. 加载器抽样检查

`ShardSet.from_index` + `MixedLoader(MixConfig(web=0.5, code=0.5), batch_size=16, seq_len=1024, seed=1)`：

- 返回 `x`、`y` 形状都是 `(16, 1024)`、`int64`；`y` 等于 `x` 左移一位；同一 `step` 取两次结果完全相同。
- 16 行里 11 行 web、5 行 code，与 0.5 的配比在这个样本量下相符。
- 解码 3 行 web、3 行 code：都是通顺英文和合法代码；跨文档的窗口里 `<|endoftext|>` 恰好出现在两篇文章的交界处。

web 样例（第 3 行，截取 230 字符，含一处文档边界）：

```
...a plan to resolve any issues that arise. With the right partner, you can be confident in your business’s future success!<|endoftext|>In ancient times astrology and astronomy were the same thing. Astrology as we know it origin
```

code 样例（第 2 行，JavaScript，截取 290 字符）：

```js
      case 3: return function(value, index, collection) {
        return func.call(context, value, index, collection);
      };
      case 4: return function(accumulator, value, index, collection) {
        return func.call(context, accumulator, value, index, collection);
      };
    }
```

FIM 样例：`MixedLoader({"code": ...}, MixConfig(web=0, code=1, fim_rate=1), batch_size=4, seq_len=1024, seed=7)` 第 0 行，
长度 1,025；标记位置 prefix 0、suffix 57、middle 849、eot 1,024，顺序正确，prefix 56 个 token、suffix 791 个、middle 174 个：

```
<|fim_prefix|> case "treegrid":
            case "treeitem":
                return true;
            }
        }
        return false;
    };
    CaretBrowsing.injectCaretStyles = function() {
        const style = ".CaretBrowsing_Caret {" +
           <|fim_suffix|>();
        CaretBrowsing.initiated = true;
        ...
```

验证集分片同样能用 `ShardSet.from_index(..., "val", ...)` 读出并解码成正常文本和代码。

## 7. 与 spec 第 6 节目标的对照

| 目标 | spec 要求 | 实测 | 结论 |
|---|---|---|---|
| web 规模 | 约 5.5B GPT-2 等价 token，16k 词表下多 10% 到 20%（即 6.05B 到 6.6B） | 6.598B（训练 6.579B + 验证 0.019B） | 达标，落在预计区间上沿 |
| code 规模 | 约 1B token | 1.190B（训练 1.166B + 验证 0.024B） | 达标，多 19% |
| 验证集 | web 与 code 各约 2000 万 token | web 1,905 万、code 2,386 万 | 在计划允许的 1500 万到 3000 万之间，不需要重跑 |
| 文本列为空的行 | 跳过并计数 | 两个来源都是 0 行 | 两个数据集都已清洗，分支未触发；单元测试覆盖 |

训练消耗对照：按 spec 第 8 节的预算，一次完整的阶梯训练（0.3B + 0.5B + 2.0B，web 0.85 / code 0.15）再接任务阶段
（0.7B，web 0.20 / code 0.80）共需 web 2.52B、code 0.98B；103M 基线单独一次 2.0B 需 web 1.70B、code 0.30B。
每一次运行对两个来源都不到一个 epoch，不需要再下载分片。

## 8. 观察与注意事项

- **代码验证集的行数口径**：`val_rows.json` 的 51,310 是原文件行号，语言过滤后实际进入验证集的是 10,113 篇；
  `estimate_val_rows` 是对过滤后的文本累计字节，所以验证 token 数仍落在目标附近。
- **FineWeb-Edu 里有少量机器改写文本**：抽样第 0 行出现 "native towards the condition"、"Countless decades" 这类同义词替换痕迹，
  属于数据集本身的噪声，比例不大，计划三评估 loss 时不必处理。
- **代码注释不全是英文**：语言过滤只看编程语言，抽到的代码里有韩文和法文注释。字节级 BPE 对它们按字节回退（每个 CJK 字符约 3 个 token），
  不会出错，只是略浪费 token。若以后要更严格，可在计划三之前对 code 加一个 ASCII 比例过滤；本计划不做。
- **后台日志的中文乱码**：`prepare run` 在后台由 GBK 代码页的控制台捕获时日志中文变成乱码，分片与 `index.json` 不受影响。
  以后在后台跑时加 `PYTHONUTF8=1`。
- **分词器只能经 `HeimiTokenizer.load` 加载**："字面特殊串按普通文本编码"这条规则不在 `heimi-16k.json` 里，直接 `tokenizers.Tokenizer.from_file` 会把文本里的 `<|endoftext|>` 识别成 id 0。审阅后已加载路径测试与模块说明。
- **加载器会核对分片大小**：`ShardSet` 打开每个分片前检查文件字节数等于 index 的 token 数 × 2，复制不完整或 index 过期时报带分片名的错误，而不是 memmap 的乱码 OSError。
- `data/raw` 的 24.4 GB 在分片写完后就不再需要，可以删除；保留 `data/tokens` 与 `data/tokenizer` 即可训练。
