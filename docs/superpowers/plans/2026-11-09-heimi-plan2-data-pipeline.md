# 黑米 计划二：数据管线 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 FineWeb-Edu 和 stack-edu 的原始 parquet 变成训练可直接读取的 uint16 token 分片，并提供按配比取 batch、支持 FIM 改写、完全确定性的加载器。

**Architecture:** `minids/data/` 下五个模块各管一段：`sources` 定义数据源与本地布局，`download` 拉取并校验，`tokenizer` 训练并封装 16k 字节级 BPE，`prepare` 把 parquet 流式分词写成分片并生成 `index.json`，`loader` 用 memmap 从分片随机取窗口并混合两个来源，`fim` 做 PSM 改写与还原。单元测试全部用 tmp 目录里的假数据，不访问网络、不依赖 GPU；最后一个任务在真实数据上跑通并出报告。

**Tech Stack:** Python 3.12、huggingface_hub（下载）、pyarrow（读 parquet）、tokenizers（BPE）、numpy（memmap 分片）、torch（batch 张量）、pytest。

**Spec:** `docs/superpowers/specs/2026-11-08-heimi-growth-lm-design.md`（本计划实现第 6 节全部内容，以及第 13.1 节中 `test_loader`、`test_fim`、`test_tokenizer` 三项；第 14 节关于下载与数据损坏的容错也在此落地）

## 本计划在整体中的位置

计划一（已合并）交付了模型与增长算子。本计划交付数据；完成后磁盘上有约 12 到 15 GB 的 token 分片和一个 `index.json`，`MixedLoader.get_batch(step)` 能返回 `(x, y)` 张量。计划三把它和模型接起来训练。

## 已核实的外部事实（2026-11-08 查 Hugging Face API）

- `HuggingFaceFW/fineweb-edu`，`sample/10BT/` 下 14 个 parquet 各 2.15 GB，不需要权限，许可 ODC-By。取前 8 个：`000_00000.parquet` 到 `007_00000.parquet`。
- `HuggingFaceTB/stack-edu`，不需要权限，按语言分目录。取 `Python/train-00000-of-00005.parquet`、`Python/train-00001-of-00005.parquet`（各 500 MB）、`Shell/train-00000-of-00001.parquet`（395 MB）、`JavaScript/train-00000-of-00003.parquet`（429 MB）、`TypeScript/train-00000-of-00001.parquet`（426 MB），合计约 2.25 GB parquet。
- 两个数据集的文本列名没能在线确认，预处理按 `text`、`content` 的顺序自动识别，都没有则报错列出全部列名。

## Global Constraints

- 词表总大小 16384；前 16 个 id 为特殊 token，顺序固定：0 `<|endoftext|>`、1 `<|fim_prefix|>`、2 `<|fim_middle|>`、3 `<|fim_suffix|>`、4 `<|pad|>`、5 到 15 `<|reserved_5|>` 到 `<|reserved_15|>`。
- 分词器是字节级 BPE（`tokenizers` 库），训练语料为 web 与 code 约 7:3 的抽样，合计约 2 GB 文本。词表一经训练不再更改。
- 数据里字面出现的特殊 token 字符串必须按普通文本编码，特殊 id 只能由代码显式插入。
- 分片格式：小端 uint16 原始数组，无文件头；每片 1 亿 token；文档之间插入 `<|endoftext|>`；文件名 `data/tokens/{web,code}/{train,val}_{NNNN}.bin`；`data/tokens/index.json` 记录每片 token 数。
- 验证集：web 与 code 各留出约 2000 万 token，取各自第一个 parquet 文件最前面的 N 行；N 由 `estimate_val_rows` 按累计文本字节 ≥ 8000 万（按每 token 约 4 字节估）一次算出并写入 `data/tokens/val_rows.json`，分词器训练和预处理都读同一个文件，保证验证集既不进训练也不进分词器语料。
- FIM 作用于加载器采样出的窗口（可能跨越文档边界，窗口内的 `<|endoftext|>` 保持原样），不是严格的单文档：加载器按固定长度从打包流取样，这是与 spec 6.5"文档级"措辞的有意差异，格式与切分规则不变。
- 加载器随机性由 `(seed, step)` 决定，恢复训练后同一 step 取到同一 batch。
- 配比由配置给出，默认阶梯阶段 web 0.85 / code 0.15，任务阶段 web 0.20 / code 0.80。
- FIM 采用 PSM 格式 `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>`，只对 code 窗口按比例应用（默认 0.5，阶梯阶段为 0）。
- 下载支持断点续传，按远端文件大小校验，失败最多重试 3 次；解析失败的行跳过并计数。
- 单元测试不访问网络、不依赖 GPU；注释中文、标识符英文；`data/` 在 git 之外。
- 每个 commit 信息用英文 conventional commits 格式，末尾加一行 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。

## Review Focus

1. 网页或代码里字面出现 `<|endoftext|>` 这类字符串：`HeimiTokenizer.encode` 必须把它编码成普通字节序列，不能产出 id 0；否则一段恶意或巧合的文本就能在训练流里插入控制 token。测试在 Task 2。
2. 一篇文档跨越分片边界（或长于一片）：写入器必须把它切开写进相邻两片，token 不丢不重，拼回来与原序列完全相同。测试在 Task 3。
3. parquet 里文本列为 null 或空字符串的行：跳过并计入 `skipped`，不写入任何 token，也不写多余的 `<|endoftext|>`。测试在 Task 3。
4. 某个分片 token 数不足 `seq_len + 1`：加载器构造时把它排除并给出警告，而不是取出越界窗口；全部分片都不足时抛出清楚的 `ValueError`。测试在 Task 5。
5. 分片里出现 ≥ 16384 的 token 值（文件损坏）：`get_batch` 在 uint16 转 int64 时检查并抛出含分片名的 `ValueError`，不能让它流到 GPU 上变成 device-side assert。测试在 Task 5。

---

## 文件结构

```
pyproject.toml                  # 依赖加 huggingface_hub、pyarrow、tokenizers（Task 1 修改）
minids/data/__init__.py
minids/data/sources.py          # 数据源注册表、本地路径规则、文本列候选
minids/data/download.py         # 下载、大小校验、重试；CLI
minids/data/tokenizer.py        # 特殊 token 常量、训练、HeimiTokenizer 封装、训练语料抽样；CLI
minids/data/prepare.py          # parquet 流式读取、ShardWriter、index.json；CLI
minids/data/fim.py              # FimIds、apply_fim_psm、undo_fim_psm
minids/data/loader.py           # ShardSet、MixedLoader
docs/data/report.md             # Task 6 产出：真实数据统计
tests/test_sources.py
tests/test_download.py
tests/test_tokenizer.py
tests/test_prepare.py
tests/test_fim.py
tests/test_loader.py
tests/helpers_data.py           # 测试共用：写迷你 parquet、训迷你分词器、写迷你分片
```

---

### Task 1: 数据源注册与下载

**Files:**
- Modify: `pyproject.toml`（dependencies 加 `huggingface_hub>=0.25`、`pyarrow>=15`、`tokenizers>=0.20`）
- Create: `minids/data/__init__.py`、`minids/data/sources.py`、`minids/data/download.py`
- Test: `tests/test_sources.py`、`tests/test_download.py`

**Interfaces:**
- Produces（`sources.py`）：
  ```python
  @dataclass(frozen=True)
  class Source:
      name: str              # "web" | "code"
      repo_id: str
      files: tuple[str, ...] # 仓库内路径
  WEB_FILES: tuple[str, ...]   # sample/10BT/000_00000.parquet … 007_00000.parquet
  CODE_FILES: tuple[str, ...]  # 上文核实的 5 个 stack-edu 文件
  SOURCES: dict[str, Source]   # {"web": ..., "code": ...}
  TEXT_COLUMNS: tuple[str, ...] = ("text", "content")
  def raw_dir(root: Path, source: Source) -> Path          # root / "raw" / source.name
  def local_path(root: Path, source: Source, path_in_repo: str) -> Path
      # raw_dir / path_in_repo.replace("/", "__")，例如 data/raw/code/Python__train-00000-of-00005.parquet
  ```
- Produces（`download.py`）：
  ```python
  class DownloadError(RuntimeError): ...
  def remote_sizes(repo_id: str, paths: Sequence[str]) -> dict[str, int]
      # HfApi().get_paths_info(repo_id, paths, repo_type="dataset") 的 size
  def download_source(
      source: Source, root: Path, retries: int = 3,
      fetch: Callable[[str, str, Path], Path] | None = None,   # (repo_id, path_in_repo, dest) -> dest
      sizes: dict[str, int] | None = None,
  ) -> list[Path]
      # 对每个文件：本地存在且大小等于远端 → 跳过；否则调用 fetch 下载到 dest，
      # 校验大小，不符则删除并重试，retries 次后抛 DownloadError（信息含文件名、期望与实际大小）。
      # fetch 为 None 时用 huggingface_hub.hf_hub_download(repo_id, path, repo_type="dataset",
      # local_dir=<临时目录>) 再移动到 dest；sizes 为 None 时调用 remote_sizes。
  ```
  CLI：`python -m minids.data.download --source {web,code,all} --root data`，逐文件打印"跳过/下载/校验通过"，结束打印总字节数。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_sources.py
def test_registry_matches_verified_files():
    assert SOURCES["web"].repo_id == "HuggingFaceFW/fineweb-edu" and len(WEB_FILES) == 8
    assert WEB_FILES[0] == "sample/10BT/000_00000.parquet" and WEB_FILES[-1] == "sample/10BT/007_00000.parquet"
    assert SOURCES["code"].repo_id == "HuggingFaceTB/stack-edu" and len(CODE_FILES) == 5
    assert "Shell/train-00000-of-00001.parquet" in CODE_FILES

def test_local_path_flattens_subdirs(tmp_path):
    p = local_path(tmp_path, SOURCES["code"], "Python/train-00000-of-00005.parquet")
    assert p == tmp_path / "raw" / "code" / "Python__train-00000-of-00005.parquet"
```

```python
# tests/test_download.py
def _fake_fetch(payloads: dict[str, bytes], fail_first: set[str] = frozenset()):
    calls = []
    def fetch(repo_id, path, dest):
        calls.append(path)
        data = payloads[path]
        if path in fail_first and calls.count(path) == 1:
            data = data[:-1]            # 第一次下载缺一个字节，模拟中断
        dest.write_bytes(data); return dest
    return fetch, calls

def test_skips_existing_file_with_matching_size(tmp_path):
    src = Source("t", "x/y", ("a.parquet",)); dest = local_path(tmp_path, src, "a.parquet")
    dest.parent.mkdir(parents=True); dest.write_bytes(b"abc")
    fetch, calls = _fake_fetch({"a.parquet": b"abc"})
    assert download_source(src, tmp_path, fetch=fetch, sizes={"a.parquet": 3}) == [dest]
    assert calls == []

def test_retries_on_size_mismatch_then_succeeds(tmp_path):
    src = Source("t", "x/y", ("a.parquet",))
    fetch, calls = _fake_fetch({"a.parquet": b"abcd"}, fail_first={"a.parquet"})
    out = download_source(src, tmp_path, fetch=fetch, sizes={"a.parquet": 4})
    assert out[0].read_bytes() == b"abcd" and calls == ["a.parquet", "a.parquet"]

def test_raises_after_retries_and_removes_partial(tmp_path):
    src = Source("t", "x/y", ("a.parquet",))
    def always_short(repo_id, path, dest): dest.write_bytes(b"ab"); return dest
    with pytest.raises(DownloadError, match="a.parquet"):
        download_source(src, tmp_path, retries=3, fetch=always_short, sizes={"a.parquet": 4})
    assert not local_path(tmp_path, src, "a.parquet").exists()
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources.py tests/test_download.py -q`
Expected: FAIL，`ModuleNotFoundError: minids.data`

- [ ] **Step 3: 改 pyproject 并安装依赖，实现 `sources.py` 与 `download.py`**

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

- [ ] **Step 4: 运行，确认通过** → 6 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml minids/data tests/test_sources.py tests/test_download.py
git commit -m "feat(data): source registry and verified, resumable download"
```

---

### Task 2: 分词器

**Files:**
- Create: `minids/data/tokenizer.py`、`tests/helpers_data.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces：
  ```python
  SPECIAL_TOKENS: list[str]   # 16 个，顺序见 Global Constraints
  VOCAB_SIZE = 16384
  EOT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, PAD = 0, 1, 2, 3, 4

  def train_tokenizer(texts: Iterable[str], out_path: Path, vocab_size: int = VOCAB_SIZE) -> "HeimiTokenizer"
      # tokenizers: models.BPE(unk_token=None) + pre_tokenizers.ByteLevel(add_prefix_space=False)
      # + decoders.ByteLevel + trainers.BpeTrainer(vocab_size, special_tokens=SPECIAL_TOKENS,
      #   initial_alphabet=ByteLevel.alphabet(), show_progress=False)；保存为 json

  class HeimiTokenizer:
      def __init__(self, tok: tokenizers.Tokenizer): ...   # 设置 tok.encode_special_tokens = True，
                                                           # 即文本里的特殊 token 字符串按普通文本切分
      @classmethod
      def load(cls, path: Path) -> "HeimiTokenizer"
      @property
      def vocab_size(self) -> int
      def encode(self, text: str) -> list[int]              # 不添加任何特殊 token
      def encode_batch(self, texts: list[str]) -> list[list[int]]
      def decode(self, ids: Sequence[int]) -> str            # 特殊 id 解码回其字符串
      def token_to_id(self, token: str) -> int

  def iter_training_corpus(root: Path, val_rows: dict[str, int],
                           web_bytes: int = 1_400_000_000, code_bytes: int = 600_000_000) -> Iterator[str]
      # web：从第一个 parquet 的第 val_rows["web"] 行之后开始顺序读，不够再读后续文件；
      # code：从第一个 parquet 的第 val_rows["code"] 行之后开始，再读其余文件；
      # 各自读到字节预算用完为止。val_rows 来自 data/tokens/val_rows.json（Task 3 的 val-rows 子命令产出）。
  ```
  CLI：`python -m minids.data.tokenizer train --root data --out data/tokenizer/heimi-16k.json --val-rows data/tokens/val_rows.json`，结束打印词表大小、语料字节数、训练秒数。
- `tests/helpers_data.py` 提供：
  ```python
  def write_parquet(path: Path, texts: list[str | None], column: str = "text") -> Path   # pyarrow 写单列表
  def tiny_tokenizer(tmp_path: Path, vocab_size: int = 300) -> HeimiTokenizer
      # 在一段固定的英文+代码小语料上训练，返回加载好的分词器（特殊 token 同样占 0..15）
  def write_shard(path: Path, tokens: Sequence[int]) -> Path   # np.uint16 写 .bin
  ```

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_tokenizer.py
def test_special_token_ids_are_fixed(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS[:5]] == [0, 1, 2, 3, 4]
    assert tok.token_to_id("<|reserved_15|>") == 15

def test_round_trip_arbitrary_text(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    s = "def f(x):\n\treturn x  # 中文 ünïcode 🙂 \x00\x7f"
    assert tok.decode(tok.encode(s)) == s

def test_literal_special_string_is_not_a_special_id(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    ids = tok.encode("see <|endoftext|> here")
    assert 0 not in ids and tok.decode(ids) == "see <|endoftext|> here"

def test_decode_single_special_id(tmp_path):
    assert tiny_tokenizer(tmp_path).decode([1]) == "<|fim_prefix|>"

def test_vocab_size_is_requested(tmp_path):
    assert tiny_tokenizer(tmp_path, vocab_size=300).vocab_size == 300

def test_encode_batch_matches_encode(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    assert tok.encode_batch(["a b", "c"]) == [tok.encode("a b"), tok.encode("c")]
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `tokenizer.py` 与 `helpers_data.py`**
- [ ] **Step 4: 运行，确认通过** → 6 passed
- [ ] **Step 5: Commit**：`git commit -m "feat(data): 16k byte-level BPE tokenizer with fixed special ids"`

---

### Task 3: 分片写入与预处理

**Files:**
- Create: `minids/data/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes：`HeimiTokenizer`、`EOT`、`TEXT_COLUMNS`、`write_parquet`。
- Produces：
  ```python
  @dataclass(frozen=True)
  class PrepareConfig:
      tokens_per_shard: int = 100_000_000
      batch_rows: int = 1024
      val_target_bytes: int = 80_000_000     # 约 2000 万 token × 4 字节

  def estimate_val_rows(parquet_path: Path, target_bytes: int) -> int
      # 顺序读第一个文件，累计非空文本的 UTF-8 字节数首次 ≥ target_bytes 时的行数（含该行）

  def iter_documents(parquet_path: Path, text_columns: Sequence[str] = TEXT_COLUMNS,
                     batch_rows: int = 1024) -> Iterator[tuple[int, str | None]]
      # 产出 (row_index, text)；用 pyarrow.parquet.ParquetFile.iter_batches(columns=[col])；
      # 列名按 text_columns 顺序取第一个存在的，都不存在抛 ValueError 列出 schema 全部列名

  class ShardWriter:
      def __init__(self, out_dir: Path, prefix: str, tokens_per_shard: int): ...
      def append(self, tokens: Sequence[int]) -> None     # 跨片时切开写入
      def close(self) -> list[dict]                         # [{"file": "train_0000.bin", "tokens": N}, ...]

  def prepare_source(source_name: str, parquet_paths: Sequence[Path], tokenizer: HeimiTokenizer,
                     out_root: Path, val_rows: int, cfg: PrepareConfig = PrepareConfig()) -> dict
      # 第一个文件的前 val_rows 行写 out_root/<source_name>/val_NNNN.bin，其余全部写 train_NNNN.bin；
      # 每篇文档 = encode(text) + [EOT]；null 或空串跳过并计 skipped（跳过的行也计入 val_rows 的行数）；
      # 返回 {"docs": int, "skipped": int, "tokens_train": int, "tokens_val": int, "val_rows": int,
      #       "train": [...], "val": [...]}

  def write_index(out_root: Path, per_source: dict[str, dict], vocab_size: int) -> Path
      # out_root/index.json：{"vocab_size": 16384, "dtype": "uint16", "eot_id": 0,
      #                       "sources": {"web": {...prepare_source 的返回...}, "code": {...}}}
  ```
  CLI 两个子命令：
  - `python -m minids.data.prepare val-rows --root data --out data/tokens/val_rows.json [--target-bytes 80000000]`：对 web 与 code 各调用 `estimate_val_rows`，写出 `{"web": rows, "code": rows}`。不需要分词器。
  - `python -m minids.data.prepare run --root data --tokenizer data/tokenizer/heimi-16k.json --val-rows data/tokens/val_rows.json --source {web,code,all}`：调用 `prepare_source` 并在全部来源完成后 `write_index`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_prepare.py
def test_iter_documents_picks_text_then_content(tmp_path):
    p = write_parquet(tmp_path / "a.parquet", ["x", None, ""], column="content")
    assert list(iter_documents(p)) == [(0, "x"), (1, None), (2, "")]

def test_iter_documents_unknown_column_raises(tmp_path):
    p = write_parquet(tmp_path / "a.parquet", ["x"], column="body")
    with pytest.raises(ValueError, match="body"):
        next(iter_documents(p))

def test_shard_writer_splits_across_shards_without_loss(tmp_path):
    w = ShardWriter(tmp_path, "train", tokens_per_shard=10)
    w.append(list(range(7))); w.append(list(range(100, 118)))   # 第二篇跨两片
    shards = w.close()
    assert [s["tokens"] for s in shards] == [10, 10, 5]
    got = np.concatenate([np.fromfile(tmp_path / s["file"], dtype=np.uint16) for s in shards])
    assert got.tolist() == list(range(7)) + list(range(100, 118))

def test_estimate_val_rows_counts_bytes_of_nonempty_rows(tmp_path):
    p = write_parquet(tmp_path / "a.parquet", ["ab", None, "", "cdef", "gh"])
    assert estimate_val_rows(p, target_bytes=5) == 4      # ab(2) + cdef(4) 在第 4 行达到 ≥ 5

def test_prepare_source_splits_val_rows_then_train_and_counts_skips(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    texts = ["hello world"] * 20 + [None, ""] + ["code = 1"] * 30
    p = write_parquet(tmp_path / "a.parquet", texts)
    stats = prepare_source("web", [p], tok, tmp_path / "tokens", val_rows=22, cfg=PrepareConfig(tokens_per_shard=64))
    assert stats["skipped"] == 2 and stats["docs"] == 50 and stats["val_rows"] == 22
    assert stats["tokens_val"] == sum(len(tok.encode("hello world")) + 1 for _ in range(20))
    assert stats["tokens_train"] == sum(len(tok.encode("code = 1")) + 1 for _ in range(30))
    val = np.concatenate([np.fromfile(tmp_path / "tokens" / "web" / s["file"], dtype=np.uint16) for s in stats["val"]])
    assert val.max() < tok.vocab_size and (val == 0).sum() == 20         # 每篇文档一个分隔符

def test_index_json_shape(tmp_path):
    idx = write_index(tmp_path, {"web": {"train": [], "val": [], "docs": 0, "skipped": 0,
                                         "tokens_train": 0, "tokens_val": 0, "val_rows": 0}}, 16384)
    d = json.loads(idx.read_text(encoding="utf-8"))
    assert d["vocab_size"] == 16384 and d["dtype"] == "uint16" and d["eot_id"] == 0 and "web" in d["sources"]
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `prepare.py`**

`encode_batch` 按 `batch_rows` 批量分词；验证集与训练集各用一个 `ShardWriter`（prefix 分别为 `val` 与 `train`）。

- [ ] **Step 4: 运行，确认通过** → 6 passed
- [ ] **Step 5: Commit**：`git commit -m "feat(data): streaming parquet tokenization into uint16 shards"`

---

### Task 4: FIM 改写

**Files:**
- Create: `minids/data/fim.py`
- Test: `tests/test_fim.py`

**Interfaces:**
- Produces：
  ```python
  @dataclass(frozen=True)
  class FimIds:
      prefix: int = 1; middle: int = 2; suffix: int = 3; eot: int = 0

  def apply_fim_psm(tokens: np.ndarray, rng: np.random.Generator, ids: FimIds = FimIds()) -> np.ndarray
      # 两个切分点 a ≤ b 从 rng.integers(0, n + 1, size=2) 排序得到；
      # 返回 [prefix_id] + tokens[:a] + [suffix_id] + tokens[a:b] 之后的 suffix … 即
      # [P] prefix [S] suffix [M] middle [EOT]，长度 n + 4

  def undo_fim_psm(tokens: np.ndarray, ids: FimIds = FimIds()) -> np.ndarray
      # 按三个标记位置拆出 prefix/suffix/middle，去掉末尾 EOT，返回 prefix + middle + suffix；
      # 标记缺失或顺序不对抛 ValueError
  ```

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_fim.py
def test_layout_and_length():
    rng = np.random.default_rng(0); t = np.arange(20, 40)
    out = apply_fim_psm(t, rng)
    assert len(out) == 24 and out[0] == 1 and out[-1] == 0
    assert list(out).count(3) == 1 and list(out).count(2) == 1
    assert list(out).index(1) < list(out).index(3) < list(out).index(2)

def test_round_trip_many_seeds():
    t = np.arange(100, 160)
    for seed in range(50):
        assert np.array_equal(undo_fim_psm(apply_fim_psm(t, np.random.default_rng(seed))), t)

def test_empty_segments_allowed():
    t = np.arange(10, 15)
    class Rng:  # 强制 a = b = 0：prefix 与 middle 都为空
        def integers(self, lo, hi, size): return np.array([0, 0])
    out = apply_fim_psm(t, Rng())
    assert np.array_equal(undo_fim_psm(out), t)

def test_undo_rejects_malformed():
    with pytest.raises(ValueError):
        undo_fim_psm(np.array([1, 5, 6, 2, 7, 0]))   # 缺 suffix 标记
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `fim.py`**
- [ ] **Step 4: 运行，确认通过** → 4 passed
- [ ] **Step 5: Commit**：`git commit -m "feat(data): PSM fill-in-the-middle transform and inverse"`

---

### Task 5: 加载器

**Files:**
- Create: `minids/data/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes：`write_shard`、`FimIds`、`apply_fim_psm`、`index.json` 的结构（Task 3）。
- Produces：
  ```python
  class ShardSet:
      def __init__(self, files: Sequence[Path], token_counts: Sequence[int], min_tokens: int): ...
          # 用 np.memmap(dtype=np.uint16, mode="r") 打开；token 数 < min_tokens 的分片排除并 warnings.warn；
          # 一个都不剩抛 ValueError
      @classmethod
      def from_index(cls, index_path: Path, source: str, split: str, min_tokens: int) -> "ShardSet"
      @property
      def total_tokens(self) -> int
      @property
      def min_tokens(self) -> int
      def sample_window(self, rng: np.random.Generator, length: int) -> tuple[np.ndarray, str]
          # 先按 (tokens - length + 1) 加权选片，再在片内均匀选起点；窗口不跨片；返回 (int64 数组, 文件名)

  @dataclass(frozen=True)
  class MixConfig:
      web: float = 0.85
      code: float = 0.15
      fim_rate: float = 0.0          # 只作用于 code 窗口

  class MixedLoader:
      def __init__(self, sets: dict[str, ShardSet], mix: MixConfig, batch_size: int, seq_len: int,
                   seed: int, vocab_size: int, fim_ids: FimIds = FimIds()): ...
          # 校验每个 ShardSet 的 min_tokens ≥ seq_len + 1，否则 ValueError（防止取出越界窗口）
      def get_batch(self, step: int) -> tuple[torch.LongTensor, torch.LongTensor]
          # rng = np.random.default_rng([seed, step])；每个窗口独立以概率 mix.code 选 code 来源；
          # 普通窗口取 seq_len + 1 个 token；被选中做 FIM 的 code 窗口取 seq_len - 3 个 token，
          # 经 apply_fim_psm 后恰好 seq_len + 1；x = tokens[:-1]，y = tokens[1:]，形状 (batch_size, seq_len)；
          # 组 batch 时检查所有值 < vocab_size，否则抛 ValueError 并写明分片文件名
      def sources_in_last_batch(self) -> list[str]   # 供测试与日志：上一个 batch 每行的来源
  ```

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_loader.py
def _two_sets(tmp_path, min_tokens=9):
    a = write_shard(tmp_path / "web_0.bin", list(range(10, 60)))       # 值在 10..59，避开特殊 id 0..3
    b = write_shard(tmp_path / "code_0.bin", list(range(100, 170)))    # 值 ≥ 100
    return {"web": ShardSet([a], [50], min_tokens), "code": ShardSet([b], [70], min_tokens)}

def test_loader_requires_min_tokens_to_cover_window(tmp_path):
    with pytest.raises(ValueError):
        MixedLoader(_two_sets(tmp_path, 9), MixConfig(), batch_size=4, seq_len=16, seed=1, vocab_size=200)

def test_batch_shapes_and_shift(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path), MixConfig(), batch_size=4, seq_len=8, seed=1, vocab_size=200)
    x, y = ld.get_batch(0)
    assert x.shape == y.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])

def test_deterministic_per_step(tmp_path):
    mk = lambda: MixedLoader(_two_sets(tmp_path), MixConfig(), 4, 8, seed=7, vocab_size=200)
    assert torch.equal(mk().get_batch(3)[0], mk().get_batch(3)[0])
    assert not torch.equal(mk().get_batch(3)[0], mk().get_batch(4)[0])

def test_windows_never_cross_shards(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path), MixConfig(web=0.5, code=0.5), 32, 8, seed=0, vocab_size=200)
    for step in range(20):
        x, _ = ld.get_batch(step)
        assert all(((row < 100).all() or (row >= 100).all()).item() for row in x)

def test_mix_ratio_converges(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path), MixConfig(web=0.85, code=0.15), 16, 8, seed=0, vocab_size=200)
    n_code = 0
    for s in range(200):
        ld.get_batch(s)
        n_code += ld.sources_in_last_batch().count("code")
    assert 0.12 <= n_code / (16 * 200) <= 0.18

def test_fim_applied_only_to_code(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path, 17), MixConfig(web=0.0, code=1.0, fim_rate=1.0), 8, 16, seed=0, vocab_size=200)
    x, y = ld.get_batch(0)
    full = torch.cat([x[:, :1], y], dim=1)             # 还原 seq_len + 1 的序列
    assert (full[:, 0] == 1).all() and (full[:, -1] == 0).all()
    ld2 = MixedLoader(_two_sets(tmp_path, 17), MixConfig(web=1.0, code=0.0, fim_rate=1.0), 8, 16, seed=0, vocab_size=200)
    x2, _ = ld2.get_batch(0)
    assert (x2[:, 0] != 1).all()

def test_fim_rate_converges(tmp_path):
    ld = MixedLoader(_two_sets(tmp_path, 17), MixConfig(web=0.0, code=1.0, fim_rate=0.5), 16, 16, seed=0, vocab_size=200)
    n_fim = sum(int((ld.get_batch(s)[0][:, 0] == 1).sum()) for s in range(100))
    assert 0.42 <= n_fim / (16 * 100) <= 0.58

def test_short_shard_excluded_with_warning(tmp_path):
    short = write_shard(tmp_path / "s.bin", list(range(5))); ok = write_shard(tmp_path / "o.bin", list(range(50)))
    with pytest.warns(UserWarning, match="s.bin"):
        ss = ShardSet([short, ok], [5, 50], min_tokens=9)
    assert ss.total_tokens == 50
    with pytest.raises(ValueError):
        ShardSet([short], [5], min_tokens=9)

def test_corrupt_token_value_raises_with_shard_name(tmp_path):
    bad = write_shard(tmp_path / "bad.bin", [1, 2, 40000] + list(range(20)))
    sets = {"web": ShardSet([bad], [23], min_tokens=9), "code": ShardSet([bad], [23], min_tokens=9)}
    ld = MixedLoader(sets, MixConfig(), 8, 8, seed=0, vocab_size=16384)
    with pytest.raises(ValueError, match="bad.bin"):
        for s in range(50): ld.get_batch(s)

def test_from_index(tmp_path):
    f = write_shard(tmp_path / "web" / "train_0000.bin", list(range(30)))
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps({"vocab_size": 16384, "dtype": "uint16", "eot_id": 0, "sources": {
        "web": {"train": [{"file": "train_0000.bin", "tokens": 30}], "val": []}}}), encoding="utf-8")
    assert ShardSet.from_index(idx, "web", "train", min_tokens=9).total_tokens == 30
```

- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现 `loader.py`**

`from_index` 中文件路径 = `index.json` 所在目录 / source / file。

- [ ] **Step 4: 运行，确认通过** → 10 passed；全量 `pytest -q` 通过
- [ ] **Step 5: Commit**：`git commit -m "feat(data): memmap shard sets and deterministic mixed loader with FIM"`

---

### Task 6: 真实数据跑通并出报告

**Files:**
- Create: `docs/data/report.md`
- 产出（不进 git）：`data/raw/`、`data/tokenizer/heimi-16k.json`、`data/tokens/`

**Interfaces:**
- Consumes：Task 1 到 5 的全部 CLI 与类。

- [ ] **Step 1: 下载**（后台运行，约 20 GB，视网速 1 到 3 小时）

```bash
.venv/Scripts/python.exe -m minids.data.download --source all --root data
```
Expected: 13 个文件全部"校验通过"，`data/raw/web` 8 个、`data/raw/code` 5 个。

- [ ] **Step 2: 算验证集行数，训练分词器**

```bash
.venv/Scripts/python.exe -m minids.data.prepare val-rows --root data --out data/tokens/val_rows.json
.venv/Scripts/python.exe -m minids.data.tokenizer train --root data --out data/tokenizer/heimi-16k.json --val-rows data/tokens/val_rows.json
```
Expected: `val_rows.json` 里 web 与 code 各为一个正整数；词表 16384；训练语料约 2 GB；耗时几分钟到十几分钟。

- [ ] **Step 3: 预处理**（后台运行，约 1 到 2 小时）

```bash
.venv/Scripts/python.exe -m minids.data.prepare run --root data --tokenizer data/tokenizer/heimi-16k.json --val-rows data/tokens/val_rows.json --source all
```
Expected: `data/tokens/index.json` 存在；web 约 60 片、code 约 10 片；每个来源 `tokens_val` 在 1500 万到 3000 万之间（字节估算的误差范围），不在这个范围就调整 `--target-bytes` 重跑 val-rows 和本步。

- [ ] **Step 4: 抽样检查**

用 `ShardSet.from_index` 与 `MixedLoader` 取一个 batch，`HeimiTokenizer.decode` 解码 3 行 web、3 行 code，肉眼确认是通顺英文和合法代码，且 `<|endoftext|>` 出现在文档边界。再用 `MixConfig(web=0, code=1, fim_rate=1)` 取一个 batch 解码一行，确认 PSM 标记顺序正确。

- [ ] **Step 5: 写 `docs/data/report.md`**

内容：每个来源的文档数、跳过数、训练与验证 token 数、分片数、每 token 平均字节数（原始文本字节 ÷ token 数，分 web 与 code）、磁盘占用、各步骤耗时、Step 4 的解码样例各一条（截断到 300 字符）。末尾一段：与 spec 第 6 节目标（web 约 5.5B 以上、code 约 1B）的对照，不足时说明差多少以及是否需要多下载分片。

- [ ] **Step 6: Commit**

```bash
git add docs/data/report.md
git commit -m "docs(data): real-data pipeline run report"
```

---

## 完成标准

- `.venv/Scripts/python.exe -m pytest -q` 全部通过（计划一的 52 个加本计划新增的约 29 个）。
- `data/tokens/index.json` 存在，web 与 code 各有 ≥ 2000 万 token 的验证集和与报告一致的训练分片。
- `MixedLoader.get_batch(step)` 在真实分片上返回形状正确、内容可解码、同 step 可复现的 batch。
- `docs/data/report.md` 给出真实统计与 spec 目标的对照。
- 计划三（训练循环）可以在此基础上开始。
