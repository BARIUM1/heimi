# 黑米 计划四：窗口（Dashboard）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做一个属于黑米的本地网页窗口：训练进行中就能看到成长曲线、每次增长的效果、当前状态、各阶段对同一提示的续写，并能选任意检查点亲手试玩。

**Architecture:** `dashboard/data.py` 只读 `runs/<run>/` 下的四个 jsonl 与检查点目录，把它们整理成前端要的序列（去重后的评测点、带标注的增长事件、参数量阶梯、训练状态与预计剩余）；`dashboard/worker.py` 是独立子进程里的生成器，按需加载检查点、最多缓存 2 个模型、超时即被杀掉重启；`dashboard/server.py` 是薄薄一层 FastAPI，校验名字后转给前两者；前端是一个内联 CSS 与 JS 的 `index.html`，每 5 秒轮询，图表用随仓库分发的 Chart.js。窗口从不写 `runs/`，训练不知道窗口存在。

**Tech Stack:** Python 3.12、FastAPI 0.162、uvicorn 0.54、pydantic 2、httpx（测试用 TestClient）、multiprocessing（spawn）、Chart.js 4.6.1（本地文件）、原生 HTML/CSS/JS、pytest。

**Spec:** `docs/superpowers/specs/2026-11-08-heimi-growth-lm-design.md`（本计划实现第 12 节全部内容与第 13.3 节窗口测试；成功标准 1.3 第 4 条"窗口能在训练进行中实时展示四个区域的内容，且训练不依赖窗口"在此落地）

## 本计划在整体中的位置

计划一到三（已合并）交付了模型、数据和训练流水线。`runs/heimi/` 里已有级 0 的完整日志、级 1 前 20 步、三个检查点，足够把窗口做出来并用真实数据验证。本计划完成后，用户在夜间训练级 1、级 2、任务阶段和对照组时，开着窗口就能看到曲线生长；全部训完后，四个区域展示完整的成长过程和与对照组的比较。

依赖计划三的接口：`minids.logging_io.read_jsonl(path, since_global_step)` 与四个文件名常量、`minids.train.checkpoint.load_model(path, device) -> (Transformer, meta)`（`meta["ladder_config"]["data"]["tokenizer"]` 是分词器路径，`meta["state"]` 是训练状态）、`minids.completion.complete_middle`、`Transformer.generate`、`minids.gpu`。

## 已核实的事实（2026-11-09）

- 依赖可装：`uv pip install --dry-run fastapi uvicorn httpx` 解析为 fastapi 0.162.2、uvicorn 0.54.0、httpx 0.28.1、pydantic 2.14.3、starlette 1.10.0。
- Chart.js 最新 4.x 为 4.6.1（2026-10-22 发布，MIT）。`https://cdn.jsdelivr.net/npm/chart.js@4.6.1/dist/chart.umd.min.js` 大小 210,082 字节，sha256 `87364cc2b2a76a6684929e7d6b80b18a897794db38400caff1fcf1791c6925d2`，文件头为 `/*!\n * Chart.js v4.6.1 ... Released under the MIT License`。下载一次放进仓库，页面不访问外网。
- 真实日志的形状：`stage_start` 事件的 data 有 `params_non_embedding`、`steps`、`growth`；`growth` 事件有 `kind`、`params_before`、`params_after`；`samples.jsonl` 里 FIM 提示的空位用 `<FILL>` 标出；所有 `time` 带时区偏移（本机为 `-05:00`）。
- `evaluate.py` 会在同一个 `global_step` 追加第二条评测记录（计划三终审的裁定）：窗口必须按 (stage, global_step) 去重，留最后一条。
- 级末评测最长约 20 分钟，期间 `metrics.jsonl` 不增长：只看"最后一条指标的时间"会把评测误判成停止，所以状态分三档。
- 仓库根目录同时有包 `dashboard/` 和脚本 `dashboard.py`（spec 第 4 节的布局）：`import dashboard` 得到的是包，所以命令行逻辑放在 `dashboard/cli.py`，根目录脚本只是三行转发。
- 8787 端口当前空闲；`runs/` 下有 `heimi`、`smoke`、`smoke2`、`smoke3` 四个 run。

## Global Constraints

- 服务只绑 `127.0.0.1`，默认端口 8787，启动后自动打开浏览器（spec 12.1）；不提供改绑其他地址的选项。
- 单页应用，界面文字全中文，每 5 秒拉取一次最新数据，页面标题与顶栏显示模型名字"黑米 Heimi"（spec 12.1）。
- 窗口只读文件，从不在 `runs/` 下创建、修改或删除任何东西；训练进程与窗口无共享状态（spec 第 2 节、12.3）。
- 后端接口（spec 12.2）：`GET /api/runs`、`GET /api/runs/{run}/metrics?since=<global_step>`、`GET /api/runs/{run}/eval`、`/events`、`/samples`、`GET /api/status`、`POST /api/generate`，请求体 `{run, checkpoint, prefix, suffix?, max_new_tokens, temperature, device}`，返回生成文本与耗时。本计划另加一个 `GET /api/runs/{run}/overview`，把去重、增长标注、参数量阶梯放在有测试的后端做。
- 生成工作进程（spec 12.3）：独立子进程，按需加载检查点，最多缓存 2 个模型，默认 CPU，请求可指定 `cuda`，超时 60 秒返回错误。
- 四个区域（spec 12.4）：成长总览、当前状态、成长样本墙、试玩台，内容按 spec 逐条实现；成长总览另加一条训练损失细线（评测每 500 步才一个点，级 0 全程只有 2 个点，没有它曲线太稀），这是本计划在 spec 之外的添加。
- 前端（spec 12.5）：一个 `index.html`，内联 CSS 与 JS，Chart.js 从本地 `static/` 加载，不依赖外网；不用框架、不用构建步骤。
- 模型产出的任何文本（续写、提示、报错）只能以文本节点写进页面，不得作为 HTML 解析。
- 单元测试在 CPU 上运行、不访问网络、不读 `runs/` 与 `data/` 下的真实文件；只有 Task 6 的实机验证用真实 `runs/heimi`。注释中文、标识符英文。
- 每个 commit 信息用英文 conventional commits 格式，末尾加一行 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`。

## Review Focus

1. URL 或请求体里的 run 名、检查点名带 `..`、斜杠、盘符：必须得到 404，绝不能读到 `runs/` 之外的文件，也不能把这样的路径交给生成进程。测试在 Task 1 与 Task 4。
2. 训练进程边写窗口边读：同一步出现两条评测记录（级末评测加 `evaluate.py` 重算）时曲线上只能有一个点；半行由 `read_jsonl` 处理。测试在 Task 1。
3. 还没有任何 run、run 目录是空的、或某一级刚开始只有事件没有指标：接口返回空列表与空状态，页面显示"还没有训练记录"，而不是 500 或一屏 JS 报错。测试在 Task 1、Task 2、Task 6，页面空状态在 Task 5 实看。
4. 生成卡住（105M 模型在 CPU 上配长上下文可能超过 60 秒）：返回超时错误，并且卡住的进程被杀掉，下一次请求能正常工作，而不是永远排在一个死进程后面。测试在 Task 3。
5. 模型补出 `<script>`、`<b>` 之类的代码文本：页面按原文显示，不被当成标记执行。测试在 Task 5（静态检查加实看）。

---

## 文件结构

```
pyproject.toml                    # 修改：依赖加 fastapi、uvicorn，dev 加 httpx，包发现加 dashboard*
.gitignore                        # 修改：加 /.claude/
minids/gpu.py                     # 修改：加 gpu_status()
dashboard/__init__.py
dashboard/data.py                 # 只读 run 目录：列 run、列检查点、成长总览序列、训练状态与预计剩余
dashboard/worker.py               # 生成：ModelCache、generate_text、worker_main、WorkerClient
dashboard/server.py               # FastAPI：create_app(runs_dir, worker, repo_root)
dashboard/cli.py                  # 命令行：build(argv)、main(argv)
dashboard/static/index.html       # 单页前端
dashboard/static/chart.umd.min.js # Chart.js 4.6.1（MIT，随仓库分发）
dashboard.py                      # 入口：转发到 dashboard.cli.main
docs/dashboard.md                 # 用法
tests/conftest.py                 # 修改：加会话级夹具 dash_runs
tests/helpers_dashboard.py        # sleepy_worker、FakeWorker
tests/test_dashboard_data.py  test_dashboard_status.py  test_dashboard_worker.py
tests/test_dashboard_server.py  test_dashboard_frontend.py  test_dashboard_cli.py
tests/test_gpu.py                 # 修改：加 gpu_status 的测试
```

与 spec 第 4 节的差异：新增 `dashboard/data.py`（把读文件与整理数据从 `server.py` 里分出来，不经 HTTP 就能测）和 `dashboard/cli.py`（原因见"已核实的事实"）。

---

### Task 1: 依赖与 run 数据读取

**Files:**
- Modify: `pyproject.toml`（dependencies 加 `fastapi>=0.110`、`uvicorn>=0.29`；dev 加 `httpx>=0.27`；`[tool.setuptools.packages.find] include` 改为 `["minids*", "dashboard*"]`）、`tests/conftest.py`
- Create: `dashboard/__init__.py`、`dashboard/data.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes：`read_jsonl`、`METRICS_FILE` 等常量、`RunLogger`、`EvalRecord`；`tiny_ladder`、`run_ladder`（建夹具）
- Produces（`tests/conftest.py`）：
  ```python
  @pytest.fixture(scope="session")
  def dash_runs(tmp_path_factory) -> tuple[Path, LadderConfig]:
      """一个真实的迷你 run：runs_dir/heimi，两级（smoke0 两步、smoke1 加深后两步），日志与检查点齐全。只读，测试不得修改它。"""
      # root = tmp_path_factory.mktemp("dash"); cfg = tiny_ladder(root, stage_steps=(2, 2)); run_ladder(cfg, root / "runs" / "heimi", "cpu")
      # 返回 (root / "runs", cfg)；检查点有 smoke0_step2.pt、smoke0_final.pt、smoke1_step2.pt、smoke1_final.pt
  ```
- Produces（`dashboard/data.py`，本任务的部分）：
  ```python
  RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

  class RunNotFound(LookupError): ...

  def safe_run_dir(runs_dir: Path, name: str) -> Path
      # 名字不匹配 RUN_NAME_RE、含 ".."、目录不存在、或解析后不在 runs_dir 之下 → RunNotFound
  def list_checkpoints(run_dir: Path) -> list[dict]
      # run_dir/ckpt 下形如 <stage>_final.pt / <stage>_step<N>.pt 的文件：
      # {"name", "stage", "kind": "final" | "step", "step": int | None, "size_mb": float, "mtime": iso 字符串}
      # final 排在前面，各自按 mtime 新到旧；.tmp、.corrupt 与其他文件忽略；目录不存在 → []
  def safe_checkpoint(run_dir: Path, name: str) -> Path
      # 只接受 list_checkpoints 里出现的名字，否则 RunNotFound
  def format_params(n: int) -> str
      # ≥ 100 万按四舍五入到 M（12_934_528 → "13M"），否则到 K（75_136 → "75K"）
  def run_summary(run_dir: Path) -> dict
      # {"name": 目录名,
      #  "stages": [{"name", "params", "steps", "growth", "done": bool}],   # 每条 stage_start 一项，done = 有对应的 stage_end
      #  "stage": 最新一条指标（没有则最新事件）所在的级 | None, "global_step": int, "tokens_total": int,
      #  "last_update": 四个文件各自最后一条记录的 time 里最新的 | None, "checkpoints": list_checkpoints(run_dir)}
  def list_runs(runs_dir: Path) -> list[dict]
      # runs_dir 下名字合法且至少有一个 jsonl 的子目录各一个 run_summary，按 last_update 新到旧（None 排最后）；runs_dir 不存在 → []
  def overview(run_dir: Path) -> dict
      # {"eval":   [{"stage","global_step","tokens_total","val_loss_web","val_loss_code","hellaswag_acc","fim_em"}],  # 同 (stage, global_step) 只留最后一条，按 tokens_total 升序
      #  "growth": [{"tokens_total","kind","params_before","params_after","label"}],   # label = f"{format_params(before)} → {format_params(after)} {'加深' if kind == 'depth' else '加宽'}"
      #  "params": [{"tokens_total","params"}]}   # 每条 stage_start 一点；末尾补一点（该 run 见过的最大 tokens_total，沿用最后的 params），已在末尾则不补
  ```

- [ ] **Step 1: 装依赖、改 pyproject、加夹具，写失败测试**

Run: `uv pip install --python .venv/Scripts/python.exe fastapi uvicorn httpx`

```python
# tests/test_dashboard_data.py
import pytest

from dashboard.data import RunNotFound, format_params, list_checkpoints, list_runs, overview, run_summary, safe_checkpoint, safe_run_dir
from minids.logging_io import EvalRecord, RunLogger, now_iso


def test_format_params():
    assert [format_params(n) for n in (12_934_528, 25_868_672, 103_248_640, 75_136, 595_328)] == ["13M", "26M", "103M", "75K", "595K"]


def test_safe_run_dir_rejects_traversal_and_unknown(tmp_path):
    runs = tmp_path / "runs"
    (runs / "heimi").mkdir(parents=True)
    (tmp_path / "secret").mkdir()
    assert safe_run_dir(runs, "heimi") == runs / "heimi"
    for bad in ("..", "../secret", "heimi/../../secret", "", ".", "nope", "a b", "heimi\\..\\..\\secret", "C:secret"):
        with pytest.raises(RunNotFound):
            safe_run_dir(runs, bad)


def test_list_runs_and_summary_on_real_tiny_run(dash_runs):
    runs_dir, _ = dash_runs
    runs = list_runs(runs_dir)
    assert [r["name"] for r in runs] == ["heimi"]
    r = runs[0]
    assert [s["name"] for s in r["stages"]] == ["smoke0", "smoke1"] and all(s["done"] for s in r["stages"])
    assert r["stages"][1]["growth"] == "depth" and r["stages"][0]["steps"] == 2 and r["stages"][1]["params"] > r["stages"][0]["params"]
    assert r["stage"] == "smoke1" and r["global_step"] == 4 and r["tokens_total"] == 4 * 256 and r["last_update"]
    names = [c["name"] for c in r["checkpoints"]]
    assert set(names) == {"smoke0_final.pt", "smoke1_final.pt", "smoke0_step2.pt", "smoke1_step2.pt"}
    assert all(n.endswith("_final.pt") for n in names[:2])


def test_list_checkpoints_ignores_partials_and_parses_names(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    for n in ("s0_step500.pt", "s0_final.pt", "s1_step20.pt", "s1_step40.tmp", "s0_step9.pt.corrupt", "notes.txt"):
        (d / n).write_bytes(b"x")
    out = list_checkpoints(tmp_path)
    assert out[0]["name"] == "s0_final.pt" and {c["name"] for c in out} == {"s0_final.pt", "s0_step500.pt", "s1_step20.pt"}
    by = {c["name"]: c for c in out}
    assert (by["s0_final.pt"]["kind"], by["s0_final.pt"]["step"], by["s0_final.pt"]["stage"]) == ("final", None, "s0")
    assert (by["s0_step500.pt"]["kind"], by["s0_step500.pt"]["step"], by["s0_step500.pt"]["stage"]) == ("step", 500, "s0")
    assert list_checkpoints(tmp_path / "no-such-run") == []


def test_safe_checkpoint_only_accepts_listed_names(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "s0_final.pt").write_bytes(b"x")
    (d / "s0_step9.pt.corrupt").write_bytes(b"x")
    (tmp_path / "outside.pt").write_bytes(b"x")
    assert safe_checkpoint(tmp_path, "s0_final.pt") == d / "s0_final.pt"
    for bad in ("../outside.pt", "s0_step9.pt.corrupt", "nope.pt", ""):
        with pytest.raises(RunNotFound):
            safe_checkpoint(tmp_path, bad)


def test_overview_dedupes_eval_and_labels_growth(tmp_path):
    log = RunLogger(tmp_path)
    log.event("stage_start", "s0", 0, 0, params_non_embedding=75_136, steps=3, accum_steps=2, micro_batch=4, growth="none")
    log.eval(EvalRecord("s0", 3, 768, 5.0, 4.0, None, None, now_iso()))
    log.eval(EvalRecord("s0", 3, 768, 4.9, 3.9, 0.25, 0.0, now_iso()))  # 同一步的第二条：evaluate.py 重算
    log.event("stage_end", "s0", 3, 768, loss_last=4.8, params_non_embedding=75_136, seconds=1.0)
    log.event("growth", "s1", 3, 768, kind="depth", params_before=75_136, params_after=150_208,
              exact_report={"exact": True}, noisy_report=None, seconds=0.1)
    log.event("stage_start", "s1", 3, 768, params_non_embedding=150_208, steps=2, accum_steps=2, micro_batch=4, growth="depth")
    log.eval(EvalRecord("s1", 5, 1280, 4.5, 3.5, None, None, now_iso()))
    o = overview(tmp_path)
    assert [(e["global_step"], e["val_loss_web"], e["hellaswag_acc"]) for e in o["eval"]] == [(3, 4.9, 0.25), (5, 4.5, None)]
    assert o["growth"] == [{"tokens_total": 768, "kind": "depth", "params_before": 75_136, "params_after": 150_208, "label": "75K → 150K 加深"}]
    assert [(p["tokens_total"], p["params"]) for p in o["params"]] == [(0, 75_136), (768, 150_208), (1280, 150_208)]


def test_empty_or_missing_run_gives_empty_results(tmp_path):
    assert overview(tmp_path) == {"eval": [], "growth": [], "params": []}
    s = run_summary(tmp_path)
    assert s["stages"] == [] and s["stage"] is None and s["global_step"] == 0 and s["last_update"] is None and s["checkpoints"] == []
    assert list_runs(tmp_path / "does-not-exist") == []
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_data.py -q`
Expected: 全部 FAIL，`ModuleNotFoundError: dashboard`

- [ ] **Step 3: 实现 `dashboard/__init__.py`（空）与 `dashboard/data.py` 的上述函数**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_data.py -q`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml dashboard/__init__.py dashboard/data.py tests/conftest.py tests/test_dashboard_data.py
git commit -m "feat(dashboard): read-only run data: listing, safe paths, growth overview series"
```

---

### Task 2: 训练状态与预计剩余

**Files:**
- Modify: `minids/gpu.py`、`dashboard/data.py`、`tests/test_gpu.py`
- Test: `tests/test_dashboard_status.py`

**Interfaces:**
- Consumes：Task 1 的 `list_runs`、`run_summary`；`read_jsonl`
- Produces（`minids/gpu.py`）：
  ```python
  def gpu_status(timeout_s: float = 2.0) -> dict | None
      # nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 的第一行
      # → {"temp_c": float, "mem_used_gb": MiB / 1024, "mem_total_gb": MiB / 1024, "util_pct": float}；任何失败 → None
  ```
- Produces（`dashboard/data.py`，在文件顶部 `from minids.gpu import gpu_status`，测试会替换这个名字）：
  ```python
  RUNNING_WITHIN_S = 180.0   # 最后一条记录在这么多秒内 → 训练中
  QUIET_WITHIN_S = 1800.0    # 超过上面但在这之内 → 可能在做评测（级末评测最长约 20 分钟）

  def training_state(last_update: datetime | None, now: datetime) -> str   # "running" | "quiet" | "stopped"
  def eta_seconds(metrics: list[dict], events: list[dict]) -> float | None
      # 当前级 = 最后一条指标的 stage；steps 取该级最新一条 stage_start 的 data.steps；
      # 每步 token = 最后一条指标的 tokens_stage / step；剩余 token = (steps - step) × 每步 token；
      # 吞吐 = 该级最近 10 条指标 tok_per_s 的中位数（每 10 步一条，即 spec 12.2 说的最近 100 步；评测后的那一条会偏低，中位数不受影响）；
      # 没有指标或找不到 stage_start → None；本级已跑完 → 0.0
  def status(runs_dir: Path, now: datetime | None = None) -> dict
      # 取 last_update 最新的 run。键：
      # "run", "state", "running"（= state == "running"）, "stage", "step", "steps", "global_step", "tokens_total",
      # "loss", "lr", "tok_per_s", "mem_gb", "gpu_temp_c"（以上取自最后一条指标，没有则 None）,
      # "eta_seconds", "last_update", "seconds_since_update", "last_eval"（最后一条评测记录或 None）, "gpu"（gpu_status()）
      # 没有任何 run → run 为 None、state "stopped"、running False，其余数值键为 None，gpu 照常
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dashboard_status.py
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from dashboard import data
from dashboard.data import eta_seconds, status, training_state
from minids.logging_io import EVENTS_FILE, EvalRecord, EventRecord, MetricsRecord, RunLogger, append_jsonl

# 所有记录的时间都由测试给定，不用 RunLogger.event（它盖的是真实的当前时间）
T0 = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone(timedelta(hours=-5)))


def _metric(stage, step, tok_per_s, when, tokens_per_step=256):
    return MetricsRecord(stage, step, step, step * tokens_per_step, step * tokens_per_step, 4.0, 1e-3, 0.5,
                         tok_per_s, 8.0, 55.0, when.isoformat(timespec="seconds"))


def test_training_state_thresholds():
    assert training_state(T0, T0 + timedelta(seconds=60)) == "running"
    assert training_state(T0, T0 + timedelta(seconds=600)) == "quiet"   # 级末评测期间
    assert training_state(T0, T0 + timedelta(seconds=3600)) == "stopped"
    assert training_state(None, T0) == "stopped"


def test_eta_uses_median_throughput_and_remaining_steps():
    events = [{"type": "stage_start", "stage": "s0", "data": {"steps": 100}}]
    metrics = [asdict(_metric("s0", s, tps, T0)) for s, tps in ((10, 1000.0), (20, 1000.0), (30, 10.0), (40, 1000.0))]
    assert eta_seconds(metrics, events) == pytest.approx(60 * 256 / 1000.0)  # 第 30 步那条被评测拖慢，不影响中位数
    assert eta_seconds([], events) is None and eta_seconds(metrics, []) is None


def test_eta_is_zero_when_stage_finished():
    events = [{"type": "stage_start", "stage": "s0", "data": {"steps": 40}}]
    assert eta_seconds([asdict(_metric("s0", 40, 1000.0, T0))], events) == 0.0


def test_status_picks_latest_run_and_reports_state(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "gpu_status", lambda: {"temp_c": 50.0, "mem_used_gb": 3.0, "mem_total_gb": 16.0, "util_pct": 5.0})
    old, new = RunLogger(tmp_path / "old"), RunLogger(tmp_path / "heimi")
    old.metrics(_metric("s0", 10, 500.0, T0 - timedelta(hours=5)))
    append_jsonl(new.path(EVENTS_FILE), EventRecord(
        "stage_start", "stage1", 0, 0, (T0 - timedelta(minutes=5)).isoformat(timespec="seconds"),
        {"params_non_embedding": 1, "steps": 100, "accum_steps": 2, "micro_batch": 4, "growth": "none"}))
    new.metrics(_metric("stage1", 10, 1000.0, T0))
    new.eval(EvalRecord("stage1", 10, 2560, 4.2, 3.3, None, None, T0.isoformat(timespec="seconds")))
    s = status(tmp_path, now=T0 + timedelta(seconds=30))
    assert s["run"] == "heimi" and s["state"] == "running" and s["running"] is True
    assert (s["stage"], s["step"], s["steps"], s["tok_per_s"], s["loss"]) == ("stage1", 10, 100, 1000.0, 4.0)
    assert s["eta_seconds"] == pytest.approx(90 * 256 / 1000.0) and s["last_eval"]["val_loss_web"] == 4.2
    assert s["gpu"]["temp_c"] == 50.0 and s["seconds_since_update"] == pytest.approx(30.0, abs=1.0)
    later = status(tmp_path, now=T0 + timedelta(hours=2))
    assert later["state"] == "stopped" and later["running"] is False


def test_status_with_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "gpu_status", lambda: None)
    s = status(tmp_path / "missing")
    assert s["run"] is None and s["state"] == "stopped" and s["running"] is False
    assert s["gpu"] is None and s["eta_seconds"] is None and s["stage"] is None and s["last_eval"] is None
```

```python
# tests/test_gpu.py（追加）
import pytest


def test_gpu_status_parses_all_fields(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="55, 3493, 16303, 7\n", stderr=""))
    assert gpu.gpu_status() == {"temp_c": 55.0, "mem_used_gb": pytest.approx(3493 / 1024), "mem_total_gb": pytest.approx(16303 / 1024), "util_pct": 7.0}


def test_gpu_status_none_on_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.gpu_status() is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_status.py tests/test_gpu.py -q`
Expected: 新增的 7 个 FAIL（`ImportError: eta_seconds` 与 `AttributeError: gpu_status`），`test_gpu.py` 原有 3 个通过

- [ ] **Step 3: 实现 `gpu_status`、`training_state`、`eta_seconds`、`status`**

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_status.py tests/test_gpu.py -q`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add minids/gpu.py dashboard/data.py tests/test_gpu.py tests/test_dashboard_status.py
git commit -m "feat(dashboard): training state, ETA from median throughput, GPU status probe"
```

---

### Task 3: 生成工作进程

**Files:**
- Create: `dashboard/worker.py`、`tests/helpers_dashboard.py`
- Test: `tests/test_dashboard_worker.py`

**Interfaces:**
- Consumes：`load_model`、`HeimiTokenizer.load`、`complete_middle`、`Transformer.generate`；夹具 `dash_runs`
- Produces（`tests/helpers_dashboard.py`）：
  ```python
  def sleepy_worker(conn, repo_root) -> None      # time.sleep(30)：一个永远不回话的工作进程，用来测超时（必须是模块级函数，spawn 要能导入它）
  class FakeWorker:                               # Task 4 用
      def __init__(self, result: dict | None = None, exc: Exception | None = None)   # result 默认 {"text": " ok", "seconds": 0.01, "tokens": 1}
      calls: list[dict]; closed: bool
      def generate(self, req: dict) -> dict        # 记录 req；有 exc 就抛
      def close(self) -> None
  ```
- Produces（`dashboard/worker.py`）：
  ```python
  MAX_CACHED_MODELS = 2
  DEFAULT_TIMEOUT_S = 60.0

  class WorkerTimeout(RuntimeError): ...
  class WorkerError(RuntimeError): ...

  class ModelCache:
      def __init__(self, max_models: int = MAX_CACHED_MODELS, repo_root: Path | None = None)
      def get(self, ckpt_path: str, device: str) -> tuple[Transformer, HeimiTokenizer]
          # 键 (绝对路径, st_mtime_ns, device)：文件被重写后自动重新加载；命中则标为最近使用；
          # 未命中：load_model(path, device)，分词器取检查点 meta["ladder_config"]["data"]["tokenizer"]
          # （相对路径按 repo_root 解析，缺省 repo_root 为当前目录）；超过 max_models 丢掉最久未用的
      def __len__(self) -> int

  def generate_text(cache: ModelCache, req: dict) -> dict
      # req 键：ckpt_path、prefix、suffix（str 或 None）、max_new_tokens、temperature、device
      # suffix 非 None：complete_middle 贪心补一行，预算 max_prefix = min(512, L // 2)、max_suffix = min(256, L // 4)、
      #   max_new = min(请求值, L - max_prefix - max_suffix - 3)，L = model.cfg.seq_len
      # suffix 为 None：提示编码后只留最后 L - max_new 个 token，generate(temperature, top_k=50)，text 为新生成部分的解码
      # cuda 上用 bf16 autocast。返回 {"text": str, "tokens": 新生成的 token 数, "seconds": float}

  def worker_main(conn, repo_root: str | None) -> None
      # 子进程入口：建一个 ModelCache；循环 conn.recv()，收到 None 退出；
      # 否则发回 {"ok": True, **generate_text(...)} 或 {"ok": False, "error": f"{类型名}: {消息}"}

  class WorkerClient:
      def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S, repo_root: Path | None = None, target=worker_main)
      def generate(self, req: dict) -> dict
          # 一把锁保证一次只跑一个请求（计时从拿到锁开始）；进程未启动或已死就用 multiprocessing 的 spawn 上下文启动；
          # 发送后 conn.poll(timeout_s)：超时 → 终止进程、丢弃管道、抛 WorkerTimeout（消息含秒数）；
          # 收到 {"ok": False} → WorkerError(error)；否则返回去掉 "ok" 的字典
      def is_alive(self) -> bool
      def close(self) -> None       # 发 None 请它退出，1 秒内不退就 terminate；可重复调用
  ```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dashboard_worker.py
import pytest

from dashboard.worker import ModelCache, WorkerClient, WorkerError, WorkerTimeout, generate_text
from tests.helpers_dashboard import sleepy_worker


def _req(ckpt, **over):
    r = {"ckpt_path": str(ckpt), "prefix": "The cat", "suffix": None, "max_new_tokens": 4, "temperature": 0.0, "device": "cpu"}
    r.update(over)
    return r


def test_cache_reuses_and_evicts_least_recently_used(dash_runs):
    ck = dash_runs[0] / "heimi" / "ckpt"
    cache = ModelCache(max_models=2)
    a1, _ = cache.get(str(ck / "smoke0_final.pt"), "cpu")
    a2, _ = cache.get(str(ck / "smoke0_final.pt"), "cpu")
    assert a1 is a2 and len(cache) == 1
    b1, _ = cache.get(str(ck / "smoke1_final.pt"), "cpu")
    cache.get(str(ck / "smoke0_final.pt"), "cpu")  # 刷新 smoke0 的使用时间
    cache.get(str(ck / "smoke0_step2.pt"), "cpu")  # 第三个：挤掉最久未用的 smoke1
    assert len(cache) == 2
    assert cache.get(str(ck / "smoke0_final.pt"), "cpu")[0] is a1
    assert cache.get(str(ck / "smoke1_final.pt"), "cpu")[0] is not b1  # 被挤掉过，重新加载


def test_generate_plain_is_reproducible_at_temperature_zero_and_fim_is_one_line(dash_runs):
    ckpt = dash_runs[0] / "heimi" / "ckpt" / "smoke1_final.pt"
    cache = ModelCache()
    out = generate_text(cache, _req(ckpt))
    assert isinstance(out["text"], str) and 0 <= out["tokens"] <= 4 and out["seconds"] >= 0
    assert generate_text(cache, _req(ckpt))["text"] == out["text"]
    fim = generate_text(cache, _req(ckpt, prefix="def f():\n    ", suffix="\n"))
    assert "\n" not in fim["text"]


def test_long_prompt_is_cropped_not_rejected(dash_runs):
    ckpt = dash_runs[0] / "heimi" / "ckpt" / "smoke0_final.pt"
    out = generate_text(ModelCache(), _req(ckpt, prefix="word " * 500, max_new_tokens=3))  # 远超 seq_len 32
    assert out["tokens"] <= 3


def test_subprocess_round_trip_and_error_reporting(dash_runs):
    ck = dash_runs[0] / "heimi" / "ckpt"
    client = WorkerClient(timeout_s=60.0)
    try:
        assert isinstance(client.generate(_req(ck / "smoke0_final.pt"))["text"], str)
        with pytest.raises(WorkerError, match="missing.pt"):
            client.generate(_req(ck / "missing.pt"))
        assert client.is_alive()  # 出错不等于进程死掉
        assert isinstance(client.generate(_req(ck / "smoke0_final.pt"))["text"], str)
    finally:
        client.close()
    assert not client.is_alive()


def test_timeout_kills_the_stuck_worker():
    client = WorkerClient(timeout_s=1.0, target=sleepy_worker)
    try:
        with pytest.raises(WorkerTimeout, match="1"):
            client.generate(_req("whatever.pt"))
        assert not client.is_alive()  # 卡住的进程已被杀掉，下一次请求会起一个新的
    finally:
        client.close()
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_worker.py -q`
Expected: FAIL，`ModuleNotFoundError: dashboard.worker`

- [ ] **Step 3: 实现 `tests/helpers_dashboard.py` 与 `dashboard/worker.py`**

`multiprocessing.get_context("spawn")`，`Pipe()` 双向；子进程 `daemon=True`，窗口退出时不留孤儿。`worker_main` 里捕获 `Exception` 逐条回报，不让一次失败的请求带走进程。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_worker.py -q`
Expected: 5 passed（含一次真实子进程启动，约 10 秒）

- [ ] **Step 5: 提交**

```bash
git add dashboard/worker.py tests/helpers_dashboard.py tests/test_dashboard_worker.py
git commit -m "feat(dashboard): generation worker subprocess with a two-model cache and kill-on-timeout"
```

---

### Task 4: 后端接口

**Files:**
- Create: `dashboard/server.py`
- Test: `tests/test_dashboard_server.py`

**Interfaces:**
- Consumes：Task 1、2 的 `data` 函数与 `RunNotFound`；Task 3 的 `WorkerClient`、`WorkerTimeout`、`WorkerError`、`FakeWorker`；`read_jsonl` 与文件名常量
- Produces：
  ```python
  class GenerateRequest(BaseModel):
      run: str
      checkpoint: str
      prefix: str = Field(min_length=1, max_length=20000)
      suffix: str | None = Field(default=None, max_length=20000)
      max_new_tokens: int = Field(default=64, ge=1, le=256)
      temperature: float = Field(default=0.8, ge=0.0, le=2.0)
      device: Literal["cpu", "cuda"] = "cpu"

  def create_app(runs_dir: Path, worker=None, repo_root: Path | None = None) -> FastAPI
      # worker 为 None 时建一个 WorkerClient(repo_root=repo_root)（进程在第一次生成时才启动）；应用关闭时调用 worker.close()
  ```
  路由：`GET /` 返回 `dashboard/static/index.html`（文件不存在时返回一句"前端尚未构建"的 200 纯文本，Task 5 之前测试可用）；`/static` 挂载静态目录；
  `GET /api/runs` → `list_runs`；`GET /api/runs/{run}/metrics`（可选 `since: int`）→ `read_jsonl(metrics, since)`；`/eval`、`/events`、`/samples` → 全量；`/overview` → `overview`；
  `GET /api/status` → `status(runs_dir)`；
  `POST /api/generate`：`safe_run_dir` 与 `safe_checkpoint` 校验后，把 `{"ckpt_path": str(绝对路径), "prefix", "suffix", "max_new_tokens", "temperature", "device"}` 交给 `worker.generate`，返回 `{"text", "seconds", "tokens", "checkpoint", "device"}`。
  错误映射：`RunNotFound` → 404；`WorkerTimeout` → 504；`WorkerError` → 500；`detail` 是中文说明加原始消息。所有处理函数用普通 `def`（FastAPI 会放进线程池，生成不会堵住轮询）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dashboard_server.py
import pytest
from fastapi.testclient import TestClient

from dashboard.server import create_app
from dashboard.worker import WorkerClient, WorkerError, WorkerTimeout
from tests.helpers_dashboard import FakeWorker


@pytest.fixture
def client(dash_runs):
    worker = FakeWorker()
    with TestClient(create_app(dash_runs[0], worker=worker)) as c:
        c.worker = worker
        c.runs_dir = dash_runs[0]
        yield c


def test_runs_listing(client):
    r = client.get("/api/runs").json()
    assert r[0]["name"] == "heimi" and r[0]["checkpoints"] and r[0]["stages"][1]["growth"] == "depth"


def test_metrics_since_and_full_logs(client):
    assert [m["global_step"] for m in client.get("/api/runs/heimi/metrics").json()] == [1, 2, 3, 4]
    assert [m["global_step"] for m in client.get("/api/runs/heimi/metrics", params={"since": 2}).json()] == [3, 4]
    for name in ("eval", "events", "samples"):
        body = client.get(f"/api/runs/heimi/{name}").json()
        assert isinstance(body, list) and body
    o = client.get("/api/runs/heimi/overview").json()
    assert o["growth"][0]["kind"] == "depth" and o["growth"][0]["label"].endswith("加深") and len(o["params"]) >= 2


def test_unknown_or_traversing_run_is_404(client):
    for path in ("/api/runs/nope/metrics", "/api/runs/%2E%2E/metrics", "/api/runs/..%2F..%2Fx/events", "/api/runs/nope/overview"):
        assert client.get(path).status_code == 404, path


def test_status_shape(client):
    s = client.get("/api/status").json()
    assert s["run"] == "heimi" and s["state"] in ("running", "quiet", "stopped") and "gpu" in s and "eta_seconds" in s


def test_generate_hands_the_worker_a_validated_absolute_path(client):
    body = {"run": "heimi", "checkpoint": "smoke1_final.pt", "prefix": "The cat", "max_new_tokens": 8, "temperature": 0.5, "device": "cpu"}
    r = client.post("/api/generate", json=body)
    assert r.status_code == 200 and r.json()["text"] == " ok" and r.json()["checkpoint"] == "smoke1_final.pt" and r.json()["device"] == "cpu"
    req = client.worker.calls[0]
    assert req["ckpt_path"] == str(client.runs_dir / "heimi" / "ckpt" / "smoke1_final.pt")
    assert req["suffix"] is None and req["max_new_tokens"] == 8 and req["temperature"] == 0.5


def test_generate_rejects_bad_names_and_out_of_range_values(client):
    base = {"run": "heimi", "checkpoint": "smoke1_final.pt", "prefix": "x"}
    assert client.post("/api/generate", json={**base, "checkpoint": "../../outside.pt"}).status_code == 404
    assert client.post("/api/generate", json={**base, "run": ".."}).status_code == 404
    assert client.post("/api/generate", json={**base, "max_new_tokens": 9999}).status_code == 422
    assert client.post("/api/generate", json={**base, "device": "mps"}).status_code == 422
    assert client.post("/api/generate", json={**base, "prefix": ""}).status_code == 422
    assert client.worker.calls == []  # 不合法的请求一个都没到生成进程


def test_generate_timeout_is_504_and_worker_error_is_500(dash_runs):
    body = {"run": "heimi", "checkpoint": "smoke1_final.pt", "prefix": "x"}
    with TestClient(create_app(dash_runs[0], worker=FakeWorker(exc=WorkerTimeout("生成超过 60 秒")))) as c:
        r = c.post("/api/generate", json=body)
        assert r.status_code == 504 and "60" in r.json()["detail"]
    with TestClient(create_app(dash_runs[0], worker=FakeWorker(exc=WorkerError("RuntimeError: CUDA 不可用")))) as c:
        r = c.post("/api/generate", json=body)
        assert r.status_code == 500 and "CUDA" in r.json()["detail"]


def test_generate_end_to_end_with_real_worker(dash_runs):
    with TestClient(create_app(dash_runs[0], worker=WorkerClient())) as c:
        r = c.post("/api/generate", json={"run": "heimi", "checkpoint": "smoke0_final.pt", "prefix": "The cat", "max_new_tokens": 4, "temperature": 0.0})
        assert r.status_code == 200 and isinstance(r.json()["text"], str) and r.json()["seconds"] >= 0


def test_worker_is_closed_on_shutdown(dash_runs):
    worker = FakeWorker()
    with TestClient(create_app(dash_runs[0], worker=worker)):
        pass
    assert worker.closed
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_server.py -q`
Expected: FAIL，`ModuleNotFoundError: dashboard.server`

- [ ] **Step 3: 实现 `dashboard/server.py`**

`RunNotFound` 用 `app.exception_handler` 统一成 404；关闭钩子用 `lifespan`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_server.py -q`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add dashboard/server.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): FastAPI backend with validated run and checkpoint names"
```

---

### Task 5: 前端单页

**Files:**
- Create: `dashboard/static/chart.umd.min.js`（下载）、`dashboard/static/index.html`
- Test: `tests/test_dashboard_frontend.py`

**Interfaces:**
- Consumes：Task 4 的全部接口
- Produces：`index.html`，固定的结构约定（测试与 Task 6 的实看都按它找元素）：
  - `<html lang="zh-CN">`，`<title>黑米 Heimi · 训练窗口</title>`，顶栏显示"黑米 Heimi"、run 选择框 `#run-select`、连接状态 `#conn`
  - 四个 `<section>`：`id="overview"`（标题"成长总览"，`<canvas id="loss-chart">` 与 `<canvas id="params-chart">`）、`id="status"`（"当前状态"）、`id="samples"`（"成长样本墙"）、`id="playground"`（"试玩台"）
  - 空状态元素 `#empty`，文字"还没有训练记录"并给出 `python train.py --config configs/ladder.yaml --run heimi` 的提示

页面行为（签名决定不了的部分）：

1. **数据**：启动与每 5000 毫秒执行一次 `refresh()`：并行取 `/api/runs`、`/api/status`、`/api/runs/{run}/overview`、`/api/runs/{run}/samples`、`/api/runs/{run}/metrics?since=<已有的最大 global_step>`（指标按 run 缓存在内存里，增量追加）；存在名字以 `baseline` 开头的 run（spec 9.1 用的是 `baseline105m`）且当前选的不是它时，再取它的 `overview`。任何一个请求失败只把 `#conn` 标成"连接中断"，不清空已有图表，下一轮照常重试。默认选中名为 `heimi` 的 run，没有则选最近更新的、名字不以 `baseline` 开头的 run。
2. **成长总览**：主图横轴 `tokens_total`（线性，刻度写成 `0.3B` / `120M`），纵轴损失。数据集：训练损失（细线，取自指标）、验证 web、验证 code（带点的线，取自 `overview.eval`）、有对照组时"对照 web""对照 code"两条虚线。每条 `overview.growth` 画一条竖直虚线并在顶部写它的 `label`（如"13M → 26M 加深"），用页面内自带的一个 Chart.js 小插件画，不引入额外库。副图：`overview.params` 的阶梯线（`stepped: true`），纵轴按 `13M` 这样的格式写。图表原地更新数据，不整张重画。
3. **当前状态**：run、状态徽标（`running` → "训练中"，`quiet` → "评测中或暂停"，`stopped` → "已停止"）、阶段、步数 `step / steps`、损失、学习率、tok/s、显存、GPU 温度、预计剩余（时分格式）、最近一次评测的四个数（没有的写"—"）。
4. **成长样本墙**：按 `prompt_id` 分列（按首次出现的顺序），列头是提示原文，下面按时间从上到下（旧在上、新在下）排每次采样的续写，每条标出阶段与累计 token；默认只显示最近 8 次（仍是旧在上），"展开全部"按钮切换。从上往下读就是它一路学会说话的过程。
5. **试玩台**：前缀与可选后缀两个文本框、run 与检查点下拉（各级 `final` 排前）、设备（CPU / GPU）、最大 token 数、温度、"生成"按钮；勾选"并排比较"出现第二个检查点下拉，两次请求先后发出、结果左右并排，各自显示耗时；出错时显示接口返回的 `detail`。
6. **安全与外观**：所有来自接口的文本一律用 `textContent` 或 `createTextNode` 写入，页面里不出现 `innerHTML`、`insertAdjacentHTML`、`document.write`。宽度小于 900 像素时四个区域单列排布，不出现横向滚动。配色、字体与图表样式在动手前加载 `dataviz` 与 `frontend-design:frontend-design` 两个技能，按它们的规则定（明暗两套配色都要可读，web 与 code 两条线在色觉障碍下可分辨，对照组用虚线而不只靠颜色）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dashboard_frontend.py
import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

from dashboard.server import create_app
from tests.helpers_dashboard import FakeWorker

STATIC = Path("dashboard/static")


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_index_is_served_with_four_regions_in_chinese(dash_runs):
    with TestClient(create_app(dash_runs[0], worker=FakeWorker())) as c:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert c.get("/static/chart.umd.min.js").status_code == 200
    html = r.text
    assert "<title>黑米 Heimi" in html and 'lang="zh-CN"' in html
    for section_id, heading in (("overview", "成长总览"), ("status", "当前状态"), ("samples", "成长样本墙"), ("playground", "试玩台")):
        assert f'id="{section_id}"' in html and heading in html
    assert 'id="loss-chart"' in html and 'id="params-chart"' in html and 'id="empty"' in html and "还没有训练记录" in html


def test_page_is_self_contained_and_offline():
    html = _html()
    assert '<script src="/static/chart.umd.min.js"></script>' in html
    assert not re.search(r"""(src|href)\s*=\s*["'](https?:)?//""", html)  # 不引用任何外网资源
    assert not re.search(r"<link[^>]+stylesheet", html)  # CSS 内联
    js = (STATIC / "chart.umd.min.js").read_bytes()
    assert hashlib.sha256(js).hexdigest() == "87364cc2b2a76a6684929e7d6b80b18a897794db38400caff1fcf1791c6925d2"  # Chart.js 4.6.1


def test_model_text_is_never_inserted_as_html():
    html = _html()
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "document.write" not in html


def test_page_polls_every_five_seconds_and_uses_every_endpoint():
    html = _html()
    assert "5000" in html
    for path in ("/api/runs", "/api/status", "/overview", "/metrics?since=", "/samples", "/api/generate"):
        assert path in html, path
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_frontend.py -q`
Expected: 4 个 FAIL（`FileNotFoundError: dashboard\static\index.html` 或页面内容断言失败）

- [ ] **Step 3: 下载 Chart.js**

Run: `curl -s --compressed -o dashboard/static/chart.umd.min.js https://cdn.jsdelivr.net/npm/chart.js@4.6.1/dist/chart.umd.min.js`
Expected: 文件 210,082 字节，sha256 为 `87364cc2…6925d2`（Step 1 的测试核对）。

- [ ] **Step 4: 加载 `dataviz` 与 `frontend-design:frontend-design` 技能，写 `dashboard/static/index.html`**

按上面"固定的结构约定"和六条页面行为实现。

- [ ] **Step 5: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_frontend.py -q`
Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
git add dashboard/static/chart.umd.min.js dashboard/static/index.html tests/test_dashboard_frontend.py
git commit -m "feat(dashboard): single-page frontend with growth overview, status, sample wall and playground"
```

---

### Task 6: 入口与实机验证

**Files:**
- Create: `dashboard/cli.py`、`dashboard.py`、`docs/dashboard.md`
- Modify: `.gitignore`（加 `/.claude/`）；`train.py` 的说明文字与 `docs/training/stage0-report.md` 里对照组命令的 `--run baseline` 改成 spec 9.1 的 `--run baseline105m`（计划三写错了 run 名；窗口按前缀认对照组，两种名字都能叠加）
- Test: `tests/test_dashboard_cli.py`

**Interfaces:**
- Consumes：`create_app`、`WorkerClient`
- Produces（`dashboard/cli.py`）：
  ```python
  @dataclass(frozen=True)
  class Options:
      runs: Path
      host: str          # 恒为 "127.0.0.1"
      port: int
      open_browser: bool

  def build(argv: Sequence[str] | None = None) -> tuple[FastAPI, Options]
      # 参数 --runs（默认 "runs"）、--port（默认 8787）、--no-browser；repo_root 取当前目录
  def main(argv: Sequence[str] | None = None) -> int
      # build → 需要时 1 秒后 webbrowser.open(f"http://127.0.0.1:{port}") → uvicorn.run(app, host, port, log_level="warning") → 0
  ```
- Produces（`dashboard.py`）：`from dashboard.cli import main` 加 `raise SystemExit(main())`，用法 `python dashboard.py --runs runs/`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dashboard_cli.py
from pathlib import Path

from fastapi.testclient import TestClient

from dashboard.cli import build


def test_defaults_bind_localhost_8787_and_open_browser(tmp_path):
    app, opts = build(["--runs", str(tmp_path)])
    assert (opts.host, opts.port, opts.open_browser, opts.runs) == ("127.0.0.1", 8787, True, tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/runs").json() == []  # 还没有任何 run 时窗口也能开
        assert c.get("/api/status").json()["run"] is None


def test_port_and_no_browser_flags(tmp_path):
    _, opts = build(["--runs", str(tmp_path), "--port", "9000", "--no-browser"])
    assert opts.port == 9000 and opts.open_browser is False and opts.host == "127.0.0.1"


def test_root_script_is_a_thin_shim():
    src = Path("dashboard.py").read_text(encoding="utf-8")
    assert "from dashboard.cli import main" in src and "SystemExit(main())" in src
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_cli.py -q`
Expected: FAIL，`ModuleNotFoundError: dashboard.cli`

- [ ] **Step 3: 实现 `dashboard/cli.py` 与 `dashboard.py`**

- [ ] **Step 4: 运行确认通过，再跑全套**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_cli.py -q`，然后 `.venv/Scripts/python.exe -m pytest -q -W error`
Expected: 3 passed；全套通过（计划三后的 162 个加本计划新增的约 35 个）

- [ ] **Step 5: 用真实的 `runs/` 在浏览器里看**

`.gitignore` 加 `/.claude/`，新建 `.claude/launch.json`：配置名 `heimi-dashboard`，`runtimeExecutable` 为 `.venv/Scripts/python.exe`，`runtimeArgs` 为 `["dashboard.py", "--runs", "runs", "--no-browser"]`，`port` 8787。用内置浏览器打开它，逐项确认：

- 顶栏是"黑米 Heimi"，run 选择框里有 `heimi`、`smoke`、`smoke2`、`smoke3`，默认选中 `heimi`。
- 成长总览：训练损失线从约 7.7 降到约 4.4；验证 web / code 各有点（级 0 第 500 步与级末）；在 0.3B 处有一条标着"13M → 26M 加深"的竖线；参数量阶梯从 13M 升到 26M。切到 `smoke3` 能看到两条竖线（加深、加宽）。
- 当前状态：run `heimi`、阶段 `stage1`、步数 `20 / 953`、状态"已停止"、最近评测 web 4.47 / code 3.49 / HellaSwag 0.237 / FIM 0。
- 成长样本墙：6 列，每列 2 条，第 500 步的在上、第 572 步的在下；`fim_py` 列头显示带 `<FILL>` 的提示原文。
- 试玩台：选 `stage0_final.pt`、CPU、前缀 "The history of the Roman Empire began when"，60 秒内出结果并显示耗时；勾"并排比较"再选 `stage1_step20.pt`，两个结果并排。前缀填 `<b>bold</b> <script>alert(1)</script>` 生成一次：页面原样显示这串字符，没有加粗、没有弹窗。
- 浏览器控制台没有报错；把视口缩到 375 像素宽没有横向滚动；切到深色配色文字与曲线仍清晰。
- 用一个空目录启动一次（`--runs` 指向新建的空目录，换一个端口）：页面显示"还没有训练记录"，控制台无报错。

发现的问题当场修，修完重跑 `tests/test_dashboard_frontend.py`。

- [ ] **Step 6: 写 `docs/dashboard.md`**

内容：启动命令与参数；四个区域各看什么；三种状态徽标的含义与阈值；试玩台默认 CPU、105M 模型配长上下文建议在不训练时选 GPU、超时 60 秒；窗口只读 `runs/`，训练时开着不影响训练；对照组的 run 名以 `baseline` 开头（按 spec 用 `baseline105m`）才会叠加虚线；Chart.js 4.6.1（MIT）随仓库分发。同时按 Files 一栏改掉 `train.py` 与级 0 报告里的对照组 run 名。

- [ ] **Step 7: 提交**

```bash
git add dashboard/cli.py dashboard.py docs/dashboard.md docs/training/stage0-report.md train.py .gitignore tests/test_dashboard_cli.py
git commit -m "feat(dashboard): dashboard.py entry and usage notes (verified in the browser on real runs)"
```

---

## 完成标准

- `.venv/Scripts/python.exe -m pytest -q -W error` 全部通过（162 个加本计划新增的约 35 个）。
- `python dashboard.py --runs runs/` 在 `http://127.0.0.1:8787` 打开窗口，四个区域用真实的 `runs/heimi` 数据显示正确，控制台无报错。
- 试玩台能用级 0 检查点在 CPU 上生成，并排比较可用；超时与出错都有中文提示。
- 空的 runs 目录、带 `..` 的名字、同一步两条评测、模型输出里的 HTML 都按 Review Focus 的要求处理。
- `docs/dashboard.md` 写明用法。
