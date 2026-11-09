"""生成工作进程：窗口的试玩台在一个独立子进程里加载检查点并生成（spec 12.3）。

独立进程的原因：加载上 GB 的检查点、在 CPU 上逐 token 生成都很慢，不能堵住窗口的轮询，
也不能和训练进程共享任何状态。最多缓存 2 个模型，并排比较正好两个。
一次请求超过 timeout_s 没回话就把进程杀掉：卡住的生成不该让后面的请求排在一个死进程后面，
下一次请求会起一个新进程。

显存是训练的：检查点一律先在 CPU 上加载（里面还有两倍于模型的优化器状态，不能跟着上显卡），
只把模型搬到请求的设备；用过 GPU 的进程空闲 IDLE_EXIT_S 秒后整个退出，
把模型和 CUDA 上下文占的显存一起还回去。只用 CPU 的进程不退，下次不用重新加载。
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch

from minids.completion import complete_middle
from minids.data.tokenizer import HeimiTokenizer
from minids.model.transformer import Transformer
from minids.train.checkpoint import load_model

MAX_CACHED_MODELS = 2
DEFAULT_TIMEOUT_S = 60.0
IDLE_EXIT_S = 180.0  # 用过 GPU 的进程空闲这么久就退出，把显存还给训练
DEFAULT_TOKENIZER = "data/tokenizer/heimi-16k.json"
N_SPECIAL = 16  # id 小于它的是特殊 token，不属于生成的正文


class WorkerTimeout(RuntimeError):
    pass


class WorkerError(RuntimeError):
    pass


class ModelCache:
    """按 (路径, 修改时间, 设备) 缓存已加载的模型与分词器，最久未用的先丢。"""

    def __init__(self, max_models: int = MAX_CACHED_MODELS, repo_root: Path | None = None):
        self.max_models = max_models
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self._items: OrderedDict[tuple, tuple[Transformer, HeimiTokenizer]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def uses_gpu(self) -> bool:
        return any(str(key[2]).startswith("cuda") for key in self._items)

    def get(self, ckpt_path: str, device: str) -> tuple[Transformer, HeimiTokenizer]:
        path = Path(ckpt_path).resolve()
        key = (str(path), path.stat().st_mtime_ns, device)  # 文件被重写后修改时间变了，自动重新加载
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        model, meta = load_model(path, "cpu")  # 优化器状态留在内存里随即丢弃，不上显卡
        if device != "cpu":
            model = model.to(device)
        # 分词器路径记在检查点的配置快照里；相对路径相对仓库根目录
        tok_path = Path(((meta.get("ladder_config") or {}).get("data") or {}).get("tokenizer") or DEFAULT_TOKENIZER)
        if not tok_path.is_absolute():
            tok_path = self.repo_root / tok_path
        self._items[key] = (model, HeimiTokenizer.load(tok_path))
        while len(self._items) > self.max_models:
            self._items.popitem(last=False)
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        return self._items[key]


def generate_text(cache: ModelCache, req: dict) -> dict:
    """req 键：ckpt_path、prefix、suffix（str 或 None）、max_new_tokens、temperature、device。"""
    t0 = time.perf_counter()
    device = req["device"]
    model, tok = cache.get(req["ckpt_path"], device)
    seq_len = model.cfg.seq_len
    wanted = int(req["max_new_tokens"])
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
    with ctx:
        if req.get("suffix") is not None:
            # 填空：贪心补一行，预算与 complete.py 相同
            max_prefix = min(512, seq_len // 2)
            max_suffix = min(256, seq_len // 4)
            max_new = max(1, min(wanted, seq_len - max_prefix - max_suffix - 3))
            text = complete_middle(model, tok, req["prefix"], req["suffix"], max_new, device, max_prefix, max_suffix)
            n_tokens = len(tok.encode(text))
        else:
            # 续写：提示太长就只留末尾，给新 token 留出位置
            max_new = max(1, min(wanted, seq_len - 1))
            ids = tok.encode(req["prefix"])[-(seq_len - max_new):]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, max_new, temperature=float(req["temperature"]), top_k=50)
            new = out[0, idx.shape[1]:].tolist()
            stop = next((i for i, t in enumerate(new) if t < N_SPECIAL), len(new))  # 文档结束符之后是下一篇，不算续写
            new = new[:stop]
            text, n_tokens = tok.decode(new), len(new)
    return {"text": text, "tokens": n_tokens, "seconds": round(time.perf_counter() - t0, 3)}


def worker_main(conn, repo_root: str | None, idle_exit_s: float = IDLE_EXIT_S) -> None:
    """子进程入口：收到 None 退出；每个请求回 {"ok": True, ...} 或 {"ok": False, "error": ...}，一次失败不带走进程。

    空闲 idle_exit_s 秒且占着显存时自己退出（窗口那一端下次请求会再起一个）。
    """
    cache = ModelCache(repo_root=Path(repo_root) if repo_root else None)
    while True:
        try:
            if not conn.poll(idle_exit_s):
                if cache.uses_gpu:
                    break
                continue
            req = conn.recv()
        except (EOFError, OSError):
            break
        if req is None:
            break
        try:
            reply = {"ok": True, **generate_text(cache, req)}
        except Exception as e:  # noqa: BLE001 —— 任何失败都回报给窗口，由它显示
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        conn.send(reply)


class WorkerClient:
    """窗口进程里的一端：按需起子进程，一次只跑一个请求，超时就杀掉重来。"""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S, repo_root: Path | None = None, target=worker_main):
        self.timeout_s = timeout_s
        self.repo_root = str(repo_root) if repo_root else None
        self._target = target
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=self._target, args=(child, self.repo_root), daemon=True)
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def _kill(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
            self._proc = None
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.close()
            self._conn = None

    def generate(self, req: dict) -> dict:
        with self._lock:  # 计时从拿到锁开始：排队等别的请求的时间不算
            reply = None
            for attempt in (1, 2):
                if not self.is_alive():
                    self._kill()
                    self._start()
                try:
                    self._conn.send(req)
                    if not self._conn.poll(self.timeout_s):
                        self._kill()
                        raise WorkerTimeout(
                            f"生成超过 {self.timeout_s:g} 秒没有完成，生成进程已重启；可以缩短输入、减少 token 数，或在不训练时改用 GPU"
                        )
                    reply = self._conn.recv()
                    break
                except (EOFError, BrokenPipeError, OSError) as e:
                    # 进程在收发之间没了：多半是空闲退出恰好撞上新请求，换一个新进程再试一次
                    self._kill()
                    if attempt == 2:
                        raise WorkerError(f"生成进程意外退出：{type(e).__name__}: {e}") from e
        if not reply.get("ok"):
            raise WorkerError(reply.get("error", "未知错误"))
        return {k: v for k, v in reply.items() if k != "ok"}

    def close(self) -> None:
        with self._lock:
            if self.is_alive():
                with contextlib.suppress(OSError, ValueError):
                    self._conn.send(None)
                self._proc.join(timeout=1.0)
            self._kill()
