"""窗口测试共用：一个永不回话的工作进程入口，和一个不起子进程的假生成器。"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient


def local_client(app) -> TestClient:
    """像浏览器那样用本机地址访问窗口：后台只认 127.0.0.1 / localhost 这两个 Host。"""
    return TestClient(app, base_url="http://127.0.0.1:8787")


def sleepy_worker(conn, repo_root) -> None:
    """用来测超时：收到请求也不回话。必须是模块级函数，spawn 出来的子进程要能按名字导入它。"""
    time.sleep(30)


def echo_worker(conn, repo_root) -> None:
    """正常回话但不加载模型：把前缀原样送回。"""
    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError):
            return
        if req is None:
            return
        conn.send({"ok": True, "text": req.get("prefix", ""), "tokens": 0, "seconds": 0.0})


def _first_start(repo_root, name: str) -> bool:
    """这个工作进程是不是第一次启动：在 repo_root（测试传进来的临时目录）下留一个标记文件来记。"""
    from pathlib import Path

    marker = Path(repo_root) / name
    if marker.exists():
        return False
    marker.write_text("started", encoding="utf-8")
    return True


def flaky_worker(conn, repo_root) -> None:
    """第一次启动时装死，被杀掉重启后正常回话：用来测"超时之后下一次请求能用"。"""
    if _first_start(repo_root, "flaky-worker-started"):
        time.sleep(30)
        return
    echo_worker(conn, repo_root)


def dying_worker(conn, repo_root) -> None:
    """第一次启动时收到请求不回话就退出（模拟空闲退出与新请求撞车），重启后正常回话。"""
    if _first_start(repo_root, "dying-worker-started"):
        conn.recv()
        return
    echo_worker(conn, repo_root)


class FakeWorker:
    """接口测试用：记录收到的请求，按预设返回结果或抛异常。"""

    def __init__(self, result: dict | None = None, exc: Exception | None = None):
        self.result = result or {"text": " ok", "seconds": 0.01, "tokens": 1}
        self.exc = exc
        self.calls: list[dict] = []
        self.closed = False

    def generate(self, req: dict) -> dict:
        self.calls.append(req)
        if self.exc is not None:
            raise self.exc
        return dict(self.result)

    def close(self) -> None:
        self.closed = True
