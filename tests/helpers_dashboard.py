"""窗口测试共用：一个永不回话的工作进程入口，和一个不起子进程的假生成器。"""

from __future__ import annotations

import time


def sleepy_worker(conn, repo_root) -> None:
    """用来测超时：收到请求也不回话。必须是模块级函数，spawn 出来的子进程要能按名字导入它。"""
    time.sleep(30)


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
