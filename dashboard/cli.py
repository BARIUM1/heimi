"""窗口的命令行：python dashboard.py --runs runs/ [--port 8787] [--no-browser]

只绑 127.0.0.1，启动后自动打开浏览器。仓库根目录的 dashboard.py 只是转发到这里：
根目录同时有包 dashboard/ 和脚本 dashboard.py，`import dashboard` 得到的是包。
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from dashboard.server import create_app

HOST = "127.0.0.1"  # 只给本机看，不提供改绑其他地址的选项
# 检查点里记的分词器路径是相对仓库根目录的；按本文件的位置定根目录，从哪个目录启动都一样
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Options:
    runs: Path
    host: str
    port: int
    open_browser: bool


def build(argv: Sequence[str] | None = None) -> tuple[FastAPI, Options]:
    p = argparse.ArgumentParser(prog="dashboard.py", description="黑米 Heimi 训练窗口")
    p.add_argument("--runs", default="runs", help="训练输出目录，里面每个子目录是一次训练")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = p.parse_args(argv)
    opts = Options(runs=Path(args.runs), host=HOST, port=args.port, open_browser=not args.no_browser)
    return create_app(opts.runs, repo_root=REPO_ROOT), opts


def main(argv: Sequence[str] | None = None) -> int:
    import uvicorn

    app, opts = build(argv)
    url = f"http://{opts.host}:{opts.port}"
    print(f"黑米的训练窗口：{url}（Ctrl+C 退出）")
    if opts.open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=opts.host, port=opts.port, log_level="warning")
    return 0
