"""黑米的训练窗口入口：python dashboard.py --runs runs/ [--port 8787] [--no-browser]"""

from dashboard.cli import main

if __name__ == "__main__":  # 这个判断不能省：生成子进程会重新导入本文件
    raise SystemExit(main())
