"""读 GPU 温度：调 nvidia-smi。任何失败都返回 None，监控不能把训练拖垮。"""

from __future__ import annotations

import subprocess


def gpu_temperature_c(timeout_s: float = 2.0) -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return float(r.stdout.strip().splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_status(timeout_s: float = 2.0) -> dict | None:
    """第一块 GPU 的温度、显存（GB）、利用率，给窗口的"当前状态"用；任何失败返回 None。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        temp, used, total, util = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return {"temp_c": temp, "mem_used_gb": used / 1024, "mem_total_gb": total / 1024, "util_pct": util}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
