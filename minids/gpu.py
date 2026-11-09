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
