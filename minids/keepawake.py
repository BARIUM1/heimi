"""Windows 防休眠：长任务（下载、预处理、训练）期间阻止系统待机，不阻止关屏。

用 SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)，退出上下文时恢复。
非 Windows 平台是空操作。
"""

import contextlib
import sys
from collections.abc import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_state(flags: int) -> int:
    import ctypes

    return int(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))


@contextlib.contextmanager
def keep_awake() -> Iterator[bool]:
    """进入时阻止待机，产出是否生效；退出时恢复默认。"""
    if sys.platform != "win32":
        yield False
        return
    previous = _set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield previous != 0
    finally:
        _set_state(ES_CONTINUOUS)
