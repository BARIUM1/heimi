import sys

import pytest

from minids.keepawake import keep_awake


@pytest.mark.skipif(sys.platform != "win32", reason="只在 Windows 上有效果")
def test_keep_awake_takes_effect_on_windows():
    with keep_awake() as active:
        assert active is True


def test_keep_awake_is_a_context_manager_everywhere():
    with keep_awake() as active:
        assert isinstance(active, bool)
