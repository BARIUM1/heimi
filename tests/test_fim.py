import numpy as np
import pytest

from minids.data.fim import apply_fim_psm, undo_fim_psm


def test_layout_and_length():
    rng = np.random.default_rng(0)
    t = np.arange(20, 40)
    out = apply_fim_psm(t, rng)
    assert len(out) == 24 and out[0] == 1 and out[-1] == 0
    assert list(out).count(3) == 1 and list(out).count(2) == 1
    assert list(out).index(1) < list(out).index(3) < list(out).index(2)


def test_round_trip_many_seeds():
    t = np.arange(100, 160)
    for seed in range(50):
        assert np.array_equal(undo_fim_psm(apply_fim_psm(t, np.random.default_rng(seed))), t)


def test_empty_segments_allowed():
    t = np.arange(10, 15)

    class Rng:  # 强制 a = b = 0：prefix 与 middle 都为空
        def integers(self, lo, hi, size):
            return np.array([0, 0])

    out = apply_fim_psm(t, Rng())
    assert np.array_equal(undo_fim_psm(out), t)


def test_undo_rejects_malformed():
    with pytest.raises(ValueError):
        undo_fim_psm(np.array([1, 5, 6, 2, 7, 0]))  # 缺 suffix 标记
