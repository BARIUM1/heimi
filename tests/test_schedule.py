import math

from minids.train.schedule import lr_at, warmup_steps


def test_warmup_steps_from_spec_fractions():
    assert warmup_steps(572, 0.02) == 11 and warmup_steps(953, 0.05) == 48
    assert warmup_steps(3814, 0.05) == 191 and warmup_steps(10, 0.0) == 1


def test_linear_warmup_reaches_peak_on_last_warmup_step():
    assert math.isclose(lr_at(0, 1000, 1e-3, 50), 1e-3 / 50)
    assert math.isclose(lr_at(49, 1000, 1e-3, 50), 1e-3)
    assert math.isclose(lr_at(50, 1000, 1e-3, 50), 1e-3)


def test_cosine_ends_at_ten_percent_and_clamps_beyond_total():
    assert math.isclose(lr_at(1000, 1000, 1e-3, 50), 1e-4)
    assert math.isclose(lr_at(1500, 1000, 1e-3, 50), 1e-4)
    mid = lr_at(525, 1000, 1e-3, 50)  # 衰减段正中
    assert math.isclose(mid, 1e-4 + 0.5 * 9e-4, rel_tol=1e-6)


def test_non_increasing_after_warmup():
    vals = [lr_at(s, 1000, 1e-3, 50) for s in range(50, 1001)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
