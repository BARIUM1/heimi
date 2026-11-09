from minids.memguard import cuda_memory_snapshot, exceeds_memory_budget


def test_exceeds_budget_boundaries():
    # 0.8 * 16 = 12.8 GB 是阈值
    assert not exceeds_memory_budget(reserved_gb=12.0, total_gb=16.0, max_frac=0.8)
    assert exceeds_memory_budget(reserved_gb=12.9, total_gb=16.0, max_frac=0.8)
    assert not exceeds_memory_budget(reserved_gb=0.0, total_gb=16.0, max_frac=0.8)


def test_snapshot_keys_and_nonnegative():
    s = cuda_memory_snapshot()
    assert set(s) == {"allocated_gb", "reserved_gb", "total_gb"}
    assert all(v >= 0 for v in s.values())
