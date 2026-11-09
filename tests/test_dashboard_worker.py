import pytest

from dashboard.worker import ModelCache, WorkerClient, WorkerError, WorkerTimeout, generate_text
from tests.helpers_dashboard import sleepy_worker


def _req(ckpt, **over):
    r = {"ckpt_path": str(ckpt), "prefix": "The cat", "suffix": None, "max_new_tokens": 4, "temperature": 0.0, "device": "cpu"}
    r.update(over)
    return r


def test_cache_reuses_and_evicts_least_recently_used(dash_runs):
    ck = dash_runs[0] / "heimi" / "ckpt"
    cache = ModelCache(max_models=2)
    a1, _ = cache.get(str(ck / "smoke0_final.pt"), "cpu")
    a2, _ = cache.get(str(ck / "smoke0_final.pt"), "cpu")
    assert a1 is a2 and len(cache) == 1
    b1, _ = cache.get(str(ck / "smoke1_final.pt"), "cpu")
    cache.get(str(ck / "smoke0_final.pt"), "cpu")  # 刷新 smoke0 的使用时间
    cache.get(str(ck / "smoke0_step2.pt"), "cpu")  # 第三个：挤掉最久未用的 smoke1
    assert len(cache) == 2
    assert cache.get(str(ck / "smoke0_final.pt"), "cpu")[0] is a1
    assert cache.get(str(ck / "smoke1_final.pt"), "cpu")[0] is not b1  # 被挤掉过，重新加载


def test_generate_plain_is_reproducible_at_temperature_zero_and_fim_is_one_line(dash_runs):
    ckpt = dash_runs[0] / "heimi" / "ckpt" / "smoke1_final.pt"
    cache = ModelCache()
    out = generate_text(cache, _req(ckpt))
    assert isinstance(out["text"], str) and 0 <= out["tokens"] <= 4 and out["seconds"] >= 0
    assert generate_text(cache, _req(ckpt))["text"] == out["text"]
    fim = generate_text(cache, _req(ckpt, prefix="def f():\n    ", suffix="\n"))
    assert "\n" not in fim["text"]


def test_long_prompt_is_cropped_not_rejected(dash_runs):
    ckpt = dash_runs[0] / "heimi" / "ckpt" / "smoke0_final.pt"
    out = generate_text(ModelCache(), _req(ckpt, prefix="word " * 500, max_new_tokens=3))  # 远超 seq_len 32
    assert out["tokens"] <= 3


def test_subprocess_round_trip_and_error_reporting(dash_runs):
    ck = dash_runs[0] / "heimi" / "ckpt"
    client = WorkerClient(timeout_s=60.0)
    try:
        assert isinstance(client.generate(_req(ck / "smoke0_final.pt"))["text"], str)
        with pytest.raises(WorkerError, match="missing.pt"):
            client.generate(_req(ck / "missing.pt"))
        assert client.is_alive()  # 出错不等于进程死掉
        assert isinstance(client.generate(_req(ck / "smoke0_final.pt"))["text"], str)
    finally:
        client.close()
    assert not client.is_alive()


def test_timeout_kills_the_stuck_worker():
    client = WorkerClient(timeout_s=1.0, target=sleepy_worker)
    try:
        with pytest.raises(WorkerTimeout, match="1"):
            client.generate(_req("whatever.pt"))
        assert not client.is_alive()  # 卡住的进程已被杀掉，下一次请求会起一个新的
    finally:
        client.close()
