import pytest

from minids.logging_io import (
    EVENTS_FILE,
    METRICS_FILE,
    EvalRecord,
    EventRecord,
    MetricsRecord,
    RunLogger,
    SampleRecord,
    append_jsonl,
    now_iso,
    read_jsonl,
)


def _metrics(gs, loss=1.0):
    return MetricsRecord(stage="s0", step=gs, global_step=gs, tokens_stage=gs * 10, tokens_total=gs * 10,
                         loss=loss, lr=1e-3, grad_norm=0.5, tok_per_s=1000.0, mem_gb=1.5, gpu_temp_c=None, time=now_iso())


def test_round_trip_all_record_types(tmp_path):
    log = RunLogger(tmp_path)
    log.metrics(_metrics(1))
    log.eval(EvalRecord("s0", 1, 10, 2.0, 3.0, None, 0.1, now_iso()))
    log.event("growth", "s1", 1, 10, params_before=5, params_after=9)
    log.sample(SampleRecord("s0", 1, 10, "p", "The", " cat", now_iso()))
    m, e, ev, s = (read_jsonl(log.path(n)) for n in ("metrics.jsonl", "eval.jsonl", "events.jsonl", "samples.jsonl"))
    assert m[0]["loss"] == 1.0 and set(m[0]) == set(MetricsRecord.__dataclass_fields__)
    assert e[0]["hellaswag_acc"] is None and e[0]["fim_em"] == 0.1
    assert ev[0]["type"] == "growth" and ev[0]["data"] == {"params_before": 5, "params_after": 9}
    assert s[0]["completion"] == " cat"


def test_since_filters_strictly_greater(tmp_path):
    p = tmp_path / METRICS_FILE
    for gs in (10, 20, 30):
        append_jsonl(p, _metrics(gs))
    assert [r["global_step"] for r in read_jsonl(p, since_global_step=20)] == [30]
    assert len(read_jsonl(p)) == 3


def test_missing_file_reads_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_half_written_last_line_is_skipped(tmp_path):
    # 窗口读的时候训练进程可能正写到一半
    p = tmp_path / METRICS_FILE
    append_jsonl(p, _metrics(1))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"stage": "s0", "step": 2, "glo')
    assert [r["step"] for r in read_jsonl(p)] == [1]


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError, match="type"):
        EventRecord("exploded", "s0", 0, 0, now_iso())


def test_unicode_kept_readable(tmp_path):
    p = tmp_path / EVENTS_FILE
    append_jsonl(p, EventRecord("stage_start", "级0", 0, 0, now_iso(), {"note": "黑米"}))
    assert "黑米" in p.read_text(encoding="utf-8") and read_jsonl(p)[0]["stage"] == "级0"
