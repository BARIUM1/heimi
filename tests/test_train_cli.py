import yaml

import train
from minids.logging_io import read_jsonl
from tests.helpers_train import tiny_ladder


def test_cli_trains_with_max_steps_and_exits_zero(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(3,))
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    rc = train.main(["--config", str(p), "--run", "t", "--runs-dir", str(tmp_path / "runs"), "--device", "cpu", "--max-steps", "2"])
    assert rc == 0
    assert (tmp_path / "runs" / "t" / "ckpt" / "smoke0_step2.pt").exists()
    assert len(read_jsonl(tmp_path / "runs" / "t" / "metrics.jsonl")) == 2


def test_tiny_smoke_yaml_parses():
    from minids.train.config import load_ladder

    cfg = load_ladder("configs/tiny_smoke.yaml")
    assert [s.name for s in cfg.stages] == ["smoke0", "smoke1", "smoke2", "smoke_task"]
    assert [cfg.steps_for(s) for s in cfg.stages] == [30, 20, 20, 10] and cfg.eval.hellaswag is False
