import json

import yaml

import evaluate
import sample
from minids.logging_io import read_jsonl
from minids.train.loop import run_ladder
from tests.helpers_train import tiny_ladder


def _trained(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(2,))
    run_ladder(cfg, tmp_path / "run", "cpu")
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg, tmp_path / "run" / "ckpt" / "smoke0_final.pt", p


def test_evaluate_appends_record_and_prints_json(tmp_path, capsys):
    cfg, ckpt, yml = _trained(tmp_path)
    n_before = len(read_jsonl(tmp_path / "run" / "eval.jsonl"))
    assert evaluate.main(["--ckpt", str(ckpt), "--config", str(yml), "--device", "cpu", "--fim-samples", "2"]) == 0
    recs = read_jsonl(tmp_path / "run" / "eval.jsonl")
    assert len(recs) == n_before + 1 and recs[-1]["stage"] == "smoke0" and recs[-1]["global_step"] == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["val_loss_web"] == recs[-1]["val_loss_web"]


def test_sample_prints_prompt_and_continuation(tmp_path, capsys):
    cfg, ckpt, _ = _trained(tmp_path)
    assert sample.main(["--ckpt", str(ckpt), "--tokenizer", cfg.data.tokenizer, "--prompt", "The cat", "--max-tokens", "3", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("The cat") and len(out) > len("The cat")
