import io
import json

import torch

import complete
import export as export_cli
from minids.export import export_model, load_exported
from minids.train.checkpoint import load_model
from minids.train.loop import run_ladder
from tests.helpers_train import tiny_ladder


def _ckpt(tmp_path):
    cfg = tiny_ladder(tmp_path, stage_steps=(1,))
    run_ladder(cfg, tmp_path / "run", "cpu")
    return cfg, tmp_path / "run" / "ckpt" / "smoke0_final.pt"


def test_export_then_load_gives_identical_logits(tmp_path):
    cfg, ckpt = _ckpt(tmp_path)
    out = export_model(ckpt, cfg.data.tokenizer, tmp_path / "exported")
    assert {p.name for p in out.iterdir()} == {"model.safetensors", "config.json", "tokenizer.json"}
    meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert meta["name"] == "heimi" and meta["stage"] == "smoke0" and meta["model"]["vocab_size"] == 300
    m, tok = load_exported(out, "cpu")
    ref, _ = load_model(ckpt, "cpu")
    x = torch.randint(16, 300, (1, 8))
    assert torch.equal(m(x)[0], ref(x)[0]) and tok.vocab_size == 300 and not m.training


def test_export_cli_and_complete_cli_round_trip(tmp_path, capsys):
    cfg, ckpt = _ckpt(tmp_path)
    assert export_cli.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "exp"), "--tokenizer", cfg.data.tokenizer]) == 0
    (tmp_path / "p.txt").write_text("def f():\n    ", encoding="utf-8")
    (tmp_path / "s.txt").write_text("\nprint(f())\n", encoding="utf-8")
    rc = complete.main(["--model", str(tmp_path / "exp"), "--prefix-file", str(tmp_path / "p.txt"),
                        "--suffix-file", str(tmp_path / "s.txt"), "--max-tokens", "4", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0 and out.endswith("\n") and out.count("\n") == 1


def test_complete_reads_prefix_from_stdin(tmp_path, capsys, monkeypatch):
    cfg, ckpt = _ckpt(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("x = "))
    rc = complete.main(["--ckpt", str(ckpt), "--tokenizer", cfg.data.tokenizer, "--max-tokens", "3", "--device", "cpu"])
    assert rc == 0 and capsys.readouterr().out.count("\n") == 1
