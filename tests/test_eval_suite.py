from dataclasses import replace

import pytest

from minids.data.tokenizer import HeimiTokenizer
from minids.eval.suite import EvalSuite
from minids.model.transformer import Transformer
from tests.helpers_train import tiny_ladder


def test_suite_runs_everything_but_hellaswag_when_disabled(tmp_path):
    cfg = tiny_ladder(tmp_path)
    tok = HeimiTokenizer.load(cfg.data.tokenizer)
    suite = EvalSuite(cfg, tok, "cpu")
    out = suite.run(Transformer(cfg.stages[0].model), micro_batch=4, hellaswag=True, fim=True)
    assert set(out) == {"val_loss_web", "val_loss_code", "hellaswag_acc", "fim_em"}
    assert out["val_loss_web"] > 0 and out["val_loss_code"] > 0
    assert out["hellaswag_acc"] is None and 0.0 <= out["fim_em"] <= 1.0


def test_fim_skipped_when_not_requested(tmp_path):
    cfg = tiny_ladder(tmp_path)
    tok = HeimiTokenizer.load(cfg.data.tokenizer)
    out = EvalSuite(cfg, tok, "cpu").run(Transformer(cfg.stages[0].model), micro_batch=4, hellaswag=False, fim=False)
    assert out["fim_em"] is None


def test_missing_hellaswag_file_is_a_clear_error(tmp_path):
    cfg = tiny_ladder(tmp_path)
    cfg = replace(cfg, eval=replace(cfg.eval, hellaswag=True))
    with pytest.raises(FileNotFoundError, match="hellaswag"):
        EvalSuite(cfg, HeimiTokenizer.load(cfg.data.tokenizer), "cpu")
