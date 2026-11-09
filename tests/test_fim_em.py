import numpy as np
import pytest

from minids.eval.fim_em import FimCase, fim_exact_match, make_fim_cases, split_documents
from tests.helpers_data import tiny_tokenizer
from tests.helpers_train import ScriptedModel

DOCS = ["a = 1\nb = 2\nc = 3\n", "one\n\ntwo\nthree", "short\n", "x\ny\nz\nw\n"]


def test_split_documents_on_eot_drops_empty():
    assert [d.tolist() for d in split_documents(np.array([5, 6, 0, 0, 7, 0]))] == [[5, 6], [7]]


def test_cases_reconstruct_doc_and_pick_nonblank_line():
    cases = make_fim_cases(DOCS, n=10, rng=np.random.default_rng(0), min_lines=3)
    assert len(cases) == 3  # "short\n" 只有一行，被排除
    assert {c.prefix + c.middle + c.suffix for c in cases} == set(DOCS) - {"short\n"}
    for c in cases:
        assert c.middle.strip() and "\n" not in c.middle


def test_cases_deterministic_per_seed():
    a = make_fim_cases(DOCS, 3, np.random.default_rng(7))
    b = make_fim_cases(DOCS, 3, np.random.default_rng(7))
    assert a == b


def test_exact_match_rate(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    cases = [FimCase("a\n", "b = 2", "\nc\n"), FimCase("a\n", "zzz", "\nc\n")]
    model = ScriptedModel(tok.encode("b = 2") + tok.encode("\n"))
    assert fim_exact_match(model, tok, cases, "cpu", 8, 4, 4) == 0.5
    with pytest.raises(ValueError):
        fim_exact_match(model, tok, [], "cpu")
