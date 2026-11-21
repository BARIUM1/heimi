import math

import pytest
import torch
import torch.nn.functional as F

from minids.model.transformer import Transformer
from minids.official.scoring import continuation_logprobs, encode_pair, rolling_logprob, rolling_windows
from tests.helpers_data import tiny_tokenizer


@pytest.fixture
def model(tiny_cfg):
    torch.manual_seed(0)
    return Transformer(tiny_cfg).eval()  # 词表 256、上下文 64


def _direct(model, ctx, cont):
    """不经过被测代码，直接算续写的对数似然之和与是否贪心。"""
    logp = F.log_softmax(model(torch.tensor([ctx + cont[:-1]]))[0][0].double(), dim=-1)
    rows = logp[len(ctx) - 1 :]
    want = torch.tensor(cont)
    return rows[torch.arange(len(cont)), want].sum().item(), bool((rows.argmax(-1) == want).all())


def _random_ids(n, seed):
    return torch.randint(16, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_score_is_the_summed_log_probability_of_the_continuation(model):
    ctx, cont = [20, 21, 22, 23], [30, 31, 32]
    ((lp, greedy),) = continuation_logprobs(model, [(ctx, cont)], "cpu", max_len=64)
    want, want_greedy = _direct(model, ctx, cont)
    assert math.isclose(lp, want, rel_tol=1e-4, abs_tol=1e-4) and greedy == want_greedy


def test_batching_and_padding_change_neither_scores_nor_order(model):
    shapes = [(3, 2), (20, 5), (1, 1), (40, 7), (9, 9), (2, 30), (15, 1)]
    pairs = [(_random_ids(n, i), _random_ids(m, 100 + i)) for i, (n, m) in enumerate(shapes)]
    one = [continuation_logprobs(model, [p], "cpu", 64)[0] for p in pairs]
    batched = continuation_logprobs(model, pairs, "cpu", 64, batch_size=3)
    assert [g for _, g in batched] == [g for _, g in one]
    assert all(math.isclose(a[0], b[0], rel_tol=1e-4, abs_tol=1e-4) for a, b in zip(batched, one))


def test_long_context_is_cut_from_the_left_and_the_continuation_kept(model):
    ctx, cont = _random_ids(100, 2), [40, 41, 42, 43, 44]
    full = continuation_logprobs(model, [(ctx, cont)], "cpu", max_len=64)[0]
    kept = continuation_logprobs(model, [(ctx[-(64 + 1 - len(cont)) :], cont)], "cpu", max_len=64)[0]
    assert math.isclose(full[0], kept[0], rel_tol=1e-6, abs_tol=1e-6)


def test_greedy_flag_is_true_only_when_every_token_is_the_models_first_choice(model):
    ctx = [20, 21, 22]
    best = model.generate(torch.tensor([ctx]), 4, temperature=0.0)[0, len(ctx) :].tolist()
    assert continuation_logprobs(model, [(ctx, best)], "cpu", 64)[0][1] is True
    other = best[:3] + [(best[3] + 1) % 256]
    assert continuation_logprobs(model, [(ctx, other)], "cpu", 64)[0][1] is False


def test_empty_continuation_scores_zero_without_running_the_model():
    assert continuation_logprobs(None, [([5], [])], "cpu", 64) == [(0.0, True)]


def test_a_continuation_longer_than_the_context_window_is_refused(model):
    continuation_logprobs(model, [([0], list(range(16, 16 + 64)))], "cpu", max_len=64)  # 恰好等于窗口：可以
    with pytest.raises(ValueError, match="rolling"):
        continuation_logprobs(model, [([0], list(range(16, 16 + 65)))], "cpu", max_len=64)


def test_empty_context_is_replaced_by_the_document_separator(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    assert encode_pair(tok, "", " The cat sat.") == ([0], tok.encode(" The cat sat."))


def test_trailing_space_of_the_context_moves_onto_the_continuation(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    ctx, cont = encode_pair(tok, "The cat ", "sat")
    assert ctx == tok.encode("The cat") and cont == tok.encode("The cat sat")[len(ctx) :]


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 128, 150])
def test_rolling_windows_cover_every_token_exactly_once(n):
    tokens = list(range(100, 100 + n))
    windows = rolling_windows(tokens, prefix_token=0, max_len=64)
    assert [t for _, cont in windows for t in cont] == tokens
    assert all(len(ctx) >= 1 and len(ctx) + len(cont) <= 65 for ctx, cont in windows)
    if windows:
        assert windows[0][0] == [0]
    for ctx, cont in windows[1:]:
        start = tokens.index(cont[0])
        assert ctx == tokens[start - len(ctx) : start]  # 紧挨在这一窗之前的原文


def test_rolling_logprob_is_the_sum_over_windows(model):
    tokens = _random_ids(150, 3)
    want = sum(lp for lp, _ in continuation_logprobs(model, rolling_windows(tokens, 0, 64), "cpu", 64))
    assert math.isclose(rolling_logprob(model, tokens, "cpu", 64), want, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(rolling_logprob(model, [30, 31, 32], "cpu", 64), _direct(model, [0], [30, 31, 32])[0], rel_tol=1e-4, abs_tol=1e-4)
    assert rolling_logprob(model, [], "cpu", 64) == 0.0
