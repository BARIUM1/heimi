import math
from dataclasses import replace

import pytest
import torch

pytest.importorskip("lm_eval")  # 没装 official 额外依赖时整份跳过；本机必须是装了的，见完成标准

from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window  # noqa: E402

from minids.model.transformer import Transformer  # noqa: E402
from minids.official.harness import HARNESS_TASKS, HeimiLM, flatten_results, harness_version  # noqa: E402
from minids.official.scoring import continuation_logprobs, encode_pair, rolling_logprob, rolling_windows  # noqa: E402
from tests.helpers_data import tiny_tokenizer  # noqa: E402
from tests.helpers_train import CountingModel  # noqa: E402


class _NumTok:
    """把 "1 2 3" 编成 [1, 2, 3]：让测试能直接指定 token。"""

    def encode(self, text):
        return [int(x) for x in text.split()]


def _req(kind, *args):
    return Instance(request_type=kind, doc={}, arguments=args, idx=0)


@pytest.fixture
def lm(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    torch.manual_seed(0)
    return HeimiLM(Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size)).eval(), tok, "cpu")


def test_loglikelihood_scores_each_request_like_the_scoring_core(lm):
    reqs = [_req("loglikelihood", "The quick brown", " fox"), _req("loglikelihood", "", " The cat sat."), _req("loglikelihood", "A man is ", "sitting")]
    got = lm.loglikelihood(reqs)
    want = continuation_logprobs(lm.model, [encode_pair(lm.tok, *r.args) for r in reqs], "cpu", lm.max_len)
    assert len(got) == 3 and all(type(g[0]) is float and type(g[1]) is bool for g in got)
    assert all(math.isclose(g[0], w[0], rel_tol=1e-4, abs_tol=1e-4) and g[1] == w[1] for g, w in zip(got, want))


def test_empty_context_is_conditioned_on_the_document_separator():
    # CountingModel 认为"下一个 token = 当前 + 1"。空上下文若真的以 id 0 为条件，"1 2 3" 就是它的首选
    lm = HeimiLM(CountingModel(vocab_size=64), _NumTok(), "cpu")
    good, bad = lm.loglikelihood([_req("loglikelihood", "", "1 2 3"), _req("loglikelihood", "", "1 3 2")])
    assert good[1] is True and bad[1] is False and good[0] > bad[0]


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 128, 150])
def test_rolling_windows_are_the_harness_windows(n):
    tokens = list(range(100, 100 + n))
    theirs = [make_disjoint_window(w) for w in get_rolling_token_windows(token_list=tokens, prefix_token=0, max_seq_len=64, context_len=1)]
    assert rolling_windows(tokens, 0, 64) == [(list(a), list(b)) for a, b in theirs]


def test_loglikelihood_rolling_scores_the_whole_text_across_windows(lm):
    text = "The quick brown fox jumps over the lazy dog. " * 12
    ids = lm.tok.encode(text)
    assert len(ids) > lm.max_len  # 确实跨了窗口
    (got,) = lm.loglikelihood_rolling([_req("loglikelihood_rolling", text)])
    assert type(got) is float and math.isclose(got, rolling_logprob(lm.model, ids, "cpu", lm.max_len), rel_tol=1e-4)


CANNED = {
    "results": {
        "blimp": {"alias": "blimp", "name": "blimp", "sample_len": 134, "acc,none": 0.75, "acc_stderr,none": "N/A", "sample_count": {"acc,none": 134}},
        "blimp_anaphor_gender_agreement": {"alias": " - x", "acc,none": 1.0, "acc_stderr,none": "N/A"},
        "hellaswag": {"name": "hellaswag", "sample_len": 4, "acc,none": 0.25, "acc_stderr,none": 0.01, "acc_norm,none": 0.5, "acc_norm_stderr,none": 0.02},
        "wikitext": {
            "word_perplexity,none": 128.03, "word_perplexity_stderr,none": "N/A",
            "byte_perplexity,none": 2.588, "byte_perplexity_stderr,none": "N/A",
            "bits_per_byte,none": 1.372, "bits_per_byte_stderr,none": "N/A",
        },
    }
}


def test_flatten_keeps_the_numbers_of_the_requested_tasks():
    assert flatten_results(CANNED, HARNESS_TASKS) == {
        "blimp_acc": 0.75, "hellaswag_acc": 0.25, "hellaswag_acc_norm": 0.5,
        "wikitext_word_perplexity": 128.03, "wikitext_byte_perplexity": 2.588, "wikitext_bits_per_byte": 1.372,
    }


def test_flatten_names_a_task_the_harness_did_not_return():
    with pytest.raises(ValueError, match="wikitext"):
        flatten_results({"results": {"blimp": {"acc,none": 0.5}}}, ("blimp", "wikitext"))


def test_generation_is_not_offered_through_the_harness(lm):
    with pytest.raises(NotImplementedError):
        lm.generate_until([])


def test_harness_version_is_the_pinned_one():
    assert harness_version() == "0.4.13"
