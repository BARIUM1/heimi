import math
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from minids.eval.hellaswag import choose_ending, ending_scores, hellaswag_accuracy, load_hellaswag
from minids.model.transformer import Transformer
from tests.helpers_data import tiny_tokenizer
from tests.helpers_train import CountingModel


def test_choose_ending_prefers_most_likely_continuation():
    m = CountingModel(vocab_size=128)
    ctx = [10, 11, 12]
    endings = [[13, 14], [20, 21, 22], [13, 99], [5]]
    assert choose_ending(m, ctx, endings, "cpu", max_len=64) == 0


def test_scores_are_per_token_means_not_sums():
    # 两个候选每个 token 都"正确"：按均值两者只差浮点噪声（log_softmax 的求和顺序随 10 所在列而变），按求和则差 3 倍
    m = CountingModel(vocab_size=128)
    s = ending_scores(m, [10, 11], [[12, 13, 14], [12]], "cpu", max_len=64)
    assert math.isclose(s[0], s[1], rel_tol=1e-3) and s[0] < 0


def test_load_hellaswag_casts_string_label(tmp_path):
    p = tmp_path / "hs.parquet"
    pq.write_table(pa.table({"ctx": ["A man is", "She"], "endings": [["a", "b", "c", "d"]] * 2,
                             "label": ["3", "0"], "ind": [1, 2]}), p)
    rows = load_hellaswag(p)
    assert rows[0] == {"ctx": "A man is", "endings": ["a", "b", "c", "d"], "label": 3} and rows[1]["label"] == 0


def test_accuracy_in_unit_interval_with_real_tiny_model(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    m = Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size))
    rows = [{"ctx": "The quick brown", "endings": ["fox", "dog", "cat", "x"], "label": 0}] * 3
    acc = hellaswag_accuracy(m, tok, rows, "cpu", max_len=64, limit=2)
    assert 0.0 <= acc <= 1.0
