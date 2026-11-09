import pytest

from minids.data.tokenizer import SPECIAL_TOKENS, HeimiTokenizer, iter_training_corpus
from tests.helpers_data import tiny_tokenizer


def test_special_token_ids_are_fixed(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS[:5]] == [0, 1, 2, 3, 4]
    assert tok.token_to_id("<|reserved_15|>") == 15


def test_round_trip_arbitrary_text(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    s = "def f(x):\n\treturn x  # 中文 ünïcode 🙂 \x00\x7f"
    assert tok.decode(tok.encode(s)) == s


def test_literal_special_string_is_not_a_special_id(tmp_path):
    # 数据里字面出现的控制串必须当普通文本，否则一段文本就能往训练流里插控制 token
    tok = tiny_tokenizer(tmp_path)
    ids = tok.encode("see <|endoftext|> here")
    assert 0 not in ids and tok.decode(ids) == "see <|endoftext|> here"


def test_decode_single_special_id(tmp_path):
    assert tiny_tokenizer(tmp_path).decode([1]) == "<|fim_prefix|>"


def test_vocab_size_is_requested(tmp_path):
    assert tiny_tokenizer(tmp_path, vocab_size=300).vocab_size == 300


def test_encode_batch_matches_encode(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    assert tok.encode_batch(["a b", "c"]) == [tok.encode("a b"), tok.encode("c")]


def test_loaded_tokenizer_keeps_literal_special_strings_plain(tmp_path):
    # 保存的 json 里不含这条规则，直接 Tokenizer.from_file 会把字面控制串识别成 id 0..15；
    # 这里走 prepare 与训练实际使用的 HeimiTokenizer.load 路径，16 个特殊串都必须保持普通文本
    tiny_tokenizer(tmp_path)
    tok = HeimiTokenizer.load(tmp_path / "tok300.json")
    for special in SPECIAL_TOKENS:
        text = f"a {special} b"
        ids = tok.encode(text)
        assert min(ids) >= 16 and tok.decode(ids) == text


def test_training_corpus_requires_val_rows_for_every_source(tmp_path):
    # val_rows 缺键必须报错，而不是默认 0 行把验证集悄悄混进分词器语料
    with pytest.raises(KeyError, match="web"):
        next(iter_training_corpus(tmp_path, {"code": 0}))
