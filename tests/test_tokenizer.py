from minids.data.tokenizer import SPECIAL_TOKENS
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
