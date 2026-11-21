import pytest

from minids.official.codebench import first_code_line
from minids.official.completers import heimi_infill, heimi_next_line
from tests.helpers_data import tiny_tokenizer
from tests.helpers_official import SpyScripted
from tests.helpers_train import ScriptedModel


def test_next_line_stops_as_soon_as_a_code_line_is_complete(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    head = tok.encode("# note\n") + tok.encode("x = 1\n")
    m = ScriptedModel(head + tok.encode("y = 2\n" * 5))
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=40)("def f():\n")
    assert first_code_line(text) == ("x = 1", True)
    assert m.pos == len(head)  # 第一行代码写完就停，没有把预算用完


def test_next_line_keeps_only_the_tail_of_a_long_prompt(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("x = 1\n"), seq_len=32)
    heimi_next_line(m, tok, "cpu", max_new_tokens=8)("a = 1\n" * 200)
    assert m.lengths[0] == 32 - 8 and max(m.lengths) <= 32


def test_generation_budget_never_exceeds_half_the_context(tmp_path):
    # 默认 64 个新 token 对上下文只有 32 的迷你模型太多：预算收到一半，提示才不会被截成空的
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("a") * 100, seq_len=32)
    heimi_next_line(m, tok, "cpu")("a = 1\n" * 200)
    assert m.lengths[0] == 16 and m.pos == 16


def test_next_line_stops_at_a_special_token_and_never_returns_its_literal(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    body = tok.encode("x = 1")
    assert heimi_next_line(ScriptedModel(body + [0] + body), tok, "cpu", 40)("p") == "x = 1"
    assert heimi_next_line(ScriptedModel([3] + body), tok, "cpu", 40)("p") == ""  # 第一个 token 就是控制标记：什么都没写


def test_infill_fim_mode_builds_the_fim_prompt_within_the_context(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("return a + b\n"), seq_len=64)
    out = heimi_infill(m, tok, "cpu", "fim", max_new_tokens=16)("def f(a, b):\n" * 30, "\nprint(f(1, 2))\n" * 30)
    assert out == "return a + b" and max(m.lengths) <= 64
    assert m.first[0] == 1 and m.first[-1] == 2 and 3 in m.first  # <|fim_prefix|> … <|fim_suffix|> … <|fim_middle|>
    assert len(m.first) == 30 + 15 + 3  # 余量 45 个 token：前缀占三分之二，后缀占三分之一


def test_infill_prefix_mode_ignores_the_suffix(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    m = SpyScripted(tok.encode("return a + b\n"), seq_len=64)
    out = heimi_infill(m, tok, "cpu", "prefix", max_new_tokens=16)("def f(a, b):\n", "SUFFIX")
    assert out == "return a + b" and m.first == tok.encode("def f(a, b):\n")


def test_unknown_infill_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        heimi_infill(ScriptedModel([0]), tiny_tokenizer(tmp_path), "cpu", "both")


@pytest.mark.parametrize("mode", ["fim", "prefix"])
def test_infill_special_first_token_gives_empty_text(tmp_path, mode):
    # 没训过 FIM 的检查点：第一个 token 就是控制标记或 EOT，结果必须是不含任何标记字面串的空文本
    tok = tiny_tokenizer(tmp_path)
    for first in (0, 1, 2, 3):
        out = heimi_infill(ScriptedModel([first] + tok.encode("x = 1")), tok, "cpu", mode, 16)("def f():\n", "\n")
        assert out == ""
