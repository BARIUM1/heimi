import pytest

from minids.official.codebench import first_code_line
from minids.official.completers import heimi_infill, heimi_next_line
from tests.helpers_data import tiny_tokenizer
from tests.helpers_official import PieceTok, RankedModel
from tests.helpers_train import ScriptedModel

PIECES = ["\n", "\n    ", "\n\n", " ", "    ", "def f():", "def f(a, b):", "x = 1", "y = 2", "# note", "return a + b", "print(f(1, 2))", "a = 1", "p"]


@pytest.fixture
def tok():
    return PieceTok(PIECES)


def _model(tok, script, seq_len=64):
    """script 的每一项是这一步想写的东西，越靠前越想写（写成文本，这里换成 id）。"""
    return RankedModel([[tok.id(p) if isinstance(p, str) else p for p in step] for step in script], tok.vocab_size, seq_len)


def test_next_line_lets_the_model_write_the_trailing_newline_itself(tok):
    # 模型最想写 "x = 1"，但提示的下一个字符是换行：只能在以换行开头的 token 里挑，它挑了"换行 + 缩进"
    m = _model(tok, [["x = 1", "\n    ", "\n"], ["x = 1"], ["\n"], ["y = 2"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=8)("def f():\n")
    assert m.seen[0] == tok.encode("def f():")  # 提示里不带结尾的换行
    assert text == "    x = 1\n"  # 返回的是接在原提示（含换行）后面的文本
    assert len(m.seen) == 3  # 第一行代码写完就停，没有把预算用完


def test_next_line_skips_comments_before_the_first_code_line(tok):
    m = _model(tok, [["\n"], ["# note"], ["\n"], ["x = 1"], ["\n"], ["y = 2"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=40)("def f():\n")
    assert text == "# note\nx = 1\n" and first_code_line(text) == ("x = 1", True)
    assert len(m.seen) == 5


def test_a_longer_tail_can_be_written_in_several_tokens(tok):
    # 结尾是一个空行：先写一个换行，再写"换行 + 缩进"；第二步模型最想写的 "x = 1" 仍然不许
    m = _model(tok, [["\n"], ["x = 1", "\n    "], ["x = 1"], ["\n"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=8)("def f():\n\n")
    assert m.seen[0] == tok.encode("def f():") and m.seen[1] == tok.encode("def f():\n")
    assert text == "    x = 1\n"


def test_next_line_keeps_only_the_tail_of_a_long_prompt(tok):
    m = _model(tok, [["\n"], ["x = 1"], ["\n"]], seq_len=32)
    heimi_next_line(m, tok, "cpu", max_new_tokens=8)("a = 1\n" * 200)
    assert len(m.seen[0]) == 32 - 8 and max(len(s) for s in m.seen) <= 32


def test_generation_budget_never_exceeds_half_the_context(tok):
    # 默认 64 个新 token 对上下文只有 32 的迷你模型太多：预算收到一半，写换行的那个 token 也算在预算里
    m = _model(tok, [["\n"], ["a = 1"]], seq_len=32)
    heimi_next_line(m, tok, "cpu")("a = 1\n" * 200)
    assert len(m.seen[0]) == 16 and len(m.seen) == 16


def test_a_prompt_without_trailing_whitespace_is_continued_as_is(tmp_path):
    real = tiny_tokenizer(tmp_path)
    body = real.encode("x = 1")
    assert heimi_next_line(ScriptedModel(body + [0] + body), real, "cpu", 40)("p") == "x = 1"
    assert heimi_next_line(ScriptedModel([3] + body), real, "cpu", 40)("p") == ""  # 第一个 token 就是控制标记：什么都没写


def test_a_special_token_right_after_the_tail_gives_empty_text(tok):
    assert heimi_next_line(_model(tok, [["\n"], [0], ["x = 1"]]), tok, "cpu", 8)("def f():\n") == ""


def test_a_tail_that_does_not_fit_the_budget_gives_empty_text(tok):
    m = _model(tok, [["\n"]])
    assert heimi_next_line(m, tok, "cpu", max_new_tokens=2)("x = 1\n\n\n") == ""
    assert len(m.seen) == 2  # 两个 token 没写完三个换行：不再往下生成


def test_a_prompt_that_is_only_whitespace_starts_from_the_document_start_token(tok):
    m = _model(tok, [["\n"], ["x = 1"], ["\n"]])
    assert heimi_next_line(m, tok, "cpu", 8)("\n") == "x = 1\n"
    assert m.seen[0] == [0]


def test_infill_fim_mode_builds_the_fim_prompt_within_the_context(tok):
    m = _model(tok, [["\n    "], ["return a + b"], ["\n"], ["y = 2"]], seq_len=64)
    prefix, suffix = "def f(a, b):\n" * 30, "\nprint(f(1, 2))\n" * 30
    out = heimi_infill(m, tok, "cpu", "fim", max_new_tokens=16)(prefix, suffix)
    assert out == "    return a + b" and max(len(s) for s in m.seen) <= 64
    # 余量 45 个 token：前缀占三分之二，后缀占三分之一；前缀结尾的换行不在提示里
    assert m.seen[0] == [1] + tok.encode(prefix.rstrip("\n"))[-30:] + [3] + tok.encode(suffix)[:15] + [2]
    assert len(m.seen) == 3  # 写到第一个换行就停


def test_infill_prefix_mode_ignores_the_suffix(tok):
    m = _model(tok, [["\n    "], ["return a + b"], ["\n"]])
    out = heimi_infill(m, tok, "cpu", "prefix", max_new_tokens=16)("def f(a, b):\n", "SUFFIX")
    assert out == "    return a + b" and m.seen[0] == tok.encode("def f(a, b):")


def test_infill_returns_an_empty_line_when_the_model_starts_with_a_blank_line(tok):
    # 模型在换行后面又写了一个换行：第一行就是空的，照常记为答错
    m = _model(tok, [["\n\n"], ["return a + b"]])
    assert heimi_infill(m, tok, "cpu", "prefix", 16)("def f(a, b):\n", "") == ""
    assert len(m.seen) == 1


def test_unknown_infill_mode_is_refused(tok):
    with pytest.raises(ValueError, match="mode"):
        heimi_infill(ScriptedModel([0]), tok, "cpu", "both")


@pytest.mark.parametrize("mode", ["fim", "prefix"])
def test_infill_special_first_token_gives_empty_text(tok, mode):
    # 没训过 FIM 的检查点：换行之后第一个 token 就是控制标记或 EOT，结果必须是不含任何标记字面串的空文本
    for first in (0, 1, 2, 3):
        out = heimi_infill(_model(tok, [["\n"], [first], ["x = 1"]]), tok, "cpu", mode, 16)("def f():\n", "\n")
        assert out == ""
