import pytest

from minids.completion import complete_middle, greedy_line
from minids.official.codebench import first_code_line
from minids.official.completers import heimi_infill, heimi_next_line
from tests.helpers_data import tiny_tokenizer
from tests.helpers_official import PieceTok, RankedModel
from tests.helpers_train import ScriptedModel

PIECES = [
    "\n", "\n    ", "\n\n", "\n\n    ", " ", "    ", "def f():", "def f(a, b):", "x = 1", "y = 2", "# note", "return a + b",
    "print(f(1, 2))", "a = 1", "p",
]


@pytest.fixture
def tok():
    return PieceTok(PIECES)


def _model(tok, script, seq_len=64):
    """script 的每一项是这一步想写的东西，越靠前越想写（写成文本，这里换成 id）。"""
    return RankedModel([[tok.id(p) if isinstance(p, str) else p for p in step] for step in script], tok.vocab_size, seq_len)


def test_next_line_lets_the_model_rewrite_the_trailing_newline_token(tok):
    # 模型最想写 "x = 1"，但被拿掉的 token 是换行：只能在以换行开头的 token 里挑，它挑了"换行 + 缩进"
    m = _model(tok, [["x = 1", "\n    ", "\n"], ["x = 1"], ["\n"], ["y = 2"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=8)("def f():\n")
    assert m.seen[0] == tok.encode("def f():")  # 提示里不带结尾的换行
    assert text == "    x = 1\n"  # 返回的是接在原提示（含换行）后面的文本
    assert len(m.seen) == 3  # 第一行代码写完就停，没有把预算用完


def test_writing_the_same_token_back_continues_exactly_from_the_original_prompt(tok):
    # 模型把换行原样写回：之后它看到的序列就是原来的整段提示，和不做任何处理时一样
    m = _model(tok, [["x = 1", "\n"], ["    "], ["x = 1"], ["\n"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=8)("def f():\n")
    assert m.seen[0] == tok.encode("def f():") and m.seen[1] == tok.encode("def f():\n")
    assert text == "    x = 1\n"


def test_next_line_skips_comments_before_the_first_code_line(tok):
    m = _model(tok, [["\n"], ["# note"], ["\n"], ["x = 1"], ["\n"], ["y = 2"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=40)("def f():\n")
    assert text == "# note\nx = 1\n" and first_code_line(text) == ("x = 1", True)
    assert len(m.seen) == 5


def test_a_blank_line_token_can_only_be_replaced_by_tokens_that_start_with_it(tok):
    # 提示以"空一行"这个 token 结尾：单独一个换行、"换行 + 缩进"都对不上，模型在对得上的里面挑
    m = _model(tok, [["\n", "\n    ", "\n\n    ", "\n\n"], ["x = 1"], ["\n"]])
    text = heimi_next_line(m, tok, "cpu", max_new_tokens=8)("def f():\n\n")
    assert m.seen[0] == tok.encode("def f():") and m.seen[1] == tok.encode("def f():\n\n    ")
    assert text == "    x = 1\n"


def test_only_the_last_whitespace_token_is_rewritten(tok):
    # 结尾三个换行被切成 "\n\n" 和 "\n" 两个 token：只拿掉最后一个，前一个留在提示里
    m = _model(tok, [["\n    "], ["x = 1"], ["\n"]])
    assert heimi_next_line(m, tok, "cpu", max_new_tokens=8)("x = 1\n\n\n") == "    x = 1\n"
    assert m.seen[0] == tok.encode("x = 1\n\n")


def test_next_line_keeps_only_the_tail_of_a_long_prompt(tok):
    m = _model(tok, [["\n"], ["x = 1"], ["\n"]], seq_len=32)
    heimi_next_line(m, tok, "cpu", max_new_tokens=8)("a = 1\n" * 200)
    # 和旧方式截到同一个位置，再拿掉最后那个换行
    assert m.seen[0] == tok.encode("a = 1\n" * 200)[-(32 - 8) : -1] and max(len(s) for s in m.seen) <= 32


def test_generation_budget_never_exceeds_half_the_context(tok):
    # 默认 64 个新 token 对上下文只有 32 的迷你模型太多：预算收到一半（16）；重写换行的那个 token 不占预算
    m = _model(tok, [["\n"], ["a = 1"]], seq_len=32)
    heimi_next_line(m, tok, "cpu")("a = 1\n" * 200)
    assert len(m.seen[0]) == 15 and len(m.seen) == 1 + 16 and max(len(s) for s in m.seen) == 31


def test_a_prompt_without_trailing_whitespace_is_continued_as_is(tmp_path):
    real = tiny_tokenizer(tmp_path)
    body = real.encode("x = 1")
    assert heimi_next_line(ScriptedModel(body + [0] + body), real, "cpu", 40)("p") == "x = 1"
    assert heimi_next_line(ScriptedModel([3] + body), real, "cpu", 40)("p") == ""  # 第一个 token 就是控制标记：什么都没写


def test_nothing_rewritten_means_exactly_the_budget(tmp_path):
    # 提示不以空白结尾：没有重写的 token，新 token 正好是预算那么多，一个不多
    real = tiny_tokenizer(tmp_path)
    m = ScriptedModel(real.encode("a") * 100, seq_len=32)
    heimi_next_line(m, real, "cpu")("p")
    assert m.pos == 16


def test_infill_without_trailing_whitespace_matches_the_projects_own_completion(tmp_path):
    # 前缀不以空白结尾时不做任何处理：结果和项目自有评测用的 greedy_line / complete_middle 一样
    real = tiny_tokenizer(tmp_path)
    script = real.encode("x = 1\ny = 2\n")
    prefix, suffix = "def f():\n    a", "\nprint(a)\n"
    budget, room = 16, 64 - 16 - 3
    got = heimi_infill(ScriptedModel(script), real, "cpu", "prefix", 16)(prefix, suffix)
    assert got == greedy_line(ScriptedModel(script), real, real.encode(prefix), budget, "cpu") == "x = 1"
    got = heimi_infill(ScriptedModel(script), real, "cpu", "fim", 16)(prefix, suffix)
    assert got == complete_middle(ScriptedModel(script), real, prefix, suffix, budget, "cpu", room * 2 // 3, room - room * 2 // 3) == "x = 1"


def test_the_real_tokenizer_goes_through_the_same_rewriting(tmp_path):
    real = tiny_tokenizer(tmp_path)
    ids = real.encode("x = 1\n")
    assert real.decode([ids[-1]]) == "\n"
    m = RankedModel([[i] for i in [ids[-1], *real.encode("y = 2\n")]], real.vocab_size)
    assert heimi_next_line(m, real, "cpu", max_new_tokens=16)("x = 1\n") == "y = 2\n"
    assert m.seen[0] == ids[:-1]


def test_a_special_token_right_after_the_rewritten_token_stops_generation(tok):
    assert heimi_next_line(_model(tok, [["\n"], [0], ["x = 1"]]), tok, "cpu", 8)("def f():\n") == ""
    # 重写时连缩进一起写了，然后就是控制标记：返回的只有那段缩进，没有任何标记的字面串
    assert heimi_next_line(_model(tok, [["\n    "], [0], ["x = 1"]]), tok, "cpu", 8)("def f():\n") == "    "


def test_a_prompt_that_is_one_whitespace_token_starts_from_the_document_start_token(tok):
    m = _model(tok, [["\n"], ["x = 1"], ["\n"]])
    assert heimi_next_line(m, tok, "cpu", 8)("\n") == "x = 1\n"
    assert m.seen[0] == [0]


def test_infill_fim_mode_builds_the_fim_prompt_within_the_context(tok):
    m = _model(tok, [["\n    "], ["return a + b"], ["\n"], ["y = 2"]], seq_len=64)
    prefix, suffix = "def f(a, b):\n" * 30, "\nprint(f(1, 2))\n" * 30
    out = heimi_infill(m, tok, "cpu", "fim", max_new_tokens=16)(prefix, suffix)
    assert out == "    return a + b" and max(len(s) for s in m.seen) <= 64
    # 余量 45 个 token：前缀占三分之二（30），后缀占三分之一（15）；前缀段最后那个换行不在提示里
    assert m.seen[0] == [1] + tok.encode(prefix)[-30:-1] + [3] + tok.encode(suffix)[:15] + [2]
    assert len(m.seen) == 3  # 写到第一个换行就停


def test_infill_fim_mode_with_an_empty_prefix_rewrites_nothing(tok):
    m = _model(tok, [["x = 1"], ["\n"]])
    assert heimi_infill(m, tok, "cpu", "fim", 16)("", "\ny = 2\n") == "x = 1"
    assert m.seen[0] == [1, 3] + tok.encode("\ny = 2\n") + [2]


def test_infill_prefix_mode_ignores_the_suffix(tok):
    m = _model(tok, [["\n    "], ["return a + b"], ["\n"]])
    out = heimi_infill(m, tok, "cpu", "prefix", max_new_tokens=16)("def f(a, b):\n", "SUFFIX")
    assert out == "    return a + b" and m.seen[0] == tok.encode("def f(a, b):")


def test_infill_prefix_mode_truncates_and_rewrites_like_next_line(tok):
    m = _model(tok, [["\n"], ["x = 1"], ["\n"]], seq_len=32)
    heimi_infill(m, tok, "cpu", "prefix", max_new_tokens=8)("a = 1\n" * 200, "")
    assert m.seen[0] == tok.encode("a = 1\n" * 200)[-(32 - 8) : -1]  # 和旧方式截到同一个位置，再拿掉最后那个换行
    # 结尾的空白是两个 token：只重写最后一个
    m = _model(tok, [["\n    "], ["x = 1"], ["\n"]])
    assert heimi_infill(m, tok, "cpu", "prefix", 8)("x = 1\n\n\n", "") == "    x = 1"
    assert m.seen[0] == tok.encode("x = 1\n\n")


def test_infill_returns_an_empty_line_when_the_model_starts_with_a_blank_line(tok):
    # 模型把换行重写成了"空一行"：第一行就是空的，照常记为答错
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
