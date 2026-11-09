from minids.completion import complete_middle, fim_prompt_ids, greedy_line
from tests.helpers_data import tiny_tokenizer
from tests.helpers_train import ScriptedModel


def test_prompt_layout_and_truncation(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    prefix, suffix = "abc def ghi", "jkl mno pqr"
    ids = fim_prompt_ids(tok, prefix, suffix, max_prefix_tokens=2, max_suffix_tokens=1)
    p_ids, s_ids = tok.encode(prefix), tok.encode(suffix)
    assert ids == [1] + p_ids[-2:] + [3] + s_ids[:1] + [2]


def test_greedy_line_stops_at_newline_and_strips_it(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    script = tok.encode("return a + b") + tok.encode("\n") + tok.encode("garbage")
    assert greedy_line(ScriptedModel(script), tok, [1, 2], 32, "cpu") == "return a + b"


def test_greedy_line_stops_at_eot_and_at_budget(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    body = tok.encode("x = 1")
    assert greedy_line(ScriptedModel(body + [0] + body), tok, [1], 32, "cpu") == "x = 1"
    assert greedy_line(ScriptedModel(body * 10), tok, [1], len(body), "cpu") == "x = 1"


def test_complete_middle_wires_prompt_and_generation(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    out = complete_middle(ScriptedModel(tok.encode("pass") + tok.encode("\n")), tok, "def f():\n    ", "\n", 8, "cpu", 4, 4)
    assert out == "pass"
