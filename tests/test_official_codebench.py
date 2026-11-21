import pytest

from minids.official.codebench import (
    edit_similarity, first_code_line, infill_scores, infill_suffix, lines_match, repobench_prompt, repobench_scores, sample_rows,
)

ROW = {
    "repo_name": "o/r", "file_path": "pkg/mod.py", "import_statement": "import os\nimport sys",
    "cropped_code": "def f():\n    x = 1\n\n\n\n\n    return x\n  \n", "next_line": "print(f())", "level": "2k",
}


def test_first_code_line_skips_blank_lines_and_comments():
    assert first_code_line("\n\n# a comment\n    x = 1\ny = 2\n") == ("    x = 1", True)


def test_first_code_line_skips_a_docstring_block():
    assert first_code_line('"""doc\nmore\n"""\nreturn 1\n') == ("return 1", True)


def test_a_one_line_docstring_opens_a_block_as_in_the_official_script():
    # 官方脚本只看行首的三引号就进入"多行注释"状态，不看同一行有没有收尾；照做才是同一把尺子
    assert first_code_line('"""one line"""\nx = 1\n"""\nz = 3\n') == ("z = 3", True)


def test_a_line_still_being_written_is_reported_as_incomplete():
    assert first_code_line("# note\nx = compute(") == ("x = compute(", False)
    assert first_code_line("x = 1\n") == ("x = 1", True)


def test_text_with_no_code_line_falls_back_to_its_first_line():
    assert first_code_line("# only a comment\n# another\n") == ("# only a comment", False)
    assert first_code_line("") == ("", False)


def test_edit_similarity_follows_fuzzywuzzy_ratio():
    assert edit_similarity("x = 1", "x = 1") == 100 and edit_similarity("", "") == 100
    assert edit_similarity("", "x = 1") == 0 and edit_similarity("x = 1", "") == 0
    assert edit_similarity("abef", "abcd") == 50  # 2 * 2 / 8
    assert edit_similarity("return a - b", "return a + b") == 92  # 2 * 11 / 24 = 0.9167


def test_lines_match_ignores_whitespace_differences_only():
    assert lines_match("  return a+b ", "return a+b") and lines_match("x  =  1", "x = 1")
    assert not lines_match("return a + b", "return a+b")


def test_prompt_is_the_official_in_file_part_with_blank_runs_collapsed():
    assert repobench_prompt(ROW) == "# Path: pkg/mod.py\nimport os\nimport sys\ndef f():\n    x = 1\n\n    return x\n"


def test_sampling_is_fixed_keeps_order_and_none_means_everything():
    rows = [{"i": i} for i in range(50)]
    a = sample_rows(rows, 10)
    assert a == sample_rows(rows, 10) and len(a) == 10 and a != rows[:10]
    assert [r["i"] for r in a] == sorted(r["i"] for r in a)
    assert sample_rows(rows, None) == rows and sample_rows(rows, 80) == rows


def test_repobench_scores_use_the_first_code_line_of_what_the_model_wrote():
    rows = [dict(ROW, next_line="print(f())"), dict(ROW, next_line="return 2"), dict(ROW, next_line="y = 3")]
    answers = iter(["\n# thinking\n  print(f())  \nmore\n", "return 3\n", ""])
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return next(answers)

    # 第 1 条：EM 命中，ES = round(100 * 20 / 24) = 83；第 2 条：ES = round(87.5) = 88；第 3 条：什么都没写，0
    assert repobench_scores(complete, rows) == {"em": 0.33333, "es": 57.0, "n": 3}
    assert seen == [repobench_prompt(r) for r in rows]


def test_infill_compares_the_first_generated_line_with_the_reference_line():
    rows = [
        {"task_id": "t0", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n"},
        {"task_id": "t1", "prompt": "p", "suffix": "s", "canonical_solution": "    x = 1\n"},
    ]
    calls = []

    def complete(prefix, suffix):
        calls.append((prefix, suffix))
        return ["return a + b\n    junk", "x = 2"][len(calls) - 1]

    assert infill_scores(complete, rows) == {"em": 0.5, "n": 2}
    # 待填那一行行末的换行交到后缀开头
    assert calls == [("def f(a, b):\n", "\n\n"), ("p", "\ns")]


def test_infill_suffix_takes_the_line_ending_of_the_missing_line():
    row = {"prompt": "def f():\n", "canonical_solution": "    x = 1\n", "suffix": "    return x\n"}
    assert infill_suffix(row) == "\n    return x\n"
    # 整篇文本一个字符都没变：提示 + 答案（去掉行末换行）+ 新后缀 == 提示 + 答案 + 原后缀
    assert row["prompt"] + row["canonical_solution"][:-1] + infill_suffix(row) == row["prompt"] + row["canonical_solution"] + row["suffix"]
    assert infill_suffix(dict(row, suffix="")) == "\n"
    assert infill_suffix({"suffix": "tail"}) == "\ntail"  # 不看答案：答案的格式由读数据的地方检查


def test_scoring_nothing_is_an_error():
    with pytest.raises(ValueError):
        repobench_scores(lambda p: "", [])
    with pytest.raises(ValueError):
        infill_scores(lambda p, s: "", [])
