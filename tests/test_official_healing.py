from minids.official.healing import allowed_ids, split_tail, whitespace_tokens, write_tail


def test_split_tail_takes_the_whole_trailing_whitespace_run():
    assert split_tail("def f():\n") == ("def f():", "\n")
    assert split_tail('    """doc"""\n\n') == ('    """doc"""', "\n\n")
    assert split_tail("x = 1 \n    \n") == ("x = 1", " \n    \n")
    assert split_tail("x = ") == ("x =", " ")
    assert split_tail("x") == ("x", "") and split_tail("") == ("", "") and split_tail(" \n") == ("", " \n")
    # 只认空格、制表符、回车、换行；别的空白字符留在提示里
    assert split_tail("x ") == ("x ", "")


def test_whitespace_tokens_keep_only_tokens_whose_text_starts_with_whitespace():
    texts = ["a", "\n", " b", "", "\n    ", "<|endoftext|>", "\t", "�"]
    assert whitespace_tokens(texts) == {1: "\n", 2: " b", 4: "\n    ", 6: "\t"}


def test_allowed_ids_fit_inside_the_tail_or_run_past_it():
    ws = {1: "\n", 2: " b", 4: "\n    ", 5: "\n\n", 6: "\n\n  ", 7: " ", 8: "\n \n"}
    assert allowed_ids("\n", ws) == [1, 4, 5, 6, 8]  # 正好写完，或者写完还多带一截
    assert allowed_ids("\n\n", ws) == [1, 5, 6]  # 先写一半也行；"\n    " 和 "\n \n" 对不上
    assert allowed_ids(" ", ws) == [2, 7]
    assert allowed_ids("\t", ws) == []


def test_write_tail_lets_the_model_choose_among_the_allowed_tokens_until_the_tail_is_covered():
    ws = {1: "\n", 4: "\n    ", 5: " ", 6: " \n"}
    calls = []

    def last(written, allowed):
        calls.append((list(written), list(allowed)))
        return allowed[-1]

    assert write_tail(" \n", ws, last, 8) == [6]  # 一个 token 正好写完
    assert calls == [([], [5, 6])]
    calls.clear()
    assert write_tail(" \n", ws, lambda written, allowed: (calls.append((list(written), list(allowed))), allowed[0])[1], 8) == [5, 1]
    assert calls == [([], [5, 6]), ([5], [1, 4])]  # 分两步写：第二步看得到第一步写了什么
    assert write_tail("\n", ws, last, 8) == [4]  # 写完还多带了缩进：照样算写完


def test_write_tail_with_nothing_to_write_never_asks_the_model():
    def boom(written, allowed):
        raise AssertionError("不该来问")

    assert write_tail("", {1: "\n"}, boom, 8) == []


def test_write_tail_gives_up_when_the_tail_cannot_be_written():
    first = lambda written, allowed: allowed[0]  # noqa: E731
    assert write_tail("\n\n\n", {1: "\n"}, first, 2) is None  # 预算内写不完
    assert write_tail("\n\n", {1: "\n"}, first, 2) == [1, 1]
    assert write_tail("\t", {1: "\n"}, first, 8) is None  # 词表里没有对得上的 token
