from minids.official.healing import Healer, whitespace_tokens

#          0    1     2     3    4         5               6     7       8         9    10
TEXTS = ["a", "\n", " b", "", "\n    ", "<|endoftext|>", "\t", "\n\n", "\n\n  ", " ", "\n "]


def test_whitespace_tokens_keep_only_tokens_whose_text_starts_with_whitespace():
    assert whitespace_tokens(TEXTS + ["�"]) == {1: "\n", 2: " b", 4: "\n    ", 6: "\t", 7: "\n\n", 8: "\n\n  ", 9: " ", 10: "\n "}


def test_a_whitespace_only_last_token_is_taken_back_and_any_token_starting_with_it_may_replace_it():
    h = Healer(TEXTS)
    # 换行：可以原样写回（1），也可以连缩进或空行一起写（4、7、8、10）
    assert h.heal([0, 1]) == ([0], "\n", [1, 4, 7, 8, 10])
    assert h.heal([0, 7]) == ([0], "\n\n", [7, 8])  # 空一行：单独一个换行的 token 对不上，不在里面
    assert h.heal([0, 9]) == ([0], " ", [2, 9])
    assert h.heal([0, 6]) == ([0], "\t", [6])


def test_only_the_last_token_is_taken_back():
    h = Healer(TEXTS)
    assert h.heal([0, 1, 1]) == ([0, 1], "\n", [1, 4, 7, 8, 10])
    assert h.heal([1]) == ([], "\n", [1, 4, 7, 8, 10])


def test_a_last_token_that_is_not_pure_whitespace_is_left_alone():
    h = Healer(TEXTS)
    assert h.heal([1, 0]) == ([1, 0], "", [])
    assert h.heal([0, 2]) == ([0, 2], "", [])  # 空格 + 字母：不是纯空白
    assert h.heal([0, 10]) == ([0, 10], "", [])  # 只认空格、制表符、回车、换行；不间断空格不算
    assert h.heal([0, 5]) == ([0, 5], "", []) and h.heal([0, 3]) == ([0, 3], "", [])
    assert h.heal([]) == ([], "", [])


def test_heal_does_not_change_the_list_it_was_given():
    ids = [0, 1]
    Healer(TEXTS).heal(ids)
    assert ids == [0, 1]
