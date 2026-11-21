from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from minids.official.reference import GPT2_ID, evaluate_reference, hf_infill_prefix, hf_next_line  # noqa: E402
from minids.official.run import evaluate_checkpoint  # noqa: E402
from minids.train.checkpoint import CheckpointError  # noqa: E402
from tests.helpers_official import PieceTok, ranked_logits, write_code_data  # noqa: E402

PIECES = ["\n", "\n    ", "\n\n", " ", "    ", "def f():", "def f(a, b):", "x = 1", "y = 2", "a = 1", "return a + b", "return x", "import os", "# Path: ", "m0.py", "m1.py", "m2.py", "p"]


@pytest.fixture
def ckpt(dash_runs):
    return dash_runs[0] / "heimi" / "ckpt" / "smoke0_final.pt"


def test_checkpoint_record_carries_training_state_and_every_metric_group(ckpt, tmp_path):
    calls = []

    def fake_harness(lm, limit=None, **kwargs):
        calls.append((type(lm).__name__, limit))
        return {"blimp_acc": 0.5, "hellaswag_acc_norm": 0.25, "wikitext_bits_per_byte": 2.0}

    rec = evaluate_checkpoint(ckpt, "cpu", write_code_data(tmp_path / "eval"), limit=7, code_limit=2, harness=fake_harness)
    assert (rec.checkpoint, rec.stage, rec.global_step, rec.tokens_total) == ("smoke0_final.pt", "smoke0", 2, 512)
    assert rec.limit == 7 and rec.code_limit == 2 and rec.params_non_embedding > 0 and rec.harness_version == "0.4.13"
    assert rec.code_method == "healed"
    assert calls == [("HeimiLM", 7)]
    m = rec.metrics
    assert m["blimp_acc"] == 0.5 and m["repobench_n"] == 2 and m["infill_n"] == 2
    assert all(0.0 <= m[k] <= 1.0 for k in ("repobench_em", "infill_em_fim", "infill_em_prefix")) and 0.0 <= m["repobench_es"] <= 100.0


def test_code_benchmarks_can_be_skipped_and_then_need_no_data(ckpt, tmp_path):
    rec = evaluate_checkpoint(ckpt, "cpu", tmp_path / "missing", code_limit=0, harness=lambda lm, **kw: {"blimp_acc": 0.5})
    assert rec.metrics == {"blimp_acc": 0.5} and rec.code_limit == 0


def test_missing_code_data_is_reported_by_file_name(ckpt, tmp_path):
    with pytest.raises(FileNotFoundError, match="repobench_python_in_file.parquet"):
        evaluate_checkpoint(ckpt, "cpu", tmp_path / "missing", harness=lambda lm, **kw: {})


def test_checkpoint_without_ladder_config_is_a_named_checkpoint_error(ckpt, tmp_path):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    del payload["ladder_config"]
    bad = tmp_path / "bad_final.pt"
    torch.save(payload, bad)
    with pytest.raises(CheckpointError, match="bad_final.pt.*内容不完整"):
        evaluate_checkpoint(bad, "cpu", tmp_path / "missing", code_limit=0, harness=lambda lm, **kw: {})


class StubTok:
    """像 transformers 的分词器那样说话：id 0 是 EOS（像 GPT-2 的 <|endoftext|>）。"""

    eos_token_id = 0

    def __init__(self, tok: PieceTok):
        self.tok = tok

    def __len__(self) -> int:
        return self.tok.vocab_size

    def encode(self, text):
        return self.tok.encode(text)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        assert clean_up_tokenization_spaces is False  # 文本要和 token 一一对得上，不能让分词器"清理"空格
        return self.tok.decode([i for i in ids if not (skip_special_tokens and i == 0)])


class StubGPT2:
    """像 transformers 的模型那样说话：调用它返回带 .logits 的对象，generate 每步取 argmax、写出 EOS 就停。

    打分见 ranked_logits，新位置的序号从第一次收到的序列长度算起（所以一个替身只答一道题）；seen 记下每次收到的序列。
    script 也可以是字典 {上一个 token: 想写的 token 列表}，键 None 管其余情况：这样不记位置，可以连着答很多道题。
    """

    def __init__(self, vocab_size: int, script, n_positions: int = 32):
        self.vocab_size, self.script = vocab_size, script
        self.config = SimpleNamespace(n_positions=n_positions)
        self.start = None
        self.seen: list[list[int]] = []
        self.generate_kwargs: dict | None = None

    def _next(self, input_ids):
        if isinstance(self.script, dict):
            ranked = self.script.get(int(input_ids[0, -1]), self.script[None])
            return ranked_logits([ranked], 0, self.vocab_size)
        if self.start is None:
            self.start = input_ids.shape[1]
        return ranked_logits(self.script, input_ids.shape[1] - self.start, self.vocab_size)

    def __call__(self, input_ids):
        self.seen.append(input_ids[0].tolist())
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), -1.0)
        logits[0, -1] = self._next(input_ids)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, **kwargs):
        self.generate_kwargs = kwargs
        if kwargs["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be greater than 0")  # 真的 generate 也这样拒绝
        self.seen.append(input_ids[0].tolist())
        for _ in range(kwargs["max_new_tokens"]):
            t = int(self._next(input_ids).argmax())
            input_ids = torch.cat([input_ids, torch.tensor([[t]], dtype=input_ids.dtype)], dim=1)
            if t == 0:
                break
        return input_ids


def _stub_hf(script, n_positions=32):
    """script 的每一项是这一步想写的东西，越靠前越想写（文本或 id）。"""
    tok = PieceTok(PIECES)
    ids = [[tok.id(p) if isinstance(p, str) else p for p in step] for step in script]
    return StubGPT2(tok.vocab_size, ids, n_positions), StubTok(tok)


def test_hf_completers_let_the_model_rewrite_the_trailing_newline_and_return_what_follows_the_prompt():
    # 模型最想写 "x = 1"，但被拿掉的 token 是换行：只能在以换行开头的 token 里挑
    script = [["x = 1", "\n    ", "\n"], ["x = 1"], ["\n"], ["y = 2"], ["\n"], [0]]
    hf_model, hf_tok = _stub_hf(script)
    prompt = "a = 1\n" * 100
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)(prompt) == "    x = 1\ny = 2\n"
    window = hf_tok.encode(prompt)[-(32 - 8) :]  # 和旧方式截到同一个位置
    assert hf_model.seen == [window[:-1], window[:-1] + [hf_tok.tok.id("\n    ")]]  # 先问一次换行怎么写，再从那里接着生成
    assert hf_model.generate_kwargs["do_sample"] is False and hf_model.generate_kwargs["max_new_tokens"] == 8  # 重写的 token 不占预算

    hf_model, hf_tok = _stub_hf(script)
    assert hf_infill_prefix(hf_model, hf_tok, "cpu", max_new_tokens=8)("def f():\n", "IGNORED") == "    x = 1"
    assert hf_model.seen[0] == hf_tok.encode("def f():")


def test_hf_writing_the_same_token_back_generates_from_the_original_prompt():
    # 模型把换行原样写回：generate 收到的就是原来的整段提示、原来的预算，和不做任何处理时一样
    hf_model, hf_tok = _stub_hf([["x = 1", "\n"], ["x = 1"], ["\n"], [0]])
    prompt = "a = 1\n" * 100
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)(prompt) == "x = 1\n"
    assert hf_model.seen[1] == hf_tok.encode(prompt)[-(32 - 8) :] and hf_model.generate_kwargs["max_new_tokens"] == 8


def test_hf_blank_line_token_can_only_be_replaced_by_tokens_that_start_with_it():
    # 提示以"空一行"这个 token 结尾：单独一个换行、"换行 + 缩进"都对不上
    hf_model, hf_tok = _stub_hf([["\n", "\n    ", "\n\n"], ["x = 1"], ["\n"], [0]])
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)("x = 1\n\n") == "x = 1\n"
    assert hf_model.seen == [hf_tok.encode("x = 1"), hf_tok.encode("x = 1\n\n")]


def test_hf_prompt_without_trailing_whitespace_goes_straight_to_generation():
    hf_model, hf_tok = _stub_hf([["x = 1"], ["\n"], [0]])
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)("p") == "x = 1\n"
    assert hf_model.seen == [hf_tok.encode("p")] and hf_model.generate_kwargs["max_new_tokens"] == 8


def test_hf_completers_never_return_special_tokens_and_survive_empty_output():
    # 换行之后第一个新 token 就是 EOS：返回的文本里不含特殊 token，不崩
    for make in (lambda m, t: hf_next_line(m, t, "cpu", max_new_tokens=8), lambda m, t: (lambda p: hf_infill_prefix(m, t, "cpu", max_new_tokens=8)(p, ""))):
        hf_model, hf_tok = _stub_hf([["\n"], [0]])
        assert make(hf_model, hf_tok)("def f():\n") == ""


def test_hf_prompt_that_is_one_whitespace_token_starts_from_eos():
    hf_model, hf_tok = _stub_hf([["\n"], ["x = 1"], ["\n"], [0]])
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)("\n") == "x = 1\n"
    assert hf_model.seen[0] == [0]


def test_reference_entry_has_no_fim_number_and_runs_the_harness_on_gpt2(tmp_path):
    tok = PieceTok(PIECES)
    # 每道题都这样答：先写"换行 + 缩进"，再写 return x，换行，结束
    script = {None: [tok.id("\n    ")], tok.id("\n    "): [tok.id("return x")], tok.id("return x"): [tok.id("\n")], tok.id("\n"): [0]}
    hf_model, hf_tok = StubGPT2(tok.vocab_size, script, n_positions=64), StubTok(tok)
    calls = []

    def fake_harness(model, limit=None, **kwargs):
        calls.append((model, limit, kwargs.get("model_args")))
        return {"blimp_acc": 0.8}

    def loader(device):
        return hf_model, hf_tok, 85_000_000

    entry = evaluate_reference("cpu", write_code_data(tmp_path / "eval"), limit=5, code_limit=2, harness=fake_harness, loader=loader)
    assert calls == [("hf", 5, f"pretrained={GPT2_ID},dtype=float32")]
    assert (entry["label"], entry["model_id"], entry["params_non_embedding"], entry["limit"], entry["code_limit"]) == ("GPT-2 small", GPT2_ID, 85_000_000, 5, 2)
    m = entry["metrics"]
    assert m["blimp_acc"] == 0.8 and m["repobench_n"] == 2 and "infill_em_prefix" in m and "infill_em_fim" not in m
    assert m["repobench_em"] == 1.0  # 迷你数据的下一行正是 "    return x"：整条链路（去掉结尾换行 → 模型写回 → 取行 → 计分）是通的
    assert entry["harness_version"] == "0.4.13" and entry["time"] and entry["code_method"] == "healed"
