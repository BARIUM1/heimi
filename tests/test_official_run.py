from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from minids.official.reference import GPT2_ID, evaluate_reference, hf_infill_prefix, hf_next_line  # noqa: E402
from minids.official.run import evaluate_checkpoint  # noqa: E402
from minids.train.checkpoint import CheckpointError  # noqa: E402
from tests.helpers_data import tiny_tokenizer  # noqa: E402
from tests.helpers_official import write_code_data  # noqa: E402


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


def _stub_hf(tmp_path, new_text="x = 1\ny = 2\n", lead_ids=()):
    """像 transformers 的模型与分词器那样说话的替身：generate 把固定的新 token 接在输入后面。"""
    tok = tiny_tokenizer(tmp_path)
    new = list(lead_ids) + tok.encode(new_text)
    seen = {}

    def generate(input_ids, **kwargs):
        seen["n"], seen["kwargs"] = input_ids.shape[1], kwargs
        return torch.cat([input_ids, torch.tensor([new], dtype=input_ids.dtype)], dim=1)

    def decode(ids, skip_special_tokens=True):
        ids = [i for i in ids if not (skip_special_tokens and i == 0)]  # id 0 是 EOT，像 GPT-2 的 <|endoftext|>
        return tok.decode(list(ids))

    hf_model = SimpleNamespace(config=SimpleNamespace(n_positions=32), generate=generate)
    hf_tok = SimpleNamespace(encode=tok.encode, decode=decode, eos_token_id=0)
    return hf_model, hf_tok, seen


def test_hf_completers_send_the_prompt_tail_and_return_only_the_new_text(tmp_path):
    hf_model, hf_tok, seen = _stub_hf(tmp_path)
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)("a = 1\n" * 100) == "x = 1\ny = 2\n"
    assert seen["n"] == 32 - 8 and seen["kwargs"]["do_sample"] is False and seen["kwargs"]["max_new_tokens"] == 8
    assert hf_infill_prefix(hf_model, hf_tok, "cpu", max_new_tokens=8)("def f():\n", "IGNORED") == "x = 1"


def test_hf_completers_never_return_special_tokens_and_survive_empty_output(tmp_path):
    # 第一个新 token 就是 EOT：返回的文本里不含特殊 token，不崩
    hf_model, hf_tok, _ = _stub_hf(tmp_path, new_text="", lead_ids=(0,))
    assert hf_next_line(hf_model, hf_tok, "cpu", max_new_tokens=8)("def f():\n") == ""
    assert hf_infill_prefix(hf_model, hf_tok, "cpu", max_new_tokens=8)("def f():\n", "") == ""
    # 文本中间夹着 EOT 也被去掉
    hf_model, hf_tok, _ = _stub_hf(tmp_path, new_text="x = 1\n", lead_ids=(0,))
    assert hf_next_line(hf_model, hf_tok, "cpu")("p") == "x = 1\n"


def test_reference_entry_has_no_fim_number_and_runs_the_harness_on_gpt2(tmp_path):
    hf_model, hf_tok, _ = _stub_hf(tmp_path)
    calls = []

    def fake_harness(model, limit=None, **kwargs):
        calls.append((model, limit, kwargs.get("model_args")))
        return {"blimp_acc": 0.8}

    entry = evaluate_reference("cpu", write_code_data(tmp_path / "eval"), limit=5, code_limit=2, harness=fake_harness, loader=lambda device: (hf_model, hf_tok, 85_000_000))
    assert calls == [("hf", 5, f"pretrained={GPT2_ID},dtype=float32")]
    assert (entry["label"], entry["model_id"], entry["params_non_embedding"], entry["limit"], entry["code_limit"]) == ("GPT-2 small", GPT2_ID, 85_000_000, 5, 2)
    m = entry["metrics"]
    assert m["blimp_acc"] == 0.8 and m["repobench_n"] == 2 and "infill_em_prefix" in m and "infill_em_fim" not in m
    assert entry["harness_version"] == "0.4.13" and entry["time"]
