from dataclasses import replace

from minids.completion import FIM_HOLE
from minids.eval.samples import generate_samples
from minids.model.transformer import Transformer
from minids.train.config import Prompt
from tests.helpers_data import tiny_tokenizer

PROMPTS = [Prompt("en", "The cat"), Prompt("fim", "def f():\n    ", "\nprint(f())\n")]


def test_one_sample_per_prompt_with_display_text(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    m = Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size))
    out = generate_samples(m, tok, PROMPTS, 4, "cpu", seed=0, max_prefix_tokens=8, max_suffix_tokens=8)
    assert [o[0] for o in out] == ["en", "fim"]
    assert out[0][1] == "The cat" and out[1][1] == "def f():\n    " + FIM_HOLE + "\nprint(f())\n"
    assert "\n" not in out[1][2]


def test_samples_reproducible_for_same_seed(tmp_path, tiny_cfg):
    tok = tiny_tokenizer(tmp_path)
    m = Transformer(replace(tiny_cfg, vocab_size=tok.vocab_size))
    a = generate_samples(m, tok, PROMPTS[:1], 6, "cpu", seed=3)
    b = generate_samples(m, tok, PROMPTS[:1], 6, "cpu", seed=3)
    assert a == b
