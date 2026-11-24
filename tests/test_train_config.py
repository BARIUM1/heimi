import pytest
import yaml

from minids.config import ModelConfig
from minids.data.loader import MixConfig
from minids.train.config import ladder_from_dict, load_ladder

STAGE2 = dict(n_layer=16, d_model=768, n_head=12, kv_lora_rank=256, ffn_hidden=2048)


def test_ladder_yaml_matches_spec_8_1():
    cfg = load_ladder("configs/ladder.yaml")
    assert [s.name for s in cfg.stages] == ["stage0", "stage1", "stage2", "task"]
    assert [s.growth for s in cfg.stages] == ["none", "depth", "width", "none"]
    assert [s.tokens for s in cfg.stages] == [300_000_000, 500_000_000, 2_000_000_000, 700_000_000]
    assert [s.lr for s in cfg.stages] == [2e-3, 1.5e-3, 6e-4, 2e-4]
    assert [s.warmup_frac for s in cfg.stages] == [0.02, 0.05, 0.05, 0.05]
    assert [s.micro_batch for s in cfg.stages] == [16, 16, 8, 8]
    assert cfg.stages[0].model == ModelConfig() and cfg.stages[2].model == ModelConfig(**STAGE2)
    assert cfg.stages[3].mix == MixConfig(web=0.2, code=0.8, fim_rate=0.5)
    assert [cfg.steps_for(s) for s in cfg.stages] == [572, 953, 3814, 1335]
    assert [cfg.accum_steps(s) for s in cfg.stages] == [32, 32, 64, 64]
    assert cfg.tokens_per_step == 524_288 and cfg.eval.interval == 500 and cfg.ckpt_interval == 500
    assert len(cfg.prompts) == 6 and sum(p.suffix is not None for p in cfg.prompts) == 1


def test_baseline_yaml_is_single_stage2_from_scratch():
    cfg = load_ladder("configs/baseline105m.yaml")
    assert len(cfg.stages) == 1 and cfg.stages[0].name == "baseline" and cfg.stages[0].growth == "none"
    assert cfg.stages[0].model == ModelConfig(**STAGE2) and cfg.stages[0].tokens == 2_000_000_000 and cfg.stages[0].lr == 6e-4


def test_baseline_task_yaml_appends_the_ladder_task_stage_to_the_baseline():
    cfg = load_ladder("configs/baseline105m_task.yaml")
    base = load_ladder("configs/baseline105m.yaml")
    ladder = load_ladder("configs/ladder.yaml")
    assert [s.name for s in cfg.stages] == ["baseline", "task"]
    assert cfg.stages[0] == base.stages[0]  # 已训练完的那一级原样不动：重跑时被跳过，不会重训
    assert cfg.stages[1] == ladder.stage("task")  # 任务阶段与阶梯的逐项相同，两边的结果才可比
    assert cfg.steps_for(cfg.stages[1]) == 1335 and cfg.accum_steps(cfg.stages[1]) == 64

    def without_stages(c):
        return {k: v for k, v in c.to_dict().items() if k != "stages"}

    assert without_stages(cfg) == without_stages(base) == without_stages(ladder)  # 种子、评测、提示等全局项一致


def _tiny_dict(**over):
    d = {
        "seed": 1, "seq_len": 32, "tokens_per_step": 256,
        "model": {"vocab_size": 300, "seq_len": 32, "n_layer": 2, "d_model": 64, "n_head": 2,
                  "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 32, "kv_lora_rank": 32, "ffn_hidden": 128},
        "prompts": [{"id": "a", "prefix": "x"}],
        "stages": [
            {"name": "s0", "growth": "none", "model": {}, "tokens": 768, "lr": 1.0e-3, "warmup_frac": 0.1,
             "micro_batch": 4, "mix": {"web": 0.5, "code": 0.5, "fim_rate": 0.0}},
            {"name": "s1", "growth": "depth", "model": {"n_layer": 4}, "tokens": 512, "lr": 1.0e-3,
             "warmup_frac": 0.1, "micro_batch": 4, "mix": {"web": 0.5, "code": 0.5, "fim_rate": 0.0}},
        ],
    }
    d.update(over)
    return d


def test_tiny_dict_round_trips_through_yaml(tmp_path):
    cfg = ladder_from_dict(_tiny_dict())
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert load_ladder(p) == cfg


def test_string_lr_from_yaml_scientific_notation_is_rejected():
    # PyYAML 把 2e-3 读成字符串；这里必须当场报错而不是在优化器里炸
    d = _tiny_dict()
    d["stages"][0]["lr"] = "2e-3"
    with pytest.raises(ValueError, match="lr"):
        ladder_from_dict(d)


def test_growth_chain_mismatch_names_stage():
    d = _tiny_dict()
    d["stages"][1]["model"] = {"n_layer": 3}
    with pytest.raises(ValueError, match="s1"):
        ladder_from_dict(d)


def test_first_stage_must_not_grow():
    d = _tiny_dict()
    d["stages"][0]["growth"] = "depth"
    with pytest.raises(ValueError, match="s0"):
        ladder_from_dict(d)


def test_tokens_per_step_must_divide_by_micro_tokens():
    with pytest.raises(ValueError, match="tokens_per_step"):
        ladder_from_dict(_tiny_dict(tokens_per_step=200))


def test_duplicate_stage_names_rejected():
    d = _tiny_dict()
    d["stages"][1]["name"] = "s0"
    d["stages"][1]["growth"] = "none"
    d["stages"][1]["model"] = {}
    with pytest.raises(ValueError, match="s0"):
        ladder_from_dict(d)
