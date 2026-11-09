import json

from scripts.make_smoke_data import write_smoke_data


def test_smoke_data_layout_matches_index(tmp_path):
    idx = write_smoke_data(tmp_path, vocab=300, tokens=5000, seed=0)
    d = json.loads(idx.read_text(encoding="utf-8"))
    assert d["vocab_size"] == 300 and set(d["sources"]) == {"web", "code"}
    for src in ("web", "code"):
        for split in ("train", "val"):
            e = d["sources"][src][split][0]
            assert (tmp_path / src / e["file"]).stat().st_size == 2 * e["tokens"] == 2 * 5000
