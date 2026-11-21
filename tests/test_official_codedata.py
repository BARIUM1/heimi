from pathlib import Path

import pytest

from minids.official import codedata
from minids.official.codedata import INFILL_LOCAL, REPOBENCH_LOCAL, load_infill, load_repobench
from tests.helpers_official import write_code_data


def test_download_fetches_once_and_leaves_nothing_behind_on_failure(tmp_path, monkeypatch):
    calls = []

    def fake(repo_id, filename, repo_type, local_dir):
        calls.append((repo_id, filename, repo_type))
        got = Path(local_dir) / "got.bin"
        got.write_bytes(b"data")
        return str(got)

    monkeypatch.setattr(codedata, "hf_hub_download", fake)
    dest = tmp_path / "eval" / "x.parquet"
    assert codedata.download_dataset_file("a/b", "data/x.parquet", dest) == dest and dest.read_bytes() == b"data"
    codedata.download_dataset_file("a/b", "data/x.parquet", dest)
    assert calls == [("a/b", "data/x.parquet", "dataset")]  # 已有就不再下

    def broken(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(codedata, "hf_hub_download", broken)
    with pytest.raises(OSError):
        codedata.download_dataset_file("a/b", "f", tmp_path / "eval" / "y.jsonl")
    assert sorted(p.name for p in (tmp_path / "eval").iterdir()) == ["x.parquet"]  # 没有半个文件，也没有临时目录


def test_ensure_code_data_downloads_only_what_is_missing(tmp_path, monkeypatch):
    asked = []

    def fake(repo, filename, dest):
        asked.append((repo, filename, Path(dest).name))
        Path(dest).write_bytes(b"x")
        return Path(dest)

    monkeypatch.setattr(codedata, "download_dataset_file", fake)
    (tmp_path / REPOBENCH_LOCAL).write_bytes(b"x")
    codedata.ensure_code_data(tmp_path)
    assert asked == [(codedata.INFILL_REPO, codedata.INFILL_FILE, INFILL_LOCAL)]


def test_loaders_return_only_the_fields_the_scorers_need(tmp_path):
    d = write_code_data(tmp_path)
    rb = load_repobench(d / REPOBENCH_LOCAL)
    assert len(rb) == 3 and rb[0] == {
        "repo_name": "o/r", "file_path": "m0.py", "import_statement": "import os",
        "cropped_code": "def f():\n    x = 1\n", "next_line": "    return x", "level": "2k",
    }
    inf = load_infill(d / INFILL_LOCAL)  # 文件末尾的空行不算一条
    assert len(inf) == 3 and inf[0] == {"task_id": "t0", "prompt": "def f(a, b):\n", "suffix": "\n", "canonical_solution": "    return a + b\n"}
