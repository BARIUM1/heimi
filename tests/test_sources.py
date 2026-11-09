from minids.data.sources import CODE_FILES, SOURCES, WEB_FILES, local_path


def test_registry_matches_verified_files():
    assert SOURCES["web"].repo_id == "HuggingFaceFW/fineweb-edu" and len(WEB_FILES) == 8
    assert WEB_FILES[0] == "sample/10BT/000_00000.parquet" and WEB_FILES[-1] == "sample/10BT/007_00000.parquet"
    assert SOURCES["code"].repo_id == "HuggingFaceTB/stack-edu" and len(CODE_FILES) == 5
    assert "Shell/train-00000-of-00001.parquet" in CODE_FILES


def test_local_path_flattens_subdirs(tmp_path):
    p = local_path(tmp_path, SOURCES["code"], "Python/train-00000-of-00005.parquet")
    assert p == tmp_path / "raw" / "code" / "Python__train-00000-of-00005.parquet"
