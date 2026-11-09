from minids.data.sources import CODE_FILES, SOURCES, WEB_FILES, local_path


def test_registry_matches_verified_files():
    assert SOURCES["web"].repo_id == "HuggingFaceFW/fineweb-edu" and len(WEB_FILES) == 8
    assert WEB_FILES[0] == "sample/10BT/000_00000.parquet" and WEB_FILES[-1] == "sample/10BT/007_00000.parquet"
    code = SOURCES["code"]
    assert code.repo_id == "codeparrot/github-code-clean" and len(CODE_FILES) == 20
    assert CODE_FILES[0] == "data/train-00000-of-00880.parquet"
    assert CODE_FILES[-1] == "data/train-00019-of-00880.parquet"
    # 该数据集混合 30 种语言，正文在 code 列，按 language 列只取四种
    assert code.text_columns == ("code",) and code.filter_column == "language"
    assert set(code.filter_values) == {"Python", "JavaScript", "TypeScript", "Shell"}
    assert SOURCES["web"].filter_column is None


def test_local_path_flattens_subdirs(tmp_path):
    p = local_path(tmp_path, SOURCES["code"], "data/train-00000-of-00880.parquet")
    assert p == tmp_path / "raw" / "code" / "data__train-00000-of-00880.parquet"
