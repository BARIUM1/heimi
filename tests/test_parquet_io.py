import pytest

from minids.data.parquet_io import iter_documents
from tests.helpers_data import write_parquet


def test_iter_documents_picks_text_then_content(tmp_path):
    p = write_parquet(tmp_path / "a.parquet", ["x", None, ""], column="content")
    assert list(iter_documents(p)) == [(0, "x"), (1, None), (2, "")]


def test_iter_documents_filters_by_column_keeping_row_numbers(tmp_path):
    p = write_parquet(
        tmp_path / "a.parquet", ["py", "js", "sh"], column="code",
        extra={"language": ["Python", "JavaScript", "Shell"]},
    )
    got = list(iter_documents(p, text_columns=("code",), filter_column="language", filter_values=("Python", "Shell")))
    assert got == [(0, "py"), (2, "sh")]


def test_iter_documents_unknown_column_raises(tmp_path):
    p = write_parquet(tmp_path / "a.parquet", ["x"], column="body")
    with pytest.raises(ValueError, match="body"):
        next(iter_documents(p))
