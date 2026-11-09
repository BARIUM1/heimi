import pytest

from minids.data.download import DownloadError, download_source
from minids.data.sources import Source, local_path


def _fake_fetch(payloads: dict[str, bytes], fail_first: set[str] = frozenset()):
    calls = []

    def fetch(repo_id, path, dest):
        calls.append(path)
        data = payloads[path]
        if path in fail_first and calls.count(path) == 1:
            data = data[:-1]  # 第一次下载缺一个字节，模拟中断
        dest.write_bytes(data)
        return dest

    return fetch, calls


def test_skips_existing_file_with_matching_size(tmp_path):
    src = Source("t", "x/y", ("a.parquet",))
    dest = local_path(tmp_path, src, "a.parquet")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"abc")
    fetch, calls = _fake_fetch({"a.parquet": b"abc"})
    assert download_source(src, tmp_path, fetch=fetch, sizes={"a.parquet": 3}) == [dest]
    assert calls == []


def test_retries_on_size_mismatch_then_succeeds(tmp_path):
    src = Source("t", "x/y", ("a.parquet",))
    fetch, calls = _fake_fetch({"a.parquet": b"abcd"}, fail_first={"a.parquet"})
    out = download_source(src, tmp_path, fetch=fetch, sizes={"a.parquet": 4})
    assert out[0].read_bytes() == b"abcd" and calls == ["a.parquet", "a.parquet"]


def test_raises_after_retries_and_removes_partial(tmp_path):
    src = Source("t", "x/y", ("a.parquet",))

    def always_short(repo_id, path, dest):
        dest.write_bytes(b"ab")
        return dest

    with pytest.raises(DownloadError, match="a.parquet"):
        download_source(src, tmp_path, retries=3, fetch=always_short, sizes={"a.parquet": 4})
    assert not local_path(tmp_path, src, "a.parquet").exists()
