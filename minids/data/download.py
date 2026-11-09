"""下载原始 parquet：按远端大小校验，不符就删掉重下，最多重试 3 次。

fetch 和 sizes 两个参数可注入，单元测试用假函数，不访问网络。
真实下载走 huggingface_hub.hf_hub_download，它自己支持断点续传；
暂存目录固定在 raw/<source>/.staging 下，重试时未完成的文件能接着下。
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from minids.data.sources import SOURCES, Source, local_path, raw_dir
from minids.keepawake import keep_awake

log = logging.getLogger(__name__)

FetchFn = Callable[[str, str, Path], Path]  # (repo_id, path_in_repo, dest) -> dest


class DownloadError(RuntimeError):
    pass


def remote_sizes(repo_id: str, paths: Sequence[str]) -> dict[str, int]:
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(repo_id, list(paths), repo_type="dataset")
    sizes = {info.path: info.size for info in infos}
    missing = [p for p in paths if sizes.get(p) is None]
    if missing:
        raise DownloadError(f"远端仓库 {repo_id} 缺少文件或没有大小信息：{missing}")
    return {p: int(sizes[p]) for p in paths}


def _hf_fetch(repo_id: str, path_in_repo: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    staging = dest.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type="dataset", local_dir=str(staging))
    shutil.move(got, dest)
    return dest


def download_source(
    source: Source,
    root: Path,
    retries: int = 3,
    fetch: FetchFn | None = None,
    sizes: dict[str, int] | None = None,
) -> list[Path]:
    fetch = fetch or _hf_fetch
    if sizes is None:
        sizes = remote_sizes(source.repo_id, source.files)

    out: list[Path] = []
    for path in source.files:
        dest = local_path(root, source, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = sizes[path]

        if dest.exists() and dest.stat().st_size == expected:
            log.info("跳过 %s（已存在，%d 字节）", path, expected)
            out.append(dest)
            continue

        last_actual = None
        for attempt in range(1, retries + 1):
            log.info("下载 %s（第 %d 次）", path, attempt)
            fetch(source.repo_id, path, dest)
            actual = dest.stat().st_size if dest.exists() else -1
            if actual == expected:
                log.info("校验通过 %s（%d 字节）", path, actual)
                break
            last_actual = actual
            log.warning("大小不符 %s：期望 %d，实际 %d，删除后重试", path, expected, actual)
            dest.unlink(missing_ok=True)
        else:
            raise DownloadError(f"{path} 下载 {retries} 次后大小仍不符：期望 {expected}，实际 {last_actual}")
        out.append(dest)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="下载黑米的原始数据分片")
    p.add_argument("--source", choices=["web", "code", "all"], default="all")
    p.add_argument("--root", default="data")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    names = ["web", "code"] if args.source == "all" else [args.source]
    total = 0
    with keep_awake():
        for name in names:
            files = download_source(SOURCES[name], Path(args.root))
            size = sum(f.stat().st_size for f in files)
            total += size
            log.info("%s：%d 个文件，%.2f GB，目录 %s", name, len(files), size / 1e9, raw_dir(Path(args.root), SOURCES[name]))
    log.info("合计 %.2f GB", total / 1e9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
