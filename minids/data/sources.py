"""数据源注册表与本地布局。

两个来源都在 2026-11-08 核实过：不需要申请权限，文件名与大小见计划二顶部。
本地把仓库内路径的斜杠换成双下划线摊平，例如
data/raw/code/Python__train-00000-of-00005.parquet。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str  # "web" | "code"
    repo_id: str
    files: tuple[str, ...]  # 仓库内路径


WEB_FILES: tuple[str, ...] = tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in range(8))

CODE_FILES: tuple[str, ...] = (
    "Python/train-00000-of-00005.parquet",
    "Python/train-00001-of-00005.parquet",
    "Shell/train-00000-of-00001.parquet",
    "JavaScript/train-00000-of-00003.parquet",
    "TypeScript/train-00000-of-00001.parquet",
)

SOURCES: dict[str, Source] = {
    "web": Source("web", "HuggingFaceFW/fineweb-edu", WEB_FILES),
    "code": Source("code", "HuggingFaceTB/stack-edu", CODE_FILES),
}

TEXT_COLUMNS: tuple[str, ...] = ("text", "content")


def raw_dir(root: Path, source: Source) -> Path:
    return Path(root) / "raw" / source.name


def local_path(root: Path, source: Source, path_in_repo: str) -> Path:
    return raw_dir(root, source) / path_in_repo.replace("/", "__")


def local_files(root: Path, source: Source) -> list[Path]:
    return [local_path(root, source, f) for f in source.files]
