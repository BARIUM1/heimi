"""数据源注册表与本地布局。

web：FineWeb-Edu 的 10B 采样集，取前 8 个分片（2026-11-08 核实，免申请）。
code：codeparrot/github-code-clean，混合 30 种语言的 GitHub 代码，正文在 code 列，
按 language 列只取 Python、JavaScript、TypeScript、Shell。每片约 355 MB，其中这四种语言
约 195 MB 正文、约 5600 万 token，取 20 片约 11 亿 token（2026-11-08 实测第 0 片）。
原计划的 stack-edu 只有元数据没有正文，已弃用。

本地把仓库内路径的斜杠换成双下划线摊平，例如
data/raw/code/data__train-00000-of-00880.parquet。
"""

from dataclasses import dataclass
from pathlib import Path

TEXT_COLUMNS: tuple[str, ...] = ("text", "content")


@dataclass(frozen=True)
class Source:
    name: str  # "web" | "code"
    repo_id: str
    files: tuple[str, ...]  # 仓库内路径
    text_columns: tuple[str, ...] = TEXT_COLUMNS  # 按顺序识别第一个存在的列
    filter_column: str | None = None  # 为 None 时不过滤
    filter_values: tuple[str, ...] = ()


WEB_FILES: tuple[str, ...] = tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in range(8))

CODE_FILES: tuple[str, ...] = tuple(f"data/train-{i:05d}-of-00880.parquet" for i in range(20))

CODE_LANGUAGES: tuple[str, ...] = ("Python", "JavaScript", "TypeScript", "Shell")

SOURCES: dict[str, Source] = {
    "web": Source("web", "HuggingFaceFW/fineweb-edu", WEB_FILES),
    "code": Source(
        "code",
        "codeparrot/github-code-clean",
        CODE_FILES,
        text_columns=("code",),
        filter_column="language",
        filter_values=CODE_LANGUAGES,
    ),
}


def raw_dir(root: Path, source: Source) -> Path:
    return Path(root) / "raw" / source.name


def local_path(root: Path, source: Source, path_in_repo: str) -> Path:
    return raw_dir(root, source) / path_in_repo.replace("/", "__")


def local_files(root: Path, source: Source) -> list[Path]:
    return [local_path(root, source, f) for f in source.files]
