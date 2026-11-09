"""黑米的分词器：16k 词表的字节级 BPE。

前 16 个 id 固定留给特殊 token（顺序见 SPECIAL_TOKENS）。
文本里字面出现的特殊 token 字符串一律按普通文本编码，特殊 id 只能由代码显式插入，
这样数据里的一段 "<|endoftext|>" 不可能变成控制 token。

这条规则不在保存的 json 文件里，只在 HeimiTokenizer 的构造函数里生效。所以 heimi-16k.json
必须经 HeimiTokenizer.load 加载；直接用 tokenizers.Tokenizer.from_file 读到的对象会把字面的
特殊串识别成 id 0..15。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from minids.data.parquet_io import iter_source_documents
from minids.data.sources import SOURCES, Source, local_files

log = logging.getLogger(__name__)

SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|pad|>",
] + [f"<|reserved_{i}|>" for i in range(5, 16)]
VOCAB_SIZE = 16384
EOT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, PAD = 0, 1, 2, 3, 4


class HeimiTokenizer:
    def __init__(self, tok: Tokenizer):
        self._tok = tok
        # True 表示文本中的特殊 token 字符串被当作普通文本切分，而不是识别成特殊 id
        self._tok.encode_special_tokens = True

    @classmethod
    def load(cls, path: Path) -> "HeimiTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def token_to_id(self, token: str) -> int:
        i = self._tok.token_to_id(token)
        if i is None:
            raise KeyError(token)
        return i


def train_tokenizer(texts: Iterable[str], out_path: Path, vocab_size: int = VOCAB_SIZE) -> HeimiTokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return HeimiTokenizer(tok)


def _iter_source_texts(source: Source, paths: Sequence[Path], skip_rows_first_file: int, byte_budget: int) -> Iterator[str]:
    used = 0
    for i, path in enumerate(paths):
        skip = skip_rows_first_file if i == 0 else 0
        for row, text in iter_source_documents(path, source):
            if row < skip or not text:
                continue
            yield text
            used += len(text.encode("utf-8"))
            if used >= byte_budget:
                return


def iter_training_corpus(
    root: Path,
    val_rows: dict[str, int],
    web_bytes: int = 1_400_000_000,
    code_bytes: int = 600_000_000,
) -> Iterator[str]:
    """分词器训练语料：web 与 code 各按字节预算顺序抽取，跳过各自第一个文件里的验证集行。"""
    web, code = SOURCES["web"], SOURCES["code"]
    # 缺键直接 KeyError：默认成 0 行会把验证集悄悄混进分词器语料
    yield from _iter_source_texts(web, local_files(root, web), val_rows["web"], web_bytes)
    yield from _iter_source_texts(code, local_files(root, code), val_rows["code"], code_bytes)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="训练黑米的 16k 分词器")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="data")
    t.add_argument("--out", default="data/tokenizer/heimi-16k.json")
    t.add_argument("--val-rows", required=True, help="prepare val-rows 产出的 json")
    t.add_argument("--web-bytes", type=int, default=1_400_000_000)
    t.add_argument("--code-bytes", type=int, default=600_000_000)
    t.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    val_rows = json.loads(Path(args.val_rows).read_text(encoding="utf-8"))
    counted = {"bytes": 0, "docs": 0}

    def counting() -> Iterator[str]:
        for text in iter_training_corpus(Path(args.root), val_rows, args.web_bytes, args.code_bytes):
            counted["bytes"] += len(text.encode("utf-8"))
            counted["docs"] += 1
            yield text

    t0 = time.perf_counter()
    tok = train_tokenizer(counting(), Path(args.out), vocab_size=args.vocab_size)
    log.info(
        "词表 %d，语料 %d 篇 %.2f GB，耗时 %.0f 秒，保存到 %s",
        tok.vocab_size, counted["docs"], counted["bytes"] / 1e9, time.perf_counter() - t0, args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
