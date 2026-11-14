"""Loader for the German part of the BRIGHTER emotion benchmark (SemEval-2025 Task 11).

Data: https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories
License: CC BY 4.0 (see DATA.md for attribution and citation).

Handling notes (verified on the raw files):
* Test and dev contain every text twice (track A and the cross-lingual track C with
  identical labels). Only track A is kept so results are not double-counted.
* One further exact-duplicate text remains in test; duplicates are dropped.
* An empty label set means "neutral".

Requires the ``eval`` extra (pyarrow).
"""

from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .base import EMOTIONS

_BASE_URL = (
    "https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories"
    "/resolve/main/deu/{split}-00000-of-00001.parquet"
)
_CACHE_DIR = Path("data/raw/brighter_deu")
_HF_SPLIT = {"train": "train", "dev": "dev", "test": "test"}
VAL_FROM_TRAIN = 300  # extra validation examples taken from train (dev alone has only 200)
SPLIT_SEED = 42


@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[frozenset[str]]

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, indices: list[int]) -> Dataset:
        return Dataset([self.texts[i] for i in indices], [self.labels[i] for i in indices])


def _download(split: str, cache_dir: Path) -> Path:
    path = cache_dir / f"{split}.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BASE_URL.format(split=_HF_SPLIT[split]), path)  # noqa: S310
    return path


def load_split(split: str, cache_dir: Path = _CACHE_DIR) -> Dataset:
    """Load one raw split ('train', 'dev' or 'test'), track A only, de-duplicated."""
    import pyarrow.parquet as pq

    table = pq.read_table(_download(split, cache_dir)).to_pydict()
    texts: list[str] = []
    labels: list[frozenset[str]] = []
    seen: set[str] = set()
    for i, text in enumerate(table["text"]):
        if "_track_a_" not in table["id"][i] or text in seen:
            continue
        seen.add(text)
        texts.append(text)
        labels.append(frozenset(e for e in EMOTIONS if table[e][i] == 1))
    return Dataset(texts, labels)


def load_splits(cache_dir: Path = _CACHE_DIR) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train, validation, test).

    Validation = official dev + ``VAL_FROM_TRAIN`` examples drawn from train with a fixed seed.
    Everything is tuned on validation; test is only used for the final report.
    """
    train, dev, test = (load_split(s, cache_dir) for s in ("train", "dev", "test"))
    idx = list(range(len(train)))
    random.Random(SPLIT_SEED).shuffle(idx)
    held = sorted(idx[:VAL_FROM_TRAIN])
    kept = sorted(idx[VAL_FROM_TRAIN:])
    val = Dataset(
        dev.texts + [train.texts[i] for i in held],
        dev.labels + [train.labels[i] for i in held],
    )
    return train.subset(kept), val, test
