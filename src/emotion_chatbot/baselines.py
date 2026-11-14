"""Trivial baselines that any real model must beat."""

from __future__ import annotations

import random

from .base import EMOTIONS, EmotionClassifier
from .data import Dataset


class NeutralBaseline(EmotionClassifier):
    """Always predicts 'no emotion'."""

    name = "always-neutral"

    def predict_labels(self, text: str) -> frozenset[str]:
        return frozenset()


class PriorBaseline(EmotionClassifier):
    """Predicts each emotion independently with its training-set frequency (seeded)."""

    name = "random-prior"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._prior = {e: 0.0 for e in EMOTIONS}

    def fit(self, data: Dataset) -> PriorBaseline:
        n = max(len(data), 1)
        self._prior = {e: sum(e in ls for ls in data.labels) / n for e in EMOTIONS}
        return self

    def predict_labels(self, text: str) -> frozenset[str]:
        return frozenset(e for e in EMOTIONS if self._rng.random() < self._prior[e])
