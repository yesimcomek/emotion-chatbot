"""Shared logic for probabilistic multi-label models (TF-IDF, BERT).

Models output one probability per emotion. A label is predicted when its probability
reaches a per-label threshold, tuned on the validation split (never on test).
Requires the ``eval`` extra (numpy).
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .base import EMOTIONS, NEUTRAL, EmotionClassifier, Prediction
from .evaluation import _prf  # per-label precision/recall/F1 helper

DEFAULT_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(2, 19))  # 0.10 .. 0.90


def tune_thresholds(
    y: np.ndarray, probs: np.ndarray, grid: tuple[float, ...] = DEFAULT_GRID
) -> np.ndarray:
    """Pick, for each emotion, the threshold on ``grid`` that maximises its F1 on (y, probs)."""
    best = np.full(y.shape[1], 0.5)
    best_f1 = np.full(y.shape[1], -1.0)
    for t in grid:
        pred = probs >= t
        tp, fp, fn = (y & pred).sum(0), (~y & pred).sum(0), (y & ~pred).sum(0)
        f1 = _prf(tp, fp, fn)[2]
        better = f1 > best_f1
        best[better], best_f1[better] = t, f1[better]
    return best


class ProbabilisticClassifier(EmotionClassifier):
    """Base class: subclasses implement ``predict_proba``; thresholds turn it into labels."""

    def __init__(self) -> None:
        self.thresholds = np.full(len(EMOTIONS), 0.5)

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probabilities of shape (len(texts), len(EMOTIONS))."""

    def _to_labels(self, probs: np.ndarray) -> list[frozenset[str]]:
        hit = probs >= self.thresholds
        return [frozenset(e for e, h in zip(EMOTIONS, row, strict=True) if h) for row in hit]

    def predict_labels_many(self, texts: list[str]) -> list[frozenset[str]]:
        return self._to_labels(self.predict_proba(texts)) if texts else []

    def predict_labels(self, text: str) -> frozenset[str]:
        return self.predict_labels_many([text])[0]

    def scores(self, text: str) -> dict[str, float]:
        return dict(zip(EMOTIONS, self.predict_proba([text])[0].tolist(), strict=True))

    def analyze(self, text: str) -> Prediction:
        probs = self.predict_proba([text])
        labels = self._to_labels(probs)[0]
        scores = dict(zip(EMOTIONS, probs[0].tolist(), strict=True))
        return Prediction(self._primary(labels, scores) if labels else NEUTRAL, labels)
