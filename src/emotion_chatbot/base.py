"""Shared label set and the classifier interface all models implement.

The task is multi-label: a text can express several emotions at once, or none
(neutral). This matches the BRIGHTER / SemEval-2025 Task 11 German benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

EMOTIONS: tuple[str, ...] = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
NEUTRAL = "neutral"
LABELS: tuple[str, ...] = (*EMOTIONS, NEUTRAL)


@dataclass(frozen=True)
class Prediction:
    """Result of classifying one text.

    ``label`` is the single most prominent emotion (or ``neutral``), used by the chatbot.
    ``labels`` is the full multi-label set (empty = neutral).
    ``negated`` lists emotions that were mentioned but negated ("nicht traurig");
    only the rule-based model fills it.
    """

    label: str
    labels: frozenset[str] = frozenset()
    negated: tuple[str, ...] = ()


class EmotionClassifier(ABC):
    """Anything that maps a German text to a set of emotion labels."""

    name: str = "classifier"

    @abstractmethod
    def predict_labels(self, text: str) -> frozenset[str]:
        """Return all emotions detected in ``text`` (empty set = neutral)."""

    def predict_labels_many(self, texts: list[str]) -> list[frozenset[str]]:
        """Batch prediction; models override this for speed."""
        return [self.predict_labels(t) for t in texts]

    def scores(self, text: str) -> dict[str, float]:
        """Per-emotion confidence used to pick the primary label (default: 1 if detected)."""
        return {e: 1.0 for e in self.predict_labels(text)}

    def analyze(self, text: str) -> Prediction:
        """Detailed prediction with a primary label."""
        labels = self.predict_labels(text)
        return Prediction(label=self._primary(labels, self.scores(text)), labels=labels)

    def predict(self, text: str) -> str:
        """Single primary label, or ``neutral``."""
        return self.analyze(text).label

    @staticmethod
    def _primary(labels: frozenset[str], scores: dict[str, float]) -> str:
        if not labels:
            return NEUTRAL
        # max() keeps the first emotion in EMOTIONS order on ties -> deterministic.
        return max((e for e in EMOTIONS if e in labels), key=lambda e: scores.get(e, 0.0))
