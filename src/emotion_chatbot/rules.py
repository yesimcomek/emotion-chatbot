"""Rule-based emotion classifier: lexicon lookup plus a simple negation window."""

from __future__ import annotations

import re

from .base import EMOTIONS, NEUTRAL, EmotionClassifier, Prediction
from .lexicon import LEXICON, NEGATION_WINDOW, NEGATORS, SUFFIXES

# Clauses end at punctuation or a contrastive conjunction; negation never crosses them.
_CLAUSE_SPLIT = re.compile(r"[.,;:!?()\n]+|\b(?:aber|sondern|doch|jedoch)\b")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _build_index() -> dict[str, list[tuple[str, int]]]:
    """Map every inflected surface form to its (emotion, weight) candidates."""
    index: dict[str, list[tuple[str, int]]] = {}
    for emotion, words in LEXICON.items():
        for word, weight in words.items():
            for suffix in SUFFIXES:
                entry = (emotion, weight)
                forms = index.setdefault(word + suffix, [])
                if entry not in forms:
                    forms.append(entry)
    return index


_INDEX = _build_index()


class RuleBasedClassifier(EmotionClassifier):
    """Weighted lexicon hits per emotion; negated hits are tracked separately.

    An emotion is predicted when its (non-negated) score reaches ``min_score``.
    The primary label is the highest-scoring one.
    """

    name = "rules"

    def __init__(self, min_score: int = 1) -> None:
        self.min_score = min_score

    def _score(self, text: str) -> tuple[dict[str, int], list[str]]:
        scores: dict[str, int] = {}
        negated: list[str] = []
        for clause in _CLAUSE_SPLIT.split(text.lower()):
            tokens = _TOKEN.findall(clause)
            for i, token in enumerate(tokens):
                hits = _INDEX.get(token)
                if not hits:
                    continue
                window = tokens[max(0, i - NEGATION_WINDOW) : i]
                is_negated = any(w in NEGATORS for w in window)
                for emotion, weight in hits:
                    if is_negated:
                        if emotion not in negated:
                            negated.append(emotion)
                    else:
                        scores[emotion] = scores.get(emotion, 0) + weight
        return scores, negated

    def predict_labels(self, text: str) -> frozenset[str]:
        scores, _ = self._score(text)
        return frozenset(e for e, s in scores.items() if s >= self.min_score)

    def scores(self, text: str) -> dict[str, float]:
        return {e: float(s) for e, s in self._score(text)[0].items()}

    def analyze(self, text: str) -> Prediction:
        scores, negated = self._score(text)
        labels = frozenset(e for e in EMOTIONS if scores.get(e, 0) >= self.min_score)
        label = self._primary(labels, {e: float(s) for e, s in scores.items()})
        return Prediction(label if labels else NEUTRAL, labels, tuple(negated))
