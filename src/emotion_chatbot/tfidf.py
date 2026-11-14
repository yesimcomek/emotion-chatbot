"""Classical ML baseline: word + character TF-IDF features with one-vs-rest logistic regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import EMOTIONS
from .data import Dataset
from .evaluation import macro_f1, to_matrix
from .ml_base import ProbabilisticClassifier, tune_thresholds

C_GRID: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0)


def _build_pipeline(c: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                     sublinear_tf=True)),
        ]
    )
    clf = OneVsRestClassifier(
        LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear")
    )
    return Pipeline([("features", features), ("clf", clf)])


class TfidfClassifier(ProbabilisticClassifier):
    name = "tfidf-lr"

    def __init__(self, c: float = 1.0) -> None:
        super().__init__()
        self.c = c
        self.pipeline = _build_pipeline(c)

    def fit(self, train: Dataset) -> TfidfClassifier:
        self.pipeline.fit(train.texts, to_matrix(train.labels).astype(int))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    @classmethod
    def train(
        cls, train: Dataset, val: Dataset, c_grid: tuple[float, ...] = C_GRID
    ) -> TfidfClassifier:
        """Fit on ``train``; choose C and per-label thresholds by macro-F1 on ``val``."""
        y_val = to_matrix(val.labels)
        best: tuple[float, TfidfClassifier] | None = None
        for c in c_grid:
            model = cls(c).fit(train)
            probs = model.predict_proba(val.texts)
            model.thresholds = tune_thresholds(y_val, probs)
            score = macro_f1(y_val, probs >= model.thresholds)
            print(f"  C={c:<5} val macro-F1={score:.4f}")
            if best is None or score > best[0]:
                best = (score, model)
        assert best is not None
        print(f"  -> selected C={best[1].c} (val macro-F1={best[0]:.4f})")
        return best[1]

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "thresholds": self.thresholds, "c": self.c,
             "emotions": EMOTIONS},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TfidfClassifier:
        import joblib

        blob = joblib.load(path)
        if tuple(blob["emotions"]) != EMOTIONS:
            raise ValueError("Saved model was trained with a different label set.")
        model = cls(blob["c"])
        model.pipeline, model.thresholds = blob["pipeline"], blob["thresholds"]
        return model
