"""Multi-label evaluation: per-label and macro/micro F1, bootstrap confidence intervals.

The headline metric is **macro-F1 over the six emotions** (the SemEval-2025 Task 11
metric): F1 is computed per emotion as a binary problem, then averaged. Neutral texts
(empty label set) only count as negatives for every emotion.

Requires the ``eval`` extra (numpy).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import EMOTIONS, EmotionClassifier
from .data import Dataset


def to_matrix(labels: list[frozenset[str]]) -> np.ndarray:
    """Binary matrix (n_texts, n_emotions) from label sets."""
    return np.array([[e in ls for e in EMOTIONS] for ls in labels], dtype=bool).reshape(
        len(labels), len(EMOTIONS)
    )


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, ...]:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f


def _counts(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (y & p).sum(0), (~y & p).sum(0), (y & ~p).sum(0)


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(_prf(*_counts(y, p))[2].mean())


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Metrics for binary matrices ``y`` (gold) and ``p`` (predicted)."""
    tp, fp, fn = _counts(y, p)
    prec, rec, f1 = _prf(tp, fp, fn)
    totals = (a.sum(keepdims=True) for a in (tp, fp, fn))
    micro_p, micro_r, micro_f = (float(x[0]) for x in _prf(*totals))
    # "Is there any emotion at all?" -- separates neutral from emotional texts.
    any_y, any_p = y.any(1, keepdims=True), p.any(1, keepdims=True)
    any_f = float(_prf(*_counts(any_y, any_p))[2][0])
    return {
        "macro_f1": float(f1.mean()),
        "micro_f1": micro_f,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "any_emotion_f1": any_f,
        "exact_match": float((y == p).all(1).mean()),
        "per_label": {
            e: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(y[:, i].sum()),
            }
            for i, e in enumerate(EMOTIONS)
        },
    }


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for macro-F1, resampling texts."""
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = [macro_f1(y[idx], p[idx]) for idx in rng.integers(0, n, (n_boot, n))]
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Compare two models on the same texts: macro-F1 difference (a - b) with 95% CI.

    ``p_better`` is the fraction of resamples where a > b (>0.975 or <0.025 ~ significant).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.array(
        [macro_f1(y[i], p_a[i]) - macro_f1(y[i], p_b[i]) for i in rng.integers(0, n, (n_boot, n))]
    )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": macro_f1(y, p_a) - macro_f1(y, p_b),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_better": float((diffs > 0).mean()),
    }


def predict_matrix(clf: EmotionClassifier, data: Dataset) -> tuple[np.ndarray, float]:
    """Predict a whole dataset; returns (binary matrix, seconds taken)."""
    start = time.perf_counter()
    preds = clf.predict_labels_many(data.texts)
    return to_matrix(preds), time.perf_counter() - start


def evaluate(clf: EmotionClassifier, data: Dataset, n_boot: int = 1000) -> dict[str, Any]:
    """Full evaluation of one classifier on one dataset."""
    y = to_matrix(data.labels)
    p, seconds = predict_matrix(clf, data)
    result = compute_metrics(y, p)
    result["macro_f1_ci95"] = bootstrap_ci(y, p, n_boot=n_boot)
    result["model"] = clf.name
    result["n"] = len(data)
    result["ms_per_text"] = 1000 * seconds / max(len(data), 1)
    return result
