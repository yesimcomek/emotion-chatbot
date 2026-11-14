"""Name -> model factory. Heavy imports are lazy, so the rule-based core needs no dependencies."""

from __future__ import annotations

from pathlib import Path

from .base import EmotionClassifier

MODEL_NAMES: tuple[str, ...] = ("rules", "neutral", "prior", "tfidf", "bert")
DEFAULT_PATHS: dict[str, Path] = {
    "tfidf": Path("models/tfidf.joblib"),
    "bert": Path("models/gbert"),
}


def load_model(name: str, path: str | Path | None = None) -> EmotionClassifier:
    """Create or load a classifier by name (trained models are read from ``path``)."""
    if name == "rules":
        from .rules import RuleBasedClassifier

        return RuleBasedClassifier()
    if name == "neutral":
        from .baselines import NeutralBaseline

        return NeutralBaseline()
    if name == "prior":
        from .baselines import PriorBaseline
        from .data import load_splits

        return PriorBaseline().fit(load_splits()[0])
    if name in DEFAULT_PATHS:
        model_path = Path(path) if path else DEFAULT_PATHS[name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained '{name}' model at {model_path}. "
                f"Train one first: emotion-chatbot train {name}"
            )
        if name == "tfidf":
            from .tfidf import TfidfClassifier

            return TfidfClassifier.load(model_path)
        from .bert import BertClassifier

        return BertClassifier.load(model_path)
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
