"""German emotion detection: rule-based, classical ML and transformer models."""

from .base import EMOTIONS, LABELS, NEUTRAL, EmotionClassifier, Prediction
from .rules import RuleBasedClassifier

__all__ = [
    "EMOTIONS",
    "LABELS",
    "NEUTRAL",
    "EmotionClassifier",
    "Prediction",
    "RuleBasedClassifier",
]
__version__ = "0.2.0"
