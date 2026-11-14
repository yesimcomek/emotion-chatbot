"""Response selection and the interactive chat loop."""

from __future__ import annotations

from collections.abc import Callable

from .base import NEUTRAL, EmotionClassifier, Prediction

RESPONSES: dict[str, str] = {
    "joy": "Das freut mich sehr!",
    "anger": "Warum bist du verärgert? Ich bin hier, um zu helfen.",
    "disgust": "Das klingt unangenehm. Was genau stößt dich ab?",
    "sadness": "Es tut mir leid, das zu hören. Möchtest du darüber sprechen?",
    "fear": "Mach dir keine Sorgen. Ich bin bei dir.",
    "surprise": "Wow, das klingt spannend!",
    NEUTRAL: "Interessant! Erzähl mir mehr.",
}

# Used when an emotion was mentioned but negated ("Ich bin nicht traurig").
NEGATED_RESPONSES: dict[str, str] = {
    "joy": "Das tut mir leid zu hören. Möchtest du darüber sprechen?",
    "anger": "Das ist gut, dass du nicht verärgert bist!",
    "sadness": "Das freut mich zu hören!",
    "fear": "Das ist gut, dass du keine Angst hast!",
    "disgust": "Gut, dass dich das nicht abstößt!",
    "surprise": "Verstehe, das scheint für dich normal zu sein.",
}


def respond(prediction: Prediction) -> str:
    """Pick a reply for a prediction; negated mentions only matter if nothing else was found."""
    if prediction.label == NEUTRAL and prediction.negated:
        return NEGATED_RESPONSES[prediction.negated[0]]
    return RESPONSES[prediction.label]


def run_chat(
    classifier: EmotionClassifier,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Read lines until the user types 'exit'. I/O is injectable for testing."""
    output_fn(f"Emotion chatbot (model: {classifier.name}). Type 'exit' to quit.")
    while True:
        try:
            text = input_fn("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            break
        if text.lower() == "exit":
            break
        if not text:
            continue
        prediction = classifier.analyze(text)
        output_fn(f"Bot: ({prediction.label}) {respond(prediction)}")
    output_fn("Bot: Bis zum nächsten Mal!")
