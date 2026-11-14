"""Small web demo: compare models side by side, or chat. Requires the ``demo`` extra (gradio).

    emotion-chatbot demo
"""

from __future__ import annotations

from .base import EMOTIONS, NEUTRAL, EmotionClassifier
from .chat import respond
from .registry import DEFAULT_PATHS, load_model

EXAMPLES = [
    "Ich bin so stolz auf dich, das hast du wirklich großartig gemacht!",
    "Ich habe Angst, dass ich die Prüfung nicht bestehe.",
    "Das ist eine Frechheit, ich bin so wütend auf diese Entscheidung.",
    "Wow, damit hätte ich nie gerechnet!",
    "Ich bin nicht traurig, nur müde.",
    "Morgen fährt der Zug um acht Uhr.",
]


def available_models() -> dict[str, EmotionClassifier]:
    """Rule-based model plus every trained model found on disk."""
    models = {"rules": load_model("rules")}
    for name, path in DEFAULT_PATHS.items():
        if path.exists():
            models[name] = load_model(name)
    return models


def model_scores(model: EmotionClassifier, text: str) -> dict[str, float]:
    """Per-emotion confidences for display (rules are binary: detected or not)."""
    if hasattr(model, "predict_proba"):
        return dict(zip(EMOTIONS, model.predict_proba([text])[0].tolist()))
    detected = model.predict_labels(text)
    return {e: (1.0 if e in detected else 0.0) for e in EMOTIONS}


def build_demo():
    import gradio as gr

    models = available_models()
    names = list(models)

    def compare(text: str):
        if not text.strip():
            return [None] * len(names)
        return [model_scores(models[n], text) for n in names]

    def chat(message: str, history, model_name: str) -> str:
        prediction = models[model_name].analyze(message)
        tag = prediction.label if prediction.label != NEUTRAL else "neutral"
        return f"({tag}) {respond(prediction)}"

    with gr.Blocks(title="German emotion detection") as demo:
        gr.Markdown(
            "# German emotion detection\n"
            "Rule-based vs. TF-IDF + logistic regression vs. fine-tuned German BERT. "
            "Multi-label: a text can show several emotions, or none. "
            "Not a mental-health tool."
        )
        with gr.Tab("Compare models"):
            box = gr.Textbox(label="German text", lines=3)
            gr.Examples(EXAMPLES, inputs=box)
            outputs = []
            with gr.Row():
                for n in names:
                    outputs.append(gr.Label(label=n, num_top_classes=len(EMOTIONS)))
            box.submit(compare, box, outputs)
            gr.Button("Analyze", variant="primary").click(compare, box, outputs)
        with gr.Tab("Chat"):
            gr.ChatInterface(
                chat,
                additional_inputs=[gr.Dropdown(names, value=names[-1], label="Model")],
            )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
