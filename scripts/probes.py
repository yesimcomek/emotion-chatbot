"""Hand-written probe sentences: explicit emotion statements, negation and neutral text.

    python scripts/probes.py            # writes results/probes.md

These sentences were written by the author (not taken from any dataset). They test whether
each model handles *explicit* emotional language and negation -- behaviour the benchmark,
which is dominated by social-media posts, barely measures. 24 sentences: indicative only,
not a statistical result. The lexicon was written by the same author, so rules are
favoured by construction.
"""

from __future__ import annotations

from pathlib import Path

from emotion_chatbot.registry import load_model

# (text, expected label set)  --  empty set = neutral
PROBES: list[tuple[str, set[str]]] = [
    ("Ich bin so glücklich, dass ich endlich bestanden habe!", {"joy"}),
    ("Das ist wunderbar, ich freue mich riesig.", {"joy"}),
    ("Ich bin unglaublich wütend auf meinen Chef.", {"anger"}),
    ("Das ist eine Frechheit, ich bin sauer.", {"anger"}),
    ("Ich bin traurig, weil mein Freund wegzieht.", {"sadness"}),
    ("Ich fühle mich einsam und niedergeschlagen.", {"sadness"}),
    ("Ich habe große Angst vor der Operation.", {"fear"}),
    ("Ich bin sehr nervös und besorgt.", {"fear"}),
    ("Das ist ekelhaft, ich muss würgen.", {"disgust"}),
    ("Wie widerlich, das Essen riecht verdorben.", {"disgust"}),
    ("Wow, damit habe ich überhaupt nicht gerechnet!", {"surprise"}),
    ("Das ist ja unfassbar, ich bin total überrascht.", {"surprise"}),
    # negation: the named emotion is explicitly absent
    ("Ich bin nicht traurig.", set()),
    ("Ich habe keine Angst.", set()),
    ("Ich bin gar nicht wütend.", set()),
    ("Ich bin nicht glücklich damit.", set()),
    ("Das ist nicht ekelhaft.", set()),
    ("Ich bin nicht überrascht.", set()),
    # neutral
    ("Der Zug fährt um acht Uhr ab.", set()),
    ("Morgen ist Dienstag.", set()),
    ("Die Besprechung findet im dritten Stock statt.", set()),
    ("Ich habe Brot und Milch gekauft.", set()),
    ("Das Paket wiegt zwei Kilogramm.", set()),
    ("Die Seite hat zwölf Kapitel.", set()),
]
GROUPS = {"explicit": range(0, 12), "negation": range(12, 18), "neutral": range(18, 24)}


def main() -> None:
    models = {name: load_model(name) for name in ("rules", "tfidf")}
    try:
        models["bert"] = load_model("bert")
    except FileNotFoundError:
        pass

    preds = {n: m.predict_labels_many([t for t, _ in PROBES]) for n, m in models.items()}
    header = "| Group | " + " | ".join(models) + " |"
    lines = [
        "# Probe sentences (hand-written, n = 24)",
        "",
        "Exact match of the predicted label set against the expected one "
        "(empty set = neutral / emotion absent).",
        "",
        "**Caveat:** the same author wrote the lexicon and these sentences, so the rule-based "
        "score is partly circular. Read this as a qualitative contrast (explicit statements and "
        "negation vs. benchmark-style text), not as a ranking of the models.",
        "",
        header,
        "|---|" + "---|" * len(models),
    ]
    for group, idx in GROUPS.items():
        cells = []
        for n in models:
            ok = sum(preds[n][i] == frozenset(PROBES[i][1]) for i in idx)
            cells.append(f"{ok}/{len(idx)}")
        lines.append(f"| {group} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-sentence predictions", "",
              "| Sentence | Expected | " + " | ".join(models) + " |",
              "|---|---|" + "---|" * len(models)]
    for i, (text, exp) in enumerate(PROBES):
        fmt = lambda s: ", ".join(sorted(s)) or "–"  # noqa: E731
        lines.append(f"| {text} | {fmt(exp)} | " + " | ".join(fmt(preds[n][i]) for n in models)
                     + " |")

    out = Path("results/probes.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
