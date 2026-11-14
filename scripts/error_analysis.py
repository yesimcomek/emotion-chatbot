"""Error analysis of the rule-based model on the validation split (never on test).

    python scripts/error_analysis.py            # writes results/error_analysis.md

Answers: how much of the data does the lexicon even touch, which lexicon words are noisy,
and what do typical misses look like?
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from emotion_chatbot import EMOTIONS, RuleBasedClassifier
from emotion_chatbot.data import load_splits
from emotion_chatbot.lexicon import NEGATORS
from emotion_chatbot.rules import _INDEX, _TOKEN

OUT = Path("results/error_analysis.md")


def main() -> None:
    _, val, _ = load_splits()
    clf = RuleBasedClassifier()
    preds = clf.predict_labels_many(val.texts)

    lines = ["# Error analysis: rule-based model (validation split)", ""]

    # 1) Coverage: how often does the lexicon fire at all?
    emotional = [i for i, ls in enumerate(val.labels) if ls]
    silent = [i for i in emotional if not preds[i]]
    lines += [
        "## Coverage",
        "",
        f"* {len(emotional)} of {len(val)} validation texts carry at least one emotion.",
        f"* The lexicon predicts **nothing** for {len(silent)} of them "
        f"({100 * len(silent) / len(emotional):.0f}%): the emotion is not expressed through a "
        "listed word.",
        "",
        "| Emotion | Gold | Recall | Texts with no lexicon hit |",
        "|---|---|---|---|",
    ]
    for e in EMOTIONS:
        gold = [i for i, ls in enumerate(val.labels) if e in ls]
        hit = [i for i in gold if e in preds[i]]
        none = [i for i in gold if not preds[i]]
        lines.append(f"| {e} | {len(gold)} | {len(hit) / len(gold):.2f} | "
                     f"{100 * len(none) / len(gold):.0f}% |")

    # 2) Noisy lexicon words: precision of each word that fired on val.
    # (emotion, word) -> [occurrences, correct]
    stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for text, gold in zip(val.texts, val.labels):
        for token in set(_TOKEN.findall(text.lower())):
            for emotion, _ in _INDEX.get(token, []):
                stats[(emotion, token)][0] += 1
                stats[(emotion, token)][1] += emotion in gold
    noisy = sorted(((h - tp, h, tp, e, w) for (e, w), (h, tp) in stats.items() if h >= 3),
                   reverse=True)[:10]
    lines += ["", "## Noisiest lexicon words (>= 3 occurrences)", "",
              "| Word | Predicts | Occurrences | Correct |", "|---|---|---|---|"]
    lines += [f"| {w} | {e} | {h} | {tp} |" for _, h, tp, e, w in noisy]

    # 3) Negation: how often is it even relevant?
    neg = sum(1 for t in val.texts if NEGATORS & set(_TOKEN.findall(t.lower())))
    lines += ["", "## Negation", "",
              f"* {neg} of {len(val)} validation texts contain a negator "
              f"({100 * neg / len(val):.0f}%), but most do not negate an *emotion word*; "
              "the rule has little to act on."]

    # 4) Error types. Verbatim texts are deliberately NOT written to the report: the data are
    #    real people's posts, some very personal, and do not belong in a public repository.
    wrong = [i for i in emotional if preds[i] and not (preds[i] & val.labels[i])]
    lines += ["", "## Error types", "",
              f"* Emotion present, nothing predicted: {len(silent)} texts.",
              f"* Emotion predicted but none of the gold emotions: {len(wrong)} texts.",
              "* Example texts are intentionally omitted (see DATA.md)."]
    lines += ["", f"_Counts: {Counter(len(p) for p in preds)} predicted labels per text._", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
