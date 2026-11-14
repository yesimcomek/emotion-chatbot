"""Render evaluation results as a Markdown table."""

from __future__ import annotations

from typing import Any

from .base import EMOTIONS


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    lines = [
        f"### Results on the `{payload['split']}` split (n = {payload['n']})",
        "",
        "| Model | Macro-F1 [95% CI] | Micro-F1 | Any-emotion F1 | "
        + " | ".join(e.capitalize() for e in EMOTIONS)
        + " | ms/text |",
        "|---|---|---|---|" + "---|" * len(EMOTIONS) + "---|",
    ]
    for r in results:
        lo, hi = r["macro_f1_ci95"]
        per = " | ".join(f"{r['per_label'][e]['f1']:.2f}" for e in EMOTIONS)
        lines.append(
            f"| {r['model']} | **{r['macro_f1']:.3f}** [{lo:.3f}, {hi:.3f}] | {r['micro_f1']:.3f} "
            f"| {r['any_emotion_f1']:.3f} | {per} | {r['ms_per_text']:.2f} |"
        )
    if payload.get("paired"):
        lines += ["", "Paired bootstrap on macro-F1 (1000 resamples):", "",
                  "| Comparison | Δ macro-F1 | 95% CI | P(first better) |", "|---|---|---|---|"]
        for name, c in payload["paired"].items():
            lines.append(
                f"| {name} | {c['diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
                f"| {c['p_better']:.3f} |"
            )
    return "\n".join(lines) + "\n"
