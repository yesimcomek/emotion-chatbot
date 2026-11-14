# German Emotion Detection: Rules vs. TF-IDF vs. BERT

[![CI](https://github.com/yesimcomek/emotion-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/yesimcomek/emotion-chatbot/actions/workflows/ci.yml)

A small, reproducible study of **multi-label emotion detection in German**, with a chatbot on top.
It began as a toy rule-based chatbot and now compares three approaches on a public benchmark
under one evaluation protocol:

| Approach | Idea |
|---|---|
| **Rule-based** | Hand-written German lexicon + clause-aware negation window. No dependencies. |
| **TF-IDF + logistic regression** | Word and character n-grams, one-vs-rest, per-label thresholds. |
| **Fine-tuned German BERT** | `deepset/gbert-base` fine-tuned for multi-label classification. *(See status below.)* |

The data is the German part of [BRIGHTER](https://arxiv.org/abs/2502.11926) (SemEval-2025 Task 11, Track A):
six emotions (anger, disgust, fear, joy, sadness, surprise), multi-label, empty set = neutral.

> **Status:** rules and TF-IDF are complete and evaluated. The BERT fine-tuning code is implemented
> but its results are **not yet included**, because the development machine (8 GB RAM, almost full disk)
> could not complete a run; see [BERT training](#bert-training).

## Results (test split, n = 2,603)

Macro-F1 over the six emotions (the SemEval metric) with 95% bootstrap confidence intervals.
Models were frozen before the test set was used; all tuning happened on a separate validation split.

| Model | Macro-F1 [95% CI] | Micro-F1 | Any-emotion F1 | ms/text |
|---|---|---|---|---|
| always-neutral | 0.000 | 0.000 | 0.000 | 0.00 |
| random (label priors) | 0.199 [0.186, 0.211] | 0.243 | 0.762 | 0.00 |
| **rules** | **0.290** [0.270, 0.307] | 0.320 | 0.633 | 0.01 |
| **TF-IDF + LR** | **0.479** [0.463, 0.495] | 0.553 | 0.862 | 0.20 |
| BERT (gbert-base) | *pending* | | | |

Per-emotion F1 on test:

| Model | Anger | Disgust | Fear | Joy | Sadness | Surprise |
|---|---|---|---|---|---|---|
| rules | 0.16 | 0.09 | 0.27 | 0.49 | 0.47 | 0.25 |
| TF-IDF + LR | 0.70 | 0.63 | 0.28 | 0.56 | 0.54 | 0.16 |

TF-IDF beats rules by +0.189 macro-F1 (paired bootstrap 95% CI [+0.168, +0.212]); both beat the
prior baseline. Full tables and paired comparisons: [`results/results_test.md`](results/results_test.md).

### What the numbers say

* **Anger and disgust are topic-driven in this data.** The most distinctive training words for them are
  about war and politics, not emotion words. A lexicon cannot reach them (anger recall on validation: 4%),
  while TF-IDF learns the topics (F1 0.70 / 0.63). This is a property of the benchmark, not necessarily
  of emotion in general.
* **The lexicon says nothing for 47% of emotional validation texts.** Details in
  [`results/error_analysis.md`](results/error_analysis.md).
* **Rare classes are unstable.** Surprise has only 162 test positives; TF-IDF scores 0.42 on validation
  but 0.16 on test. Macro-F1 differences of a few points on single classes should not be over-read.
* **Validation scores are optimistic.** TF-IDF: 0.519 on validation vs. 0.479 on test, because per-label
  thresholds are tuned on 500 validation texts.

### Complementary failure modes (probe sentences)

On 24 hand-written sentences ([`results/probes.md`](results/probes.md)) the picture flips:

| Probe group | rules | TF-IDF |
|---|---|---|
| explicit emotion statements ("Ich bin so wütend") | 11/12 | 5/12 |
| negated emotions ("Ich bin nicht traurig") | 6/6 | 0/6 |
| neutral statements | 6/6 | 3/6 |

TF-IDF is a bag of words: it reads "nicht traurig" as *sadness*, and it has learned the benchmark's
topics rather than explicit emotional language. **Caveat:** I wrote both the lexicon and the probes, so the
rules' score is partly circular. Treat this as a qualitative illustration, not a ranking.

## Quick start

```bash
git clone https://github.com/yesimcomek/emotion-chatbot.git
cd emotion-chatbot
python -m venv .venv && source .venv/bin/activate

pip install -e .                 # rule-based chatbot only (no dependencies)
python -m emotion_chatbot        # chat in German; type "exit" to quit
```

```text
Du: Ich bin nicht traurig
Bot: (neutral) Das freut mich zu hören!
Du: Ich habe Angst vor der Prüfung, bin aber auch stolz
Bot: (fear) Mach dir keine Sorgen. Ich bin bei dir.
```

Optional extras:

```bash
pip install -e ".[eval]"         # numpy, pyarrow, scikit-learn  -> evaluation + TF-IDF model
pip install -e ".[bert]"         # + torch, transformers         -> BERT fine-tuning
pip install -e ".[demo]"         # + gradio                      -> web demo
pip install -e ".[dev]"          # pytest, ruff
```

```bash
emotion-chatbot train tfidf                                   # ~10 s, saves models/tfidf.joblib
emotion-chatbot chat --model tfidf
emotion-chatbot evaluate --models rules tfidf --split val     # downloads the data on first use
emotion-chatbot evaluate --models neutral prior rules tfidf --split test
emotion-chatbot demo                                          # side-by-side model comparison
python scripts/error_analysis.py                              # why the lexicon fails (validation only)
python scripts/probes.py                                      # hand-written probe sentences
```

### BERT training

```bash
pip install -e ".[bert]"
emotion-chatbot train bert          # deepset/gbert-base, 5 epochs, effective batch 16, saves models/gbert
emotion-chatbot evaluate --models rules tfidf bert --split test
```

This is memory-hungry: on the 8 GB MacBook used for development it exhausted RAM and swap and
was stopped before finishing an epoch. Use a machine with a GPU (CUDA or Apple MPS) and
spare memory, e.g. a free Colab T4 *(untested)*. If memory is tight, lower `--batch-size`
(default 8, with `--grad-accum 2`).

## Method

**Data.** BRIGHTER German, Track A. Inspecting the raw files showed that every dev/test text appears twice
(Track A and the cross-lingual Track C) and that dev holds only 200 unique texts; the loader keeps Track A,
drops duplicates, and builds the splits below. See [`DATA.md`](DATA.md).

| Split | Texts | Used for |
|---|---|---|
| train | 2,303 | fitting |
| validation | 500 (official dev + 300 from train, seed 42) | C, thresholds, epoch selection, lexicon sanity checks |
| test | 2,603 | final report only |

**Protocol.** Hyper-parameters (TF-IDF's `C`), per-label decision thresholds and the BERT epoch are chosen on
validation. The lexicon was extended once after looking at *training* statistics and checked on validation;
it was not tuned on test. Test results come from frozen models.

**Metrics.** Macro-F1 over the six emotions (per-emotion binary F1, averaged), micro-F1, "any emotion" F1
(emotional vs. neutral), exact match, 95% percentile-bootstrap CIs (1,000 resamples of texts) and paired
bootstrap for model differences.

**Rule-based model.** Lexicon of ~170 base forms with weights (2 = clear, 1 = weak), common German
inflection suffixes, and negation: a negator (*nicht, kein-, nie, ohne, …*) within three tokens before an
emotion word, inside the same clause (punctuation and *aber/sondern/doch/jedoch* end a clause), cancels it.

**TF-IDF model.** Word 1–2-grams + character 2–5-grams, one-vs-rest logistic regression with balanced class
weights, `C` selected on validation.

## Project layout

```text
src/emotion_chatbot/
  base.py        label set, Prediction, EmotionClassifier interface
  lexicon.py     German emotion lexicon, negators
  rules.py       rule-based classifier
  tfidf.py       TF-IDF + logistic regression
  bert.py        BERT fine-tuning / inference
  ml_base.py     probabilities -> labels, threshold tuning
  data.py        BRIGHTER loader and splits
  evaluation.py  metrics, bootstrap CIs
  chat.py        replies and chat loop
  cli.py         chat / train / evaluate / demo
  demo.py        Gradio demo
scripts/         error_analysis.py, probes.py
tests/           pytest suite (offline)
results/         metrics, tables, error analysis
```

## Limitations and ethics

* **Not a mental-health tool.** The chatbot's replies are canned templates triggered by a classifier that is
  makes frequent errors (macro-F1 ≈ 0.48 at best on the test set). Do not use it for anything that
  matters to someone's wellbeing.
* Labels are subjective and the data is mostly social-media text, much of it political. Models will not
  transfer to other domains (customer reviews, clinical text) without re-evaluation.
* Annotator demographics and cultural context affect what counts as, e.g., "disgust".
* The chatbot's replies are in German only; the classifiers are German-only.
* Test results come from a single training run (seed 42); no variance over seeds is reported.

## License and attribution

Code: MIT ([`LICENSE`](LICENSE)). Data is not redistributed; BRIGHTER is CC BY 4.0, please cite the
papers listed in [`DATA.md`](DATA.md). `deepset/gbert-base` is MIT-licensed.

---

*Ursprüngliche Version:* Ein einfacher regelbasierter Chatbot, der Emotionen in deutschen Texten anhand von
Schlüsselwörtern erkennt. Der Chatbot antwortet weiterhin auf Deutsch.
