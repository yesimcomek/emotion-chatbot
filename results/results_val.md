### Results on the `val` split (n = 500)

| Model | Macro-F1 [95% CI] | Micro-F1 | Any-emotion F1 | Anger | Disgust | Fear | Joy | Sadness | Surprise | ms/text |
|---|---|---|---|---|---|---|---|---|---|---|
| always-neutral | **0.000** [0.000, 0.000] | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random-prior | **0.222** [0.190, 0.253] | 0.258 | 0.745 | 0.34 | 0.32 | 0.20 | 0.21 | 0.20 | 0.06 | 0.00 |
| rules | **0.328** [0.281, 0.367] | 0.349 | 0.630 | 0.08 | 0.14 | 0.38 | 0.52 | 0.51 | 0.34 | 0.02 |
| tfidf-lr | **0.519** [0.479, 0.556] | 0.555 | 0.851 | 0.67 | 0.63 | 0.31 | 0.55 | 0.54 | 0.42 | 0.26 |

Paired bootstrap on macro-F1 (1000 resamples):

| Comparison | Δ macro-F1 | 95% CI | P(first better) |
|---|---|---|---|
| random-prior vs always-neutral | +0.192 | [+0.164, +0.220] | 1.000 |
| rules vs always-neutral | +0.328 | [+0.281, +0.367] | 1.000 |
| rules vs random-prior | +0.136 | [+0.084, +0.183] | 1.000 |
| tfidf-lr vs always-neutral | +0.519 | [+0.479, +0.556] | 1.000 |
| tfidf-lr vs random-prior | +0.327 | [+0.281, +0.373] | 1.000 |
| tfidf-lr vs rules | +0.191 | [+0.145, +0.243] | 1.000 |
