### Results on the `test` split (n = 2603)

| Model | Macro-F1 [95% CI] | Micro-F1 | Any-emotion F1 | Anger | Disgust | Fear | Joy | Sadness | Surprise | ms/text |
|---|---|---|---|---|---|---|---|---|---|---|
| always-neutral | **0.000** [0.000, 0.000] | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random-prior | **0.199** [0.186, 0.211] | 0.243 | 0.762 | 0.30 | 0.34 | 0.10 | 0.21 | 0.18 | 0.08 | 0.00 |
| rules | **0.290** [0.270, 0.307] | 0.320 | 0.633 | 0.16 | 0.09 | 0.27 | 0.49 | 0.47 | 0.25 | 0.01 |
| tfidf-lr | **0.479** [0.463, 0.495] | 0.553 | 0.862 | 0.70 | 0.63 | 0.28 | 0.56 | 0.54 | 0.16 | 0.20 |

Paired bootstrap on macro-F1 (1000 resamples):

| Comparison | Δ macro-F1 | 95% CI | P(first better) |
|---|---|---|---|
| random-prior vs always-neutral | +0.198 | [+0.184, +0.212] | 1.000 |
| rules vs always-neutral | +0.290 | [+0.270, +0.307] | 1.000 |
| rules vs random-prior | +0.092 | [+0.066, +0.115] | 1.000 |
| tfidf-lr vs always-neutral | +0.479 | [+0.463, +0.495] | 1.000 |
| tfidf-lr vs random-prior | +0.281 | [+0.260, +0.301] | 1.000 |
| tfidf-lr vs rules | +0.189 | [+0.168, +0.212] | 1.000 |
