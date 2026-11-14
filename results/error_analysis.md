# Error analysis: rule-based model (validation split)

## Coverage

* 371 of 500 validation texts carry at least one emotion.
* The lexicon predicts **nothing** for 175 of them (47%): the emotion is not expressed through a listed word.

| Emotion | Gold | Recall | Texts with no lexicon hit |
|---|---|---|---|
| anger | 136 | 0.04 | 68% |
| disgust | 156 | 0.08 | 60% |
| fear | 51 | 0.29 | 45% |
| joy | 105 | 0.66 | 27% |
| sadness | 99 | 0.46 | 32% |
| surprise | 33 | 0.27 | 45% |

## Noisiest lexicon words (>= 3 occurrences)

| Word | Predicts | Occurrences | Correct |
|---|---|---|---|
| gut | joy | 34 | 15 |
| gute | joy | 19 | 6 |
| danke | joy | 20 | 9 |
| gerne | joy | 13 | 2 |
| liebe | joy | 10 | 1 |
| schön | joy | 17 | 9 |
| leider | sadness | 19 | 12 |
| schlimm | fear | 7 | 0 |
| krank | disgust | 10 | 4 |
| schlecht | sadness | 9 | 3 |

## Negation

* 234 of 500 validation texts contain a negator (47%), but most do not negate an *emotion word*; the rule has little to act on.

## Error types

* Emotion present, nothing predicted: 175 texts.
* Emotion predicted but none of the gold emotions: 55 texts.
* Example texts are intentionally omitted (see DATA.md).

_Counts: Counter({0: 249, 1: 188, 2: 57, 3: 5, 4: 1}) predicted labels per text._
