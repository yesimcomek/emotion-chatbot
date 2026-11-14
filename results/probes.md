# Probe sentences (hand-written, n = 24)

Exact match of the predicted label set against the expected one (empty set = neutral / emotion absent).

**Caveat:** the same author wrote the lexicon and these sentences, so the rule-based score is partly circular. Read this as a qualitative contrast (explicit statements and negation vs. benchmark-style text), not as a ranking of the models.

| Group | rules | tfidf |
|---|---|---|
| explicit | 11/12 | 5/12 |
| negation | 6/6 | 0/6 |
| neutral | 6/6 | 3/6 |

## Per-sentence predictions

| Sentence | Expected | rules | tfidf |
|---|---|---|---|
| Ich bin so glücklich, dass ich endlich bestanden habe! | joy | joy | joy |
| Das ist wunderbar, ich freue mich riesig. | joy | joy | joy |
| Ich bin unglaublich wütend auf meinen Chef. | anger | anger, surprise | – |
| Das ist eine Frechheit, ich bin sauer. | anger | anger | disgust, fear |
| Ich bin traurig, weil mein Freund wegzieht. | sadness | sadness | joy, sadness |
| Ich fühle mich einsam und niedergeschlagen. | sadness | sadness | sadness |
| Ich habe große Angst vor der Operation. | fear | fear | fear |
| Ich bin sehr nervös und besorgt. | fear | fear | joy, sadness |
| Das ist ekelhaft, ich muss würgen. | disgust | disgust | anger, disgust |
| Wie widerlich, das Essen riecht verdorben. | disgust | disgust | disgust |
| Wow, damit habe ich überhaupt nicht gerechnet! | surprise | surprise | joy, surprise |
| Das ist ja unfassbar, ich bin total überrascht. | surprise | surprise | joy, surprise |
| Ich bin nicht traurig. | – | – | sadness |
| Ich habe keine Angst. | – | – | fear, joy |
| Ich bin gar nicht wütend. | – | – | anger, fear |
| Ich bin nicht glücklich damit. | – | – | joy |
| Das ist nicht ekelhaft. | – | – | anger, disgust |
| Ich bin nicht überrascht. | – | – | joy, surprise |
| Der Zug fährt um acht Uhr ab. | – | – | anger |
| Morgen ist Dienstag. | – | – | – |
| Die Besprechung findet im dritten Stock statt. | – | – | – |
| Ich habe Brot und Milch gekauft. | – | – | joy |
| Das Paket wiegt zwei Kilogramm. | – | – | – |
| Die Seite hat zwölf Kapitel. | – | – | disgust |
