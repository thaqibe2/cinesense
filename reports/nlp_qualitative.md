# NLP qualitative analysis: title -> genre

Best model: char n-gram TF-IDF + One-vs-Rest Logistic Regression. Predicted genres use threshold 0.40.

## Hand-picked illustrative titles

| Title | Predicted (top probs) |
| --- | --- |
| The Last Silent Romance | Action, Drama, Romance, Comedy (Action 0.62, Drama 0.51) |
| Robo Death Killer 3000 | Action (Action 0.97, Comedy 0.27) |
| A Quiet Documentary About Bees | Documentary, Short (Documentary 0.95, Short 0.61) |
| Love Actually Forever | Romance, Comedy, Drama (Romance 0.90, Comedy 0.83) |
| Galactic War Machine | Documentary, Action, Short (Documentary 0.79, Action 0.76) |
| Mr. Bean's Holiday Cartoon | Comedy, Short, Animation, Action (Comedy 0.95, Short 0.69) |

## Correct predictions (real test films)

| Title | True genres | Predicted |
| --- | --- | --- |
| Night of Henna | Comedy, Romance | Comedy, Drama, Romance |
| Gycklarnas afton | Drama | Drama |
| Day Without a Mexican, A | Short | Animation, Comedy, Romance, Short |
| Buena onda | Drama, Short | Comedy, Drama |
| Cachorros | Drama, Short | Comedy, Drama, Short |
| Deux anglaises et le continent, Les | Drama, Romance | Documentary, Drama, Short |

## Failure cases (no overlap with true genres)

| Title | True genres | Predicted |
| --- | --- | --- |
| Touch Me | Short | Comedy, Drama, Romance |
| Sub Down | Drama | Action, Animation, Romance, Short |
| 12 mesyatsev | Animation | Comedy, Drama, Short |
| Blokpost | Drama | Comedy, Romance |
| Zwischensaison | Comedy | Documentary, Drama |
| Pep Squad | Comedy | Action, Drama |

**Observation:** the classifier latches onto lexical cues ("romance", "war", "documentary", cartoon-ish words) and does well on Drama/Comedy, but short or genre-neutral titles (proper names, abstract words) are underdetermined, which drives most failures and the low Romance F1.
