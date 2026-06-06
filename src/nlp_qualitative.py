"""Generate representative title->genre predictions and failure cases (qualitative
NLP evaluation). Saves reports/nlp_qualitative.md."""
import os, sys
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_processed, GENRES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf = joblib.load(os.path.join(ROOT, "models", "nlp_genre.joblib"))
df = load_processed(); te = df[df.split == "test"].reset_index(drop=True)


def top_genres(title, k=2, thr=0.4):
    p = clf.predict_proba([title])[0]
    pairs = sorted(zip(GENRES, p), key=lambda x: -x[1])
    chosen = [g for g, v in pairs if v >= thr] or [pairs[0][0]]
    return chosen, pairs[:k]


def true_genres(row):
    return [g for g in GENRES if row[g] == 1] or ["(none flagged)"]


lines = ["# NLP qualitative analysis: title -> genre\n",
         "Best model: char n-gram TF-IDF + One-vs-Rest Logistic Regression. "
         "Predicted genres use threshold 0.40.\n",
         "## Hand-picked illustrative titles\n",
         "| Title | Predicted (top probs) |", "| --- | --- |"]
for t in ["The Last Silent Romance", "Robo Death Killer 3000", "A Quiet Documentary About Bees",
          "Love Actually Forever", "Galactic War Machine", "Mr. Bean's Holiday Cartoon"]:
    chosen, probs = top_genres(t)
    pp = ", ".join(f"{g} {v:.2f}" for g, v in probs)
    lines.append(f"| {t} | {', '.join(chosen)} ({pp}) |")

# Correct vs failure cases from the real test set
rng = np.random.default_rng(7)
hits, misses = [], []
for i in rng.choice(len(te), 400, replace=False):
    row = te.iloc[i]
    chosen, _ = top_genres(row["title"])
    tg = set(true_genres(row)); pg = set(chosen)
    rec = (row["title"], sorted(tg), sorted(pg))
    if tg & pg and len(hits) < 6:
        hits.append(rec)
    elif not (tg & pg) and len(misses) < 6 and "(none flagged)" not in tg:
        misses.append(rec)

lines += ["\n## Correct predictions (real test films)\n", "| Title | True genres | Predicted |", "| --- | --- | --- |"]
for t, tg, pg in hits:
    lines.append(f"| {t} | {', '.join(tg)} | {', '.join(pg)} |")
lines += ["\n## Failure cases (no overlap with true genres)\n", "| Title | True genres | Predicted |", "| --- | --- | --- |"]
for t, tg, pg in misses:
    lines.append(f"| {t} | {', '.join(tg)} | {', '.join(pg)} |")
lines += ["\n**Observation:** the classifier latches onto lexical cues "
          "(\"romance\", \"war\", \"documentary\", cartoon-ish words) and does well on "
          "Drama/Comedy, but short or genre-neutral titles (proper names, abstract "
          "words) are underdetermined, which drives most failures and the low Romance F1."]

out = os.path.join(ROOT, "reports", "nlp_qualitative.md")
open(out, "w", newline="\n").write("\n".join(lines) + "\n")
print("saved", out); print("\n".join(lines[:20]))
