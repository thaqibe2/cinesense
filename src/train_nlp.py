"""
train_nlp.py — NLP block: predict film GENRES from the title alone (multi-label),
comparing three classical NLP approaches, and fit the TitleFeaturizer that the
numeric block consumes for fusion.

Run after data_prep.py. Saves:
  models/nlp_genre.joblib        — best title->genre multi-label classifier
  models/title_featurizer.joblib — fitted TitleFeaturizer for fusion features
  reports/nlp_metrics.json       — comparison metrics
  reports/figures/06_nlp_per_genre_f1.png
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_processed, GENRES
from nlp_features import TitleFeaturizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
FIG = os.path.join(REPORTS, "figures")
for d in (MODELS, FIG):
    os.makedirs(d, exist_ok=True)
SEED = 42


def build_candidates():
    """Three classical NLP approaches for title -> multi-label genre."""
    return {
        "word_tfidf_logreg": Pipeline([
            ("vec", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=5, lowercase=True)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=400, class_weight="balanced", C=3.0))),
        ]),
        "char_tfidf_logreg": Pipeline([
            ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, lowercase=True)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=400, class_weight="balanced", C=3.0))),
        ]),
        "count_nb": Pipeline([
            ("vec", CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=5, lowercase=True)),
            ("clf", OneVsRestClassifier(MultinomialNB())),
        ]),
    }


def main():
    df = load_processed()
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    Xtr, Ytr = tr["title"].astype(str), tr[GENRES].values
    Xva, Yva = va["title"].astype(str), va[GENRES].values
    Xte, Yte = te["title"].astype(str), te[GENRES].values

    results = {}
    fitted = {}
    for name, pipe in build_candidates().items():
        pipe.fit(Xtr, Ytr)
        pred_va = pipe.predict(Xva)
        results[name] = {
            "val_micro_f1": round(f1_score(Yva, pred_va, average="micro", zero_division=0), 4),
            "val_macro_f1": round(f1_score(Yva, pred_va, average="macro", zero_division=0), 4),
        }
        fitted[name] = pipe
        print(f"{name:22s} val micro-F1={results[name]['val_micro_f1']:.4f} "
              f"macro-F1={results[name]['val_macro_f1']:.4f}")

    best = max(results, key=lambda k: results[k]["val_micro_f1"])
    print("BEST:", best)

    # Final test metrics for the best model.
    best_pipe = fitted[best]
    pred_te = best_pipe.predict(Xte)
    test_micro = round(f1_score(Yte, pred_te, average="micro", zero_division=0), 4)
    test_macro = round(f1_score(Yte, pred_te, average="macro", zero_division=0), 4)
    per_genre = f1_score(Yte, pred_te, average=None, zero_division=0)
    per_genre = {g: round(float(s), 4) for g, s in zip(GENRES, per_genre)}
    print("TEST micro-F1", test_micro, "macro-F1", test_macro)
    print("per-genre F1:", per_genre)

    # Per-genre F1 figure
    fig, ax = plt.subplots(figsize=(6, 4))
    s = pd.Series(per_genre).sort_values()
    ax.barh(s.index, s.values, color="#5b8c5a")
    ax.set_title(f"Title->genre per-genre F1 (test) — {best}")
    ax.set_xlabel("F1"); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "06_nlp_per_genre_f1.png"), dpi=110)
    plt.close(fig)

    # Fit fusion featurizer on TRAIN titles only (no leakage).
    feat = TitleFeaturizer(svd_dims=20, ngram=(3, 5), min_df=5, random_state=SEED)
    feat.fit(Xtr)

    joblib.dump(best_pipe, os.path.join(MODELS, "nlp_genre.joblib"))
    joblib.dump(feat, os.path.join(MODELS, "title_featurizer.joblib"))
    with open(os.path.join(REPORTS, "nlp_metrics.json"), "w") as f:
        json.dump({"comparison": results, "best": best,
                   "test_micro_f1": test_micro, "test_macro_f1": test_macro,
                   "per_genre_test_f1": per_genre}, f, indent=2)
    print("Saved nlp_genre.joblib, title_featurizer.joblib, nlp_metrics.json")


if __name__ == "__main__":
    main()
