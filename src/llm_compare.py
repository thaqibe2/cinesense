"""
llm_compare.py - compare two NLP approaches for title -> genre:
  (A) the classical char-TFIDF + Logistic Regression classifier, and
  (B) an LLM (OpenAI) prompted with only the title.

Both are scored against the true genre flags on a SMALL, capped sample of test
films (default 60, hard max 200) to keep API cost near zero - this does NOT
scan the full dataset. Saves reports/llm_comparison.md.

Usage:
    setx-or-export OPENAI_API_KEY ...   # must be set
    python src/llm_compare.py [N]
"""
import os, sys, json, time
import numpy as np
import joblib
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_processed, GENRES
import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARD_MAX = 200


def to_vec(genres):
    return [1 if g in genres else 0 for g in GENRES]


def classical_genres(model, title, threshold=0.40):
    probs = model.predict_proba([title])[0]
    pairs = sorted(zip(GENRES, probs), key=lambda x: -x[1])
    return [g for g, p in pairs if p >= threshold] or [pairs[0][0]]


def main():
    if not llm.llm_available():
        print("OPENAI_API_KEY not set (or openai not installed). "
              "Set the key and re-run. Nothing was sent to the API."); return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n = max(5, min(n, HARD_MAX))
    print(f"Comparing on {n} sampled test titles using model '{llm.DEFAULT_MODEL}' "
          f"(approx cost < $0.01)...")

    df = load_processed()
    te = df[df.split == "test"].reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(te), size=n, replace=False)
    sample = te.iloc[idx].reset_index(drop=True)

    genre_model = joblib.load(os.path.join(ROOT, "models", "nlp_genre.joblib"))

    y_true, y_cls, y_llm, rows = [], [], [], []
    for r in sample.itertuples():
        true_g = [g for g in GENRES if getattr(r, g) == 1]
        cls_g = classical_genres(genre_model, r.title)
        llm_g = llm.llm_genres_for_title(r.title) or []
        y_true.append(to_vec(true_g)); y_cls.append(to_vec(cls_g)); y_llm.append(to_vec(llm_g))
        rows.append((r.title, true_g, cls_g, llm_g))
        time.sleep(0.05)

    y_true, y_cls, y_llm = np.array(y_true), np.array(y_cls), np.array(y_llm)
    def scores(yp):
        return (round(f1_score(y_true, yp, average="micro", zero_division=0), 4),
                round(f1_score(y_true, yp, average="macro", zero_division=0), 4))
    cls_micro, cls_macro = scores(y_cls)
    llm_micro, llm_macro = scores(y_llm)

    lines = [f"# NLP approach comparison: classical vs LLM (title -> genre)\n",
             f"Sample: {n} random test films. LLM model: `{llm.DEFAULT_MODEL}`.\n",
             "| Approach | micro-F1 | macro-F1 |", "| --- | --- | --- |",
             f"| Classical (char-TFIDF + LogReg) | {cls_micro} | {cls_macro} |",
             f"| LLM (prompted with title only) | {llm_micro} | {llm_macro} |",
             "\n## Example predictions\n",
             "| Title | True | Classical | LLM |", "| --- | --- | --- | --- |"]
    for t, tg, cg, lg in rows[:12]:
        lines.append(f"| {t} | {', '.join(tg) or '-'} | {', '.join(cg) or '-'} | {', '.join(lg) or '-'} |")
    lines.append("\n**Takeaway:** both approaches face the same hard ceiling - a bare title "
                 "is weak evidence of genre. The LLM uses world knowledge of real titles, while "
                 "the classical model only sees character patterns; the table shows where each wins.")
    out = os.path.join(ROOT, "reports", "llm_comparison.md")
    open(out, "w", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines[:6])); print("\nSaved", out)


if __name__ == "__main__":
    main()
