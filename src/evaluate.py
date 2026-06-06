"""
evaluate.py - deeper evaluation & error analysis for CineSense.

Adds beyond train_numeric.py:
  (1) Realistic "sparse-metadata" scenario: a brand-new film where votes and
      genres are unknown. We blank those at inference and compare numeric-only
      vs fused -> shows the title-NLP block carries real weight when metadata
      is absent (the app's actual use case).
  (2) Feature-group permutation importance (numeric vs title-NLP groups).
  (3) Error analysis: largest residuals.
  (4) Fairness check: RMSE for non-Latin-script and very short titles.

Run after train_nlp.py + train_numeric.py. Saves reports/eval_metrics.json and
figures 10-12.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_numeric import build_features, make_model, feats, NUM_FEATS, GENRES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
FIG = os.path.join(REPORTS, "figures")


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    numeric_ct, featurizer, data, parts = build_features()
    ytr, yte = data["train"]["y"], data["test"]["y"]
    Xtr_num, Xtr_f = feats(data["train"], "numeric"), feats(data["train"], "fused")
    Xte_num, Xte_f = feats(data["test"], "numeric"), feats(data["test"], "fused")

    m_num = make_model("hist_gbm").fit(Xtr_num, ytr)
    m_fus = make_model("hist_gbm").fit(Xtr_f, ytr)
    out = {}

    # (1) Sparse-metadata scenario: blank votes (->train median) and genres (->0).
    te = parts["test"].copy()
    log_votes_idx = NUM_FEATS.index("log_votes")
    med_lv = float(np.median(data["train"]["num"][:, log_votes_idx]))  # already standardized space? no:
    # NOTE: data["..."]["num"] is already transformed (scaled). To blank in raw space we rebuild.
    te_sparse = te.copy()
    te_sparse["log_votes"] = parts["train"]["log_votes"].median()
    te_sparse["votes"] = int(parts["train"]["votes"].median())
    for g in GENRES:
        te_sparse[g] = 0
    te_sparse["n_genres"] = 0
    # transform sparse rows
    Xnum_s = numeric_ct.transform(te_sparse[NUM_FEATS + ["mpaa"]])
    Xnum_s = Xnum_s.toarray() if hasattr(Xnum_s, "toarray") else np.asarray(Xnum_s)
    Xtxt_s = featurizer.transform(te_sparse["title"].astype(str).tolist())
    Xf_s = np.hstack([Xnum_s, Xtxt_s])
    r_num_s = rmse(yte, m_num.predict(Xnum_s))
    r_fus_s = rmse(yte, m_fus.predict(Xf_s))
    out["sparse_metadata_scenario"] = {
        "desc": "votes & genres unknown (new/unreleased film)",
        "rmse_numeric_only": round(r_num_s, 4),
        "rmse_fused": round(r_fus_s, 4),
        "improvement_pct": round((r_num_s - r_fus_s) / r_num_s * 100, 2)}
    print("Sparse-metadata: numeric=%.4f fused=%.4f impr=%+.2f%%" %
          (r_num_s, r_fus_s, out["sparse_metadata_scenario"]["improvement_pct"]))

    # (2) Permutation importance by GROUP (numeric block vs title-NLP block).
    rng = np.random.default_rng(42)
    idx = rng.choice(len(yte), size=min(3000, len(yte)), replace=False)
    Xs, ys = Xte_f[idx], yte[idx]
    n_num = Xte_num.shape[1]
    base = rmse(ys, m_fus.predict(Xs))
    def group_importance(cols):
        Xp = Xs.copy()
        for c in cols:
            Xp[:, c] = rng.permutation(Xp[:, c])
        return rmse(ys, m_fus.predict(Xp)) - base
    imp_numeric = group_importance(list(range(n_num)))
    imp_title = group_importance(list(range(n_num, Xs.shape[1])))
    out["group_permutation_importance_rmse_increase"] = {
        "numeric_block": round(imp_numeric, 4), "title_nlp_block": round(imp_title, 4)}
    print("Group importance (RMSE increase): numeric=%.4f title-NLP=%.4f" % (imp_numeric, imp_title))

    # (3) Largest residuals (fused).
    pred = m_fus.predict(Xte_f)
    te["pred"] = pred; te["resid"] = te["rating"] - te["pred"]
    worst = te.reindex(te["resid"].abs().sort_values(ascending=False).index).head(8)
    out["largest_residuals"] = [
        {"title": str(r.title), "year": int(r.year), "votes": int(r.votes),
         "true": float(r.rating), "pred": round(float(r.pred), 2)}
        for r in worst.itertuples()]

    # (4) Fairness: non-Latin script titles & very short titles.
    def non_latin(t):
        return any(ord(ch) > 591 for ch in str(t))
    te["non_latin"] = te["title"].apply(non_latin)
    te["short_title"] = te["title_n_words"] <= 1
    grp = {}
    for name, mask in [("latin_script", ~te.non_latin), ("non_latin_script", te.non_latin),
                       ("multi_word_title", ~te.short_title), ("single_word_title", te.short_title)]:
        seg = te[mask]
        grp[name] = {"n": int(len(seg)),
                     "rmse_fused": round(rmse(seg.rating, seg.pred), 4) if len(seg) else None}
    out["fairness_by_title_type"] = grp
    print("Fairness:", {k: v["rmse_fused"] for k, v in grp.items()})

    # Figures
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["numeric\nblock", "title-NLP\nblock"], [imp_numeric, imp_title], color=["#9aa0a6", "#d8743b"])
    ax.set_ylabel("RMSE increase when shuffled"); ax.set_title("Feature-group importance (fused)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "10_group_importance.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["numeric-only", "fused"], [r_num_s, r_fus_s], color=["#9aa0a6", "#d8743b"])
    ax.set_ylabel("test RMSE"); ax.set_title("Sparse-metadata scenario\n(votes & genres unknown)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "11_sparse_scenario.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(te["resid"], bins=50, color="#3b7dd8", edgecolor="white")
    ax.set_xlabel("residual (true - pred)"); ax.set_title("Residual distribution (fused, test)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "12_residuals.png"), dpi=110); plt.close(fig)

    with open(os.path.join(REPORTS, "eval_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved eval_metrics.json + figures 10-12")


if __name__ == "__main__":
    main()
