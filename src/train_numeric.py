"""
train_numeric.py - ML Numeric block + cross-block FUSION.

Predicts IMDb `rating` (regression). Two experiments:
  (1) Model comparison on the fused feature set: Ridge vs RandomForest vs
      HistGradientBoosting  (satisfies "compare >=2 models").
  (2) Ablation with the best model: numeric-only vs text-only vs FUSED, where the
      text features come from the NLP block's TitleFeaturizer. Includes a
      cold-start analysis (RMSE for low-vote vs high-vote films).

Reuses the title featurizer fitted in train_nlp.py (models/title_featurizer.joblib).
Run after data_prep.py AND train_nlp.py. Saves:
  models/rating_model.joblib   - deployable FusedRatingModel (used by the app)
  reports/numeric_metrics.json
  reports/figures/07_model_comparison.png, 08_ablation_by_votes.png, 09_pred_vs_true.png
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_processed, GENRES
from nlp_features import TitleFeaturizer
from fused_model import FusedRatingModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
FIG = os.path.join(REPORTS, "figures")
for d in (MODELS, FIG):
    os.makedirs(d, exist_ok=True)
SEED = 42

NUM_FEATS = ["year", "length", "log_votes", "log_budget", "has_budget", "n_genres", "decade"] + GENRES
CAT_FEATS = ["mpaa"]
TITLE_COL = "title"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def make_model(name):
    if name == "ridge":
        return Ridge(alpha=10.0, random_state=SEED)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=40, max_depth=12, min_samples_leaf=30,
                                     n_jobs=-1, random_state=SEED)
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                             l2_regularization=1.0, random_state=SEED)
    raise ValueError(name)


def build_features():
    df = load_processed()
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    numeric_ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])
    numeric_ct.fit(parts["train"][NUM_FEATS + CAT_FEATS])
    featurizer = joblib.load(os.path.join(MODELS, "title_featurizer.joblib"))
    data = {}
    for s, d in parts.items():
        Xnum = numeric_ct.transform(d[NUM_FEATS + CAT_FEATS])
        Xnum = Xnum.toarray() if hasattr(Xnum, "toarray") else np.asarray(Xnum)
        Xtxt = featurizer.transform(d[TITLE_COL].astype(str).tolist())
        data[s] = {"num": Xnum, "txt": Xtxt, "y": d["rating"].values, "votes": d["votes"].values}
    return numeric_ct, featurizer, data, parts


def feats(d, kind):
    if kind == "numeric":
        return d["num"]
    if kind == "text":
        return d["txt"]
    return np.hstack([d["num"], d["txt"]])


def main():
    numeric_ct, featurizer, data, parts = build_features()
    ytr, yva, yte = data["train"]["y"], data["val"]["y"], data["test"]["y"]
    baseline = rmse(yte, np.full_like(yte, ytr.mean()))
    print("Baseline (predict train mean) test RMSE = %.4f" % baseline)

    print("\n=== Model comparison (fused) ===")
    Xtr_f, Xva_f = feats(data["train"], "fused"), feats(data["val"], "fused")
    comp, fused_fitted = {}, {}
    for m in ["ridge", "random_forest", "hist_gbm"]:
        mdl = make_model(m).fit(Xtr_f, ytr)
        pv = mdl.predict(Xva_f)
        comp[m] = {"val_rmse": round(rmse(yva, pv), 4),
                   "val_mae": round(float(mean_absolute_error(yva, pv)), 4),
                   "val_r2": round(float(r2_score(yva, pv)), 4)}
        fused_fitted[m] = mdl
        print("%-16s val RMSE=%.4f MAE=%.4f R2=%.4f" % (m, comp[m]["val_rmse"], comp[m]["val_mae"], comp[m]["val_r2"]))
    best_model = min(comp, key=lambda k: comp[k]["val_rmse"])
    print("BEST model:", best_model)

    print("\n=== Ablation (test) ===")
    ablation, preds = {}, {}
    for kind in ["numeric", "text", "fused"]:
        Xtr_k = feats(data["train"], kind)
        Xte_k = feats(data["test"], kind)
        mdl = fused_fitted[best_model] if kind == "fused" else make_model(best_model).fit(Xtr_k, ytr)
        pt = mdl.predict(Xte_k)
        preds[kind] = pt
        ablation[kind] = {"test_rmse": round(rmse(yte, pt), 4),
                          "test_mae": round(float(mean_absolute_error(yte, pt)), 4),
                          "test_r2": round(float(r2_score(yte, pt)), 4)}
        print("%-8s RMSE=%.4f MAE=%.4f R2=%.4f" % (kind, ablation[kind]["test_rmse"], ablation[kind]["test_mae"], ablation[kind]["test_r2"]))

    print("\n=== Cold-start: RMSE by vote bucket ===")
    votes = data["test"]["votes"]
    coldstart = {}
    for lo, hi, lab in [(0, 20, "<20"), (20, 100, "20-100"), (100, 1000, "100-1k"), (1000, 10**12, ">=1k")]:
        mask = (votes >= lo) & (votes < hi)
        if mask.sum() == 0:
            continue
        rn = rmse(yte[mask], preds["numeric"][mask])
        rf = rmse(yte[mask], preds["fused"][mask])
        coldstart[lab] = {"n": int(mask.sum()), "rmse_numeric": round(rn, 4),
                          "rmse_fused": round(rf, 4), "improvement_pct": round((rn - rf) / rn * 100, 2)}
        print("votes %-7s n=%5d numeric=%.4f fused=%.4f impr=%+.2f%%" % (lab, int(mask.sum()), rn, rf, coldstart[lab]["improvement_pct"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(comp), [comp[n]["val_rmse"] for n in comp], color="#3b7dd8")
    ax.axhline(baseline, ls="--", c="gray", label="mean baseline")
    ax.set_ylabel("val RMSE"); ax.set_title("Model comparison (fused)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "07_model_comparison.png"), dpi=110); plt.close(fig)

    labs = list(coldstart)
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, [coldstart[l]["rmse_numeric"] for l in labs], w, label="numeric-only", color="#9aa0a6")
    ax.bar(x + w/2, [coldstart[l]["rmse_fused"] for l in labs], w, label="fused (+title NLP)", color="#d8743b")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel("vote bucket"); ax.set_ylabel("test RMSE")
    ax.set_title("Cold-start: title NLP helps most for low-vote films"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "08_ablation_by_votes.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yte, preds["fused"], s=4, alpha=0.15, color="#3b7dd8")
    ax.plot([1, 10], [1, 10], "r--"); ax.set_xlabel("true rating"); ax.set_ylabel("predicted (fused)")
    ax.set_title("Predicted vs true (fused, test)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "09_pred_vs_true.png"), dpi=110); plt.close(fig)

    deployable = FusedRatingModel(numeric_ct, featurizer, fused_fitted[best_model], NUM_FEATS, CAT_FEATS, TITLE_COL)
    joblib.dump(deployable, os.path.join(MODELS, "rating_model.joblib"))
    payload = {"baseline_test_rmse": round(baseline, 4), "model_comparison_val": comp,
               "best_model": best_model, "ablation_test": ablation, "coldstart": coldstart,
               "num_feats": NUM_FEATS, "cat_feats": CAT_FEATS}
    with open(os.path.join(REPORTS, "numeric_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved rating_model.joblib + numeric_metrics.json")


if __name__ == "__main__":
    main()
