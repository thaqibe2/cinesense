"""
data_prep.py — Load, clean, and feature-engineer the IMDb `movies` dataset.

Data source: the `movies` table bundled with the `pydataset` package
(the classic ggplot2 `movies` dataset, ~58k films scraped from IMDb).
It is loaded fully offline, which keeps the whole project reproducible.

IMPORTANT (leakage): columns r1..r10 hold the percentage of votes that
awarded each score 1..10. They are a direct decomposition of `rating`
(the target) and are therefore dropped — using them would be label leakage.

Outputs:
  data/processed.csv  — cleaned table with engineered columns and a `split`
                        column (train/val/test) for fully reproducible runs.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA_DIR, "processed.csv")

GENRES = ["Action", "Animation", "Comedy", "Drama", "Documentary", "Romance", "Short"]
LEAKAGE_COLS = [f"r{i}" for i in range(1, 11)]


def load_raw() -> pd.DataFrame:
    """Load the raw movies table from pydataset (offline)."""
    from pydataset import data as pyd
    return pyd("movies").reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop the rating-distribution columns (label leakage).
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    # Basic sanity filtering.
    df = df[(df["length"] > 0) & (df["length"] <= 600)]      # drop 0-min and absurd runtimes
    df = df[(df["year"] >= 1900) & (df["year"] <= 2010)]
    df = df[df["votes"] >= 5]                                 # need a minimum of votes to trust a rating
    df = df[df["title"].astype(str).str.strip().str.len() > 0]
    df = df.reset_index(drop=True)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Numeric engineered features.
    df["log_votes"] = np.log1p(df["votes"])
    df["log_budget"] = np.log1p(df["budget"])          # mostly NaN; HistGBM handles it, others imputed
    df["has_budget"] = df["budget"].notna().astype(int)
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["n_genres"] = df[GENRES].sum(axis=1)
    df["mpaa"] = df["mpaa"].fillna("None").astype(str)

    # Title-derived light text stats (used by the NLP block too).
    t = df["title"].astype(str)
    df["title_n_chars"] = t.str.len()
    df["title_n_words"] = t.str.split().apply(len)
    df["title_is_question"] = t.str.contains(r"\?", regex=True).astype(int)
    df["title_has_number"] = t.str.contains(r"\d", regex=True).astype(int)
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified-by-rating-bucket 70/15/15 train/val/test split."""
    from sklearn.model_selection import train_test_split
    df = df.copy()
    strat = pd.cut(df["rating"], bins=[0, 4, 5, 6, 7, 8, 10], labels=False, include_lowest=True)
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=strat[tmp])
    split = np.array(["train"] * len(df), dtype=object)
    split[va] = "val"
    split[te] = "test"
    df["split"] = split
    return df


def build() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_raw()
    df = clean(df)
    df = engineer(df)
    df = make_splits(df)
    df.to_csv(PROCESSED, index=False)
    return df


def load_processed() -> pd.DataFrame:
    """Load processed data, building it on first call."""
    if not os.path.exists(PROCESSED):
        return build()
    return pd.read_csv(PROCESSED)


if __name__ == "__main__":
    df = build()
    print(f"Processed rows: {len(df):,}  cols: {df.shape[1]}")
    print("Split counts:", df["split"].value_counts().to_dict())
    print("Rating mean/std:", round(df.rating.mean(), 3), round(df.rating.std(), 3))
    print("Saved ->", PROCESSED)
