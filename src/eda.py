"""
eda.py — Exploratory Data Analysis for CineSense.

Produces figures in reports/figures/ and prints key findings used in the
documentation. Run after data_prep.py.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import load_processed, GENRES

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "reports", "figures")
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("saved", path)


def main():
    df = load_processed()
    print(f"Rows: {len(df):,}")

    # 1. Rating distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["rating"], bins=40, color="#3b7dd8", edgecolor="white")
    ax.set_title("IMDb rating distribution"); ax.set_xlabel("rating"); ax.set_ylabel("count")
    save(fig, "01_rating_dist.png")

    # 2. Rating vs log_votes (hexbin)
    fig, ax = plt.subplots(figsize=(6, 4))
    hb = ax.hexbin(df["log_votes"], df["rating"], gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_title("Rating vs log(votes)"); ax.set_xlabel("log1p(votes)"); ax.set_ylabel("rating")
    save(fig, "02_rating_vs_votes.png")

    # 3. Mean rating by genre
    means = {g: df.loc[df[g] == 1, "rating"].mean() for g in GENRES}
    means = pd.Series(means).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(means.index, means.values, color="#d8743b")
    ax.axvline(df["rating"].mean(), ls="--", c="gray", label="overall mean")
    ax.set_title("Mean rating by genre"); ax.set_xlabel("mean rating"); ax.legend()
    save(fig, "03_rating_by_genre.png")

    # 4. Mean rating by decade
    dec = df.groupby("decade")["rating"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dec.index, dec.values, marker="o", color="#3b7dd8")
    ax.set_title("Mean rating by decade"); ax.set_xlabel("decade"); ax.set_ylabel("mean rating")
    save(fig, "04_rating_by_decade.png")

    # 5. Correlation heatmap (numeric)
    numcols = ["rating", "year", "length", "log_votes", "n_genres",
               "title_n_chars", "title_n_words"]
    corr = df[numcols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numcols))); ax.set_xticklabels(numcols, rotation=45, ha="right")
    ax.set_yticks(range(len(numcols))); ax.set_yticklabels(numcols)
    for i in range(len(numcols)):
        for j in range(len(numcols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax); ax.set_title("Numeric feature correlations")
    save(fig, "05_correlation.png")

    # ---- Key findings (printed) ----
    print("\n=== KEY FINDINGS ===")
    print("Correlation rating~log_votes:", round(df["rating"].corr(df["log_votes"]), 3))
    print("Correlation rating~length   :", round(df["rating"].corr(df["length"]), 3))
    print("Correlation rating~year     :", round(df["rating"].corr(df["year"]), 3))
    print("Highest-rated genre :", means.idxmax(), round(means.max(), 3))
    print("Lowest-rated genre  :", means.idxmin(), round(means.min(), 3))
    low = df[df["votes"] < 20]
    print(f"Low-vote films (<20 votes): {len(low):,} ({len(low)/len(df)*100:.1f}%) "
          f"— rating std {low['rating'].std():.2f} vs overall {df['rating'].std():.2f}")


if __name__ == "__main__":
    main()
