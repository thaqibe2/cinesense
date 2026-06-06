"""
nlp_features.py — NLP title featurizer used by BOTH the numeric (fusion) model
and the deployed app. Kept in its own module so joblib-pickled objects that
reference these classes can be unpickled at inference time.

The featurizer turns a raw movie title into a fixed numeric vector:
  - char n-gram TF-IDF reduced with TruncatedSVD  -> latent "title style" dims
  - VADER sentiment (compound / pos / neg)        -> emotional tone of the title
  - word-rarity features from the wordfreq corpus -> mean Zipf frequency and
    out-of-vocabulary fraction (coined / foreign / exotic titles)
  - simple style stats (length, word count, '?', digits)

Two distinct external data sources are used here, both bundled offline:
  - the VADER sentiment lexicon (`vaderSentiment`)
  - the English word-frequency corpus (`wordfreq`, aggregated from many text
    collections), used to score how common/rare a title's words are.
Both are separate from the IMDb table.
"""
from __future__ import annotations
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordfreq import zipf_frequency

_WORD = re.compile(r"\S+")
_TOKEN = re.compile(r"[A-Za-z']+")


def _word_rarity(title: str):
    """Return (mean Zipf frequency, fraction out-of-vocabulary) for title words."""
    toks = _TOKEN.findall(title.lower())
    if not toks:
        return 0.0, 1.0
    zipfs = [zipf_frequency(w, "en") for w in toks]
    oov = sum(1 for z in zipfs if z == 0.0) / len(zipfs)
    return float(np.mean(zipfs)), float(oov)


def _clean(title: str) -> str:
    return str(title).strip()


class TitleFeaturizer(BaseEstimator, TransformerMixin):
    """Raw titles -> dense numeric matrix (SVD dims + sentiment + style)."""

    def __init__(self, svd_dims: int = 20, ngram=(3, 5), min_df: int = 5, random_state: int = 42):
        self.svd_dims = svd_dims
        self.ngram = ngram
        self.min_df = min_df
        self.random_state = random_state

    def fit(self, X, y=None):
        titles = [_clean(t) for t in X]
        self.tfidf_ = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram,
                                      min_df=self.min_df, lowercase=True)
        Z = self.tfidf_.fit_transform(titles)
        self.svd_ = TruncatedSVD(n_components=self.svd_dims, random_state=self.random_state)
        self.svd_.fit(Z)
        self.sia_ = SentimentIntensityAnalyzer()
        self.feature_names_ = (
            [f"svd_{i}" for i in range(self.svd_dims)]
            + ["vader_compound", "vader_pos", "vader_neg",
               "title_mean_zipf", "title_frac_oov",
               "title_n_chars", "title_n_words", "title_is_question", "title_has_number"]
        )
        return self

    def _style(self, titles):
        rows = []
        for t in titles:
            s = self.sia_.polarity_scores(t)
            mean_zipf, frac_oov = _word_rarity(t)
            rows.append([
                s["compound"], s["pos"], s["neg"],
                mean_zipf, frac_oov,
                len(t), len(_WORD.findall(t)),
                1 if "?" in t else 0,
                1 if re.search(r"\d", t) else 0,
            ])
        return np.asarray(rows, dtype=float)

    def transform(self, X):
        titles = [_clean(t) for t in X]
        Z = self.tfidf_.transform(titles)
        svd = self.svd_.transform(Z)
        style = self._style(titles)
        return np.hstack([svd, style])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
