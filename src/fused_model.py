"""
fused_model.py — deployable wrapper holding the fitted numeric preprocessor,
the NLP TitleFeaturizer, and the regressor. Lives in its own module so the
joblib artifact can be unpickled by the app.

This object is the concrete realisation of the cross-block integration:
predict() builds structured features AND title-NLP features, concatenates them,
and feeds the single regressor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class FusedRatingModel:
    def __init__(self, numeric_ct, featurizer, model, num_feats, cat_feats, title_col="title"):
        self.numeric_ct = numeric_ct      # fitted ColumnTransformer (num impute+scale, cat OHE)
        self.featurizer = featurizer      # fitted TitleFeaturizer
        self.model = model                # fitted regressor
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.title_col = title_col

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        Xnum = self.numeric_ct.transform(df[self.num_feats + self.cat_feats])
        if hasattr(Xnum, "toarray"):
            Xnum = Xnum.toarray()
        Xtxt = self.featurizer.transform(df[self.title_col].astype(str).tolist())
        return np.hstack([np.asarray(Xnum), np.asarray(Xtxt)])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(self._matrix(df))
        return np.clip(pred, 1.0, 10.0)
