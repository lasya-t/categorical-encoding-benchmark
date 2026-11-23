"""Encoders compared in the benchmark. All follow scikit-learn's fit/transform API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CountEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with how often it appeared in training (log-scaled). Unseen -> 0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.counts_ = {c: X[c].value_counts() for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([np.log1p(X[c].map(self.counts_[c]).fillna(0).to_numpy()) for c in X.columns])


class NaiveTargetEncoder(BaseEstimator, TransformerMixin):
    """Mean target per category, computed on the same rows it is applied to during fit.

    This is the common mistake: each training row's own label leaks into its encoding,
    so rare categories look perfectly predictive in training and the model overfits.
    Smoothing toward the global mean (m) softens it but doesn't remove the leak.
    """

    def __init__(self, m: float = 0.0):
        self.m = m

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.prior_ = y.mean()
        self.maps_ = {}
        for c in X.columns:
            g = pd.DataFrame({"x": X[c].to_numpy(), "y": y}).groupby("x")["y"].agg(["sum", "count"])
            self.maps_[c] = (g["sum"] + self.m * self.prior_) / (g["count"] + self.m)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X[c].map(self.maps_[c]).fillna(self.prior_).to_numpy() for c in X.columns])
