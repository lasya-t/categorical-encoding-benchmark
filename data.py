"""Synthetic churn data with categorical columns of very different cardinality.

Each category gets a hidden effect on churn. High-cardinality columns have a
long tail of rare categories, which is where encoders differ most.
"""
import numpy as np
import pandas as pd

COLUMNS = {
    # name: (number of categories, strength of effect, zipf skew)
    "channel": (3, 0.6, 1.0),
    "plan": (12, 0.8, 1.2),
    "device_model": (80, 0.7, 1.3),
    "store_id": (600, 0.9, 1.1),
    "zip_code": (3000, 0.9, 1.05),
}


def make(n: int = 40_000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"tenure_months": rng.integers(1, 72, n), "monthly_bill": rng.gamma(4, 20, n).round(2)})
    logit = -2.2 - 0.03 * df["tenure_months"].to_numpy() + 0.006 * df["monthly_bill"].to_numpy()
    for col, (k, strength, skew) in COLUMNS.items():
        weights = 1 / np.arange(1, k + 1) ** skew
        cats = rng.choice(k, n, p=weights / weights.sum())
        effects = rng.normal(0, strength, k)
        df[col] = pd.Series([f"{col}_{c}" for c in cats])
        logit = logit + effects[cats]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y
