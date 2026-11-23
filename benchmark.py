"""Which way of encoding high-cardinality categoricals works best, for linear models and for trees?"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from data import COLUMNS, make
from encoders import CountEncoder, NaiveTargetEncoder

CATS = list(COLUMNS)
NUMS = ["tenure_months", "monthly_bill"]

ENCODERS = {
    "one-hot": lambda: OneHotEncoder(handle_unknown="ignore", min_frequency=5),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda: CountEncoder(),
    "target, naive (leaky)": lambda: NaiveTargetEncoder(m=0),
    "target, naive + smoothing": lambda: NaiveTargetEncoder(m=20),
    "target, cross-fitted": lambda: TargetEncoder(target_type="binary", cv=5, random_state=0),
}

MODELS = {
    "logistic": lambda: LogisticRegression(max_iter=3000, C=0.5),
    "hist-gboost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
}


def build(enc_name, model_name):
    pre = ColumnTransformer([("cat", ENCODERS[enc_name](), CATS), ("num", StandardScaler(), NUMS)],
                            sparse_threshold=0.0)
    steps = [pre]
    if model_name == "logistic" and enc_name != "one-hot":
        steps.append(StandardScaler())
    return make_pipeline(*steps, MODELS[model_name]())


def run(n=40_000, seed=0):
    df, y = make(n, seed)
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, random_state=seed, stratify=y)
    rows = []
    for model_name in MODELS:
        for enc_name in ENCODERS:
            t0 = time.perf_counter()
            pipe = build(enc_name, model_name).fit(Xtr, ytr)
            secs = time.perf_counter() - t0
            width = pipe[0].transform(Xtr.head(5)).shape[1]
            rows.append({
                "model": model_name, "encoder": enc_name,
                "train AUC": roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1]),
                "test AUC": roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]),
                "features": width, "fit s": secs,
            })
    out = pd.DataFrame(rows)
    out["gap"] = out["train AUC"] - out["test AUC"]
    return out


if __name__ == "__main__":
    df, y = make()
    vc = df["zip_code"].value_counts()
    print(f"{len(df):,} rows, churn rate {y.mean():.1%}; zip_code has {len(vc):,} values, "
          f"{(vc < 5).sum():,} of them seen fewer than 5 times\n")
    res = run()
    pd.set_option("display.width", 140)
    print(res.round(4).to_string(index=False))
