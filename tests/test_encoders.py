import numpy as np
import pandas as pd

from data import make
from encoders import CountEncoder, NaiveTargetEncoder


def test_count_encoder_unseen_is_zero():
    enc = CountEncoder().fit(pd.DataFrame({"c": ["a", "a", "b"]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "new"]})).ravel()
    np.testing.assert_allclose(out, np.log1p([2, 1, 0]))


def test_target_encoder_means_and_prior():
    X = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    enc = NaiveTargetEncoder().fit(X, [1, 1, 0, 1])
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "zzz"]})).ravel()
    np.testing.assert_allclose(out, [1.0, 0.5, 0.75])


def test_smoothing_pulls_rare_categories_to_prior():
    X = pd.DataFrame({"c": ["rare"] + ["common"] * 99})
    y = [1] + [0] * 99
    raw = NaiveTargetEncoder(m=0).fit(X, y).transform(pd.DataFrame({"c": ["rare"]}))[0, 0]
    smooth = NaiveTargetEncoder(m=20).fit(X, y).transform(pd.DataFrame({"c": ["rare"]}))[0, 0]
    assert raw == 1.0 and smooth < 0.1


def test_naive_target_encoding_leaks_on_noise():
    # a pure-noise ID column looks predictive in training when encoded on its own labels
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"id": rng.integers(0, 5000, 10000).astype(str)})
    y = rng.integers(0, 2, 10000)
    enc = NaiveTargetEncoder().fit(X, y).transform(X).ravel()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, enc) > 0.75  # pure noise, yet "predictive"


def test_data_shapes():
    df, y = make(2000)
    assert len(df) == 2000 and set(np.unique(y)) == {0, 1}
