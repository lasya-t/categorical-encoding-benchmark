# Categorical Encoding Benchmark

How should high-cardinality categoricals like ZIP code, store ID or device model be turned into numbers? This benchmark compares six encoders with a linear model and a tree model on churn data where most ZIP codes are rare.

## Data

40,000 synthetic customers with a 21% churn rate. Five categorical columns, each with hidden effects on churn:

| Column | Categories |
|---|---|
| channel | 3 |
| plan | 12 |
| device_model | 80 |
| store_id | 600 |
| zip_code | 2,667 (1,683 of them seen fewer than 5 times) |

There are also two numeric columns (tenure and monthly bill). 70/30 train/test split.

## Encoders

| Encoder | How it works |
|---|---|
| one-hot | one column per category; categories seen fewer than 5 times are grouped together |
| ordinal | an arbitrary integer per category |
| count | log of how often the category appeared |
| **target, naive** | mean churn per category, computed on the same rows the model trains on |
| target, naive + smoothing | the same, shrunk toward the overall mean with weight m=20 |
| **target, cross-fitted** | scikit-learn's `TargetEncoder`: each training row is encoded using the other folds only |

## Results (test AUC)

| Encoder | Logistic | HistGBoost | Columns |
|---|---|---|---|
| one-hot | **0.831** | 0.814 | 1,329 |
| ordinal | 0.645 | 0.786 | 7 |
| count | 0.681 | 0.789 | 7 |
| target, naive (leaky) | 0.799 | **0.747** ↓ | 7 |
| target, naive + smoothing | 0.825 | 0.808 | 7 |
| target, cross-fitted | 0.804 | **0.820** | 7 |

## What this shows

- **Naive target encoding leaks the label.** A training row's own churn value is part of its category mean, so rare ZIP codes look perfectly predictive in training. With gradient boosting, train AUC hit 0.910 but test AUC fell to 0.747, the worst of any encoder for that model. One test shows the leak directly: a **pure-noise ID column scores AUC > 0.75** once target-encoded this way.
- **Cross-fitting fixes the leak.** It's the best option for trees (0.820), with only 7 columns instead of 1,329, and it fits 60× faster than one-hot (0.5 s vs 30 s).
- **For the linear model, one-hot wins.** Here the data was generated with an additive effect per category, which is exactly what one-hot plus logistic regression can represent. Smoothed target encoding comes close.
- **Ordinal and count encoding are poor for linear models** because the numbers they assign have no meaningful order. Trees can still split them apart, so the gap is smaller.
- Caveat: the "train AUC" for `TargetEncoder` is computed with the full-data encoding at predict time, so its train/test gap overstates any overfitting. The test AUC is the number to trust.

Results are from a single split. Small differences (under about 0.01 AUC) shouldn't be over-read.

## How to run

```bash
pip install -r requirements.txt
python benchmark.py     # ~1 minute
pytest
```

## Next steps
- Add CatBoost's ordered target statistics, plus hashing and embedding encoders
- Average over several seeds and report confidence intervals
