# Person B experiment results

Smoke test: False. Dataset B, 25 outer folds.

| Model | Accuracy mean +/- std | F1 | AUC | Train accuracy |
|---|---:|---:|---:|---:|
| gaussian_nb | 0.7960 +/- 0.0305 | 0.7441 | 0.8546 | 0.8086 |

Selected parameter frequencies:

model        selected_params          
gaussian_nb  {"var_smoothing": 0.0001}    16
             {"var_smoothing": 0.001}      3
             {"var_smoothing": 1e-05}      3
             {"var_smoothing": 1e-06}      1
             {"var_smoothing": 1e-12}      2

Limitations:
- B preprocessing is precomputed on train/test covariates, not refitted in folds.
- StandardScaler is fitted only within inner/outer training folds.
- Scaled/unscaled KNN are tuned separately on identical outer/inner splits.
- K curves hold Euclidean distance and uniform weights fixed; use inner scores only.
- GaussianNB receives unscaled features; smoothing is relative to largest feature variance.
- Repeated-fold standard deviation is not a confidence interval; no significance test.
- No final full-data model or Kaggle submission is generated.
- Runtime differs from Person C sklearn 1.6.1; compare descriptively.
