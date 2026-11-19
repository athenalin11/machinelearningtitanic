# Person B experiment results

Smoke test: False. Dataset B, 25 outer folds.

| Model | Accuracy mean +/- std | F1 | AUC | Train accuracy |
|---|---:|---:|---:|---:|
| knn_unscaled | 0.7827 +/- 0.0285 | 0.7056 | 0.8196 | 0.9045 |
| knn_scaled | 0.8229 +/- 0.0174 | 0.7553 | 0.8615 | 0.8671 |

Scaled minus unscaled kNN mean accuracy difference: +0.0401.
This compares independently inner-tuned procedures on paired outer folds, not a fixed-hyperparameter effect.
See the uniform-vote K curves for a controlled comparison at identical K and distance.

Selected parameter frequencies:

model         selected_params                                     
knn_scaled    {"clf__n_neighbors": 11, "clf__weights": "uniform"}     3
              {"clf__n_neighbors": 15, "clf__weights": "uniform"}     5
              {"clf__n_neighbors": 21, "clf__weights": "uniform"}     1
              {"clf__n_neighbors": 31, "clf__weights": "uniform"}     2
              {"clf__n_neighbors": 5, "clf__weights": "uniform"}      3
              {"clf__n_neighbors": 51, "clf__weights": "distance"}    3
              {"clf__n_neighbors": 7, "clf__weights": "uniform"}      5
              {"clf__n_neighbors": 9, "clf__weights": "uniform"}      3
knn_unscaled  {"clf__n_neighbors": 3, "clf__weights": "distance"}     1
              {"clf__n_neighbors": 3, "clf__weights": "uniform"}      5
              {"clf__n_neighbors": 5, "clf__weights": "distance"}     3
              {"clf__n_neighbors": 5, "clf__weights": "uniform"}      4
              {"clf__n_neighbors": 7, "clf__weights": "distance"}     4
              {"clf__n_neighbors": 7, "clf__weights": "uniform"}      5
              {"clf__n_neighbors": 9, "clf__weights": "distance"}     2
              {"clf__n_neighbors": 9, "clf__weights": "uniform"}      1

Limitations:
- B preprocessing is precomputed on train/test covariates, not refitted in folds.
- StandardScaler is fitted only within inner/outer training folds.
- Scaled/unscaled KNN are tuned separately on identical outer/inner splits.
- K curves hold Euclidean distance and uniform weights fixed; use inner scores only.
- GaussianNB receives unscaled features; smoothing is relative to largest feature variance.
- Repeated-fold standard deviation is not a confidence interval; no significance test.
- No final full-data model or Kaggle submission is generated.
- Runtime differs from Person C sklearn 1.6.1; compare descriptively.
