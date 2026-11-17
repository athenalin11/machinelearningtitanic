"""Person C, Phase 1: Decision Tree nested CV on the team's existing dataset B.

Run from any directory. --smoke-test uses two outer folds and a reduced grid,
writing only to smoke/ subdirectories. The default runs the agreed 25 folds.
Depth curves aggregate INNER-CV training/validation accuracy, not outer test
scores used for parameter selection. Importance is averaged over outer models.
Dataset B's precomputed, label-free preprocessing is retained for comparability.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/B_v4_FamilyFlags_AgeTitlePclassParch_train.csv"
DEPTHS = [2, 3, 4, 5, 6, 8, None]
LEAVES = [1, 5, 10, 20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    frame = pd.read_csv(DATA)
    X = frame.drop(columns=["PassengerId", "Survived", "fold"])
    y = frame["Survived"]
    if X.shape != (891, 23) or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Expected 891 rows, 23 finite numeric dataset-B features")
    if set(y.unique()) != {0, 1} or not frame.PassengerId.is_unique:
        raise ValueError("Invalid target or duplicate PassengerId")
    out = ROOT / "team/results/person_c"
    figs = ROOT / "report/figures"
    if args.smoke_test:
        out, figs = out / "smoke", figs / "person_c_smoke"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    depths = [2, 4] if args.smoke_test else DEPTHS
    leaves = [1, 5] if args.smoke_test else LEAVES
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    folds, tuning, importance = [], [], []
    for fold, (train, test) in enumerate(outer.split(X, y)):
        if args.smoke_test and fold == 2:
            break
        search = GridSearchCV(
            DecisionTreeClassifier(criterion="gini", random_state=0),
            {"max_depth": depths, "min_samples_leaf": leaves},
            scoring="accuracy", cv=StratifiedKFold(5, shuffle=True, random_state=1),
            n_jobs=args.jobs, return_train_score=True, error_score="raise",
        )
        start = perf_counter()
        search.fit(X.iloc[train], y.iloc[train])
        model = search.best_estimator_
        pred = model.predict(X.iloc[test])
        row = {
            "outer_fold": fold + 1, "repeat": fold // 5 + 1,
            "split": fold % 5 + 1, "n_train": len(train), "n_test": len(test),
            "accuracy": accuracy_score(y.iloc[test], pred),
            "f1": f1_score(y.iloc[test], pred, zero_division=0),
            "roc_auc": roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]),
            "train_accuracy": model.score(X.iloc[train], y.iloc[train]),
            "selected_max_depth": str(search.best_params_["max_depth"]),
            "selected_min_samples_leaf": search.best_params_["min_samples_leaf"],
            "best_inner_accuracy": search.best_score_,
            "fit_seconds": perf_counter() - start,
        }
        folds.append(row)
        importance.append(model.feature_importances_)
        for i, params in enumerate(search.cv_results_["params"]):
            tuning.append({"outer_fold": fold + 1,
                           "max_depth": str(params["max_depth"]),
                           "min_samples_leaf": params["min_samples_leaf"],
                           "inner_train_accuracy": search.cv_results_["mean_train_score"][i],
                           "inner_validation_accuracy": search.cv_results_["mean_test_score"][i]})
        print(f"Fold {fold+1:02d}: accuracy={row['accuracy']:.4f}, "
              f"params={search.best_params_}", flush=True)
    results = pd.DataFrame(folds)
    results.to_csv(out / "decision_tree_folds.csv", index=False)
    summary = {"Model": "DecisionTree", "n_outer_folds": len(results),
               "Acc": results.accuracy.mean(), "Acc_std": results.accuracy.std(ddof=0),
               "F1": results.f1.mean(), "AUC": results.roc_auc.mean(),
               "TrainAcc": results.train_accuracy.mean()}
    pd.DataFrame([summary]).to_csv(out / "results_DecisionTree.csv", index=False)
    results.groupby(["selected_max_depth", "selected_min_samples_leaf"], dropna=False).size().rename(
        "outer_fold_count").reset_index().to_csv(out / "decision_tree_parameter_counts.csv", index=False)
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(out / "decision_tree_inner_grid.csv", index=False)
    # At each depth, choose leaf size using only that outer fold's inner scores.
    selected = tuning.loc[tuning.groupby(["outer_fold", "max_depth"])[
        "inner_validation_accuracy"].idxmax()]
    curve = selected.groupby("max_depth").agg(
        train_mean=("inner_train_accuracy", "mean"),
        validation_mean=("inner_validation_accuracy", "mean"),
        validation_std=("inner_validation_accuracy", lambda s: s.std(ddof=0)),
    ).reindex([str(d) for d in depths])
    curve.to_csv(out / "decision_tree_depth_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(depths))
    ax.plot(positions, curve.train_mean, "o-", label="Inner training")
    ax.errorbar(positions, curve.validation_mean, yerr=curve.validation_std,
                fmt="o-", capsize=3, label="Inner validation (mean ± std)")
    ax.set(xticks=positions, xticklabels=[str(d) if d else "Unlimited" for d in depths],
           xlabel="Maximum tree depth", ylabel="Accuracy",
           title="Decision Tree depth: inner-CV diagnostics")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figs / "person_c_tree_depth.png", dpi=180); plt.close(fig)
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": np.mean(importance, axis=0),
                        "importance_std": np.std(importance, axis=0)}).sort_values("importance_mean")
    imp.to_csv(out / "decision_tree_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.feature, imp.importance_mean, xerr=imp.importance_std, capsize=2)
    ax.set(xlabel="Mean Gini importance ± std across outer models",
           title="Decision Tree feature importance (impurity based)")
    fig.tight_layout(); fig.savefig(figs / "person_c_tree_importance.png", dpi=180); plt.close(fig)
    metadata = {"smoke_test": args.smoke_test, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                "dataset": str(DATA.relative_to(ROOT)), "features": list(X.columns),
                "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                "outer": {"splits": 5, "repeats": 5, "seed": 42},
                "inner": {"splits": 5, "shuffle": True, "seed": 1},
                "grid": {"max_depth": depths, "min_samples_leaf": leaves},
                "criterion": "gini", "tree_random_state": 0, "scoring": "accuracy",
                "std_ddof": 0,
                "notes": ["Precomputed dataset B preprocessing is not refitted within folds.",
                          "Depth curves are inner-CV diagnostics, not unbiased outer estimates.",
                          "Impurity importances can favour continuous/high-cardinality features."]}
    (out / "decision_tree_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
