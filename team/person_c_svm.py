"""Phase 2: scaled Linear/RBF SVC on dataset B, with nested 5x5 outer CV.

Default: full experiment. --smoke-test: two outer folds and reduced grids,
saved separately. Hyperparameter figures use inner CV only. Existing Decision
Tree files and shared reports are never written. Run from any directory.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/B_v4_FamilyFlags_AgeTitlePclassParch_train.csv"
LINEAR_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]
RBF_C = [0.03, 0.1, 0.3, 1, 3, 10, 100]
GAMMA = ["scale", 0.001, 0.01, 0.03, 0.1]


def plot_diagnostics(grid, summary, out, figs):
    linear = grid[grid.model == "linear"].groupby("C").agg(
        train_mean=("inner_train_accuracy", "mean"),
        validation_mean=("inner_validation_accuracy", "mean"),
        validation_std=("inner_validation_accuracy", lambda s: s.std(ddof=0)))
    linear.to_csv(out / "svm_linear_c_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(linear.index, linear.train_mean, "o-", label="Inner training")
    ax.errorbar(linear.index, linear.validation_mean, yerr=linear.validation_std,
                fmt="o-", capsize=3, label="Inner validation (mean ± std)")
    ax.set(xscale="log", xlabel="C (log scale)", ylabel="Accuracy",
           title="Linear SVM: inner-CV C experiment")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figs / "svm_linear_c.png", dpi=180); plt.close(fig)

    rbf = grid[grid.model == "rbf"]
    heat = rbf.pivot_table(index="C", columns="gamma", values="inner_validation_accuracy")
    gamma_order = [str(g) for g in GAMMA if str(g) in heat.columns]
    heat = heat.reindex(columns=gamma_order).sort_index()
    heat.to_csv(out / "svm_rbf_c_gamma_heatmap.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns,
           yticks=range(len(heat.index)), yticklabels=heat.index,
           xlabel="Gamma ('scale' is fitted separately per training fold)", ylabel="C",
           title="RBF SVM: mean inner-CV validation accuracy")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center",
                    color="white" if im.norm(heat.iloc[i, j]) < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Accuracy"); fig.tight_layout()
    fig.savefig(figs / "svm_rbf_c_gamma_heatmap.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for ax, (metric, label) in zip(axes, [("accuracy", "Accuracy"), ("f1", "F1"),
                                       ("roc_auc", "ROC-AUC"), ("train_accuracy", "Training accuracy")]):
        ax.bar(summary.model, summary[f"{metric}_mean"],
               yerr=summary[f"{metric}_std"], capsize=4, color=["#3575a5", "#de8f30"])
        ax.set(title=label, ylim=(0, 1.05))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean ± std over outer folds")
    fig.suptitle("Linear vs RBF SVM: nested CV")
    fig.tight_layout(); fig.savefig(figs / "svm_kernel_comparison.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    frame = pd.read_csv(DATA)
    X, y = frame.drop(columns=["PassengerId", "Survived", "fold"]), frame.Survived
    if X.shape != (891, 23) or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Expected dataset B: 891 rows, 23 finite numeric features")
    if set(y.unique()) != {0, 1} or not frame.PassengerId.is_unique:
        raise ValueError("Invalid labels or duplicate passenger IDs")
    out, figs = ROOT / "team/results/person_c", ROOT / "report/figures"
    if args.smoke_test:
        out, figs = out / "svm_smoke", figs / "svm_smoke"
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    grids = {
        "linear": {"svc__C": [0.1, 1] if args.smoke_test else LINEAR_C},
        "rbf": {"svc__C": [0.1, 1] if args.smoke_test else RBF_C,
                "svc__gamma": ["scale", 0.01] if args.smoke_test else GAMMA},
    }
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X, y))
    if args.smoke_test:
        splits = splits[:2]
    rows, tuning = [], []
    for kernel, params in grids.items():
        for fold, (train, test) in enumerate(splits, 1):
            search = GridSearchCV(
                Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel=kernel, shrinking=False))]),
                params, scoring="accuracy",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                n_jobs=args.jobs, return_train_score=True, error_score="raise")
            start = perf_counter()
            search.fit(X.iloc[train], y.iloc[train])
            model = search.best_estimator_
            if model["svc"].fit_status_ != 0:
                raise RuntimeError("SVC did not converge")
            pred = model.predict(X.iloc[test])
            row = {"model": kernel, "outer_fold": fold, "repeat": (fold-1)//5+1,
                   "split": (fold-1)%5+1, "n_train": len(train), "n_test": len(test),
                   "accuracy": accuracy_score(y.iloc[test], pred),
                   "f1": f1_score(y.iloc[test], pred, zero_division=0),
                   "roc_auc": roc_auc_score(y.iloc[test], model.decision_function(X.iloc[test])),
                   "train_accuracy": model.score(X.iloc[train], y.iloc[train]),
                   "selected_C": search.best_params_["svc__C"],
                   "selected_gamma": str(search.best_params_.get("svc__gamma", "not_applicable")),
                   "best_inner_accuracy": search.best_score_,
                   "n_support_vectors": int(model["svc"].n_support_.sum()),
                   "fit_seconds": perf_counter()-start}
            rows.append(row)
            for i, p in enumerate(search.cv_results_["params"]):
                tuning.append({"model": kernel, "outer_fold": fold, "C": p["svc__C"],
                               "gamma": str(p.get("svc__gamma", "not_applicable")),
                               "inner_train_accuracy": search.cv_results_["mean_train_score"][i],
                               "inner_validation_accuracy": search.cv_results_["mean_test_score"][i]})
            print(f"{kernel} fold {fold:02d}: acc={row['accuracy']:.4f}, "
                  f"params={search.best_params_}", flush=True)
    results, grid = pd.DataFrame(rows), pd.DataFrame(tuning)
    results.to_csv(out / "svm_folds.csv", index=False)
    grid.to_csv(out / "svm_inner_grid.csv", index=False)
    summaries = []
    for kernel, group in results.groupby("model", sort=False):
        group.to_csv(out / f"svm_{kernel}_folds.csv", index=False)
        summary = {"model": kernel, "n_outer_folds": len(group)}
        for metric in ["accuracy", "f1", "roc_auc", "train_accuracy"]:
            summary[f"{metric}_mean"] = group[metric].mean()
            summary[f"{metric}_std"] = group[metric].std(ddof=0)
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "svm_summary.csv", index=False)
    results.groupby(["model", "selected_C", "selected_gamma"]).size().rename(
        "outer_fold_count").reset_index().to_csv(out / "svm_parameter_counts.csv", index=False)
    plot_diagnostics(grid, summary, out, figs)
    metadata = {"smoke_test": args.smoke_test, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                "dataset": str(DATA.relative_to(ROOT)),
                "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                "features": list(X.columns), "outer": {"splits": 5, "repeats": 5, "seed": 42},
                "actual_outer_folds_per_model": len(splits),
                "inner": {"splits": 5, "shuffle": True, "seed": 1},
                "grids": grids, "scoring": "accuracy", "auc_source": "decision_function",
                "pipeline": ["StandardScaler", "SVC"], "SVC_probability": False,
                "SVC_shrinking": False, "SVC_tol": 0.001, "jobs": args.jobs,
                "std_ddof": 0,
                "notes": ["Existing B preprocessing is not refitted within folds.",
                          "Scaler is fitted inside each inner/outer training fold.",
                          "C curve and heatmap are inner-CV diagnostics, not outer performance estimates.",
                          "Selected parameter frequencies are reported; no single full-data model is fitted."]}
    (out / "svm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
