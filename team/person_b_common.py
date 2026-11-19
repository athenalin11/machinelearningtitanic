"""Shared evaluation/output utilities for Person B model-specific scripts.

Run from any directory. Only writes Person B files. --smoke-test writes to
separate smoke directories; the default uses all 25 common outer folds.
No model selection uses outer test scores or Kaggle scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'titanic-person-b-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/B_v4_FamilyFlags_AgeTitlePclassParch_train.csv'


def diagnostics(grid, summary, out, figs):
    # Keep distance and voting fixed for this controlled K/scaling diagnostic.
    # All points are INNER scores; they are not additional held-out estimates.
    if grid.model.str.startswith('knn').any():
        g = grid[grid.model.str.startswith('knn') & (grid.weights == 'uniform')]
        curves = g.groupby(['model', 'k'], as_index=False).agg(
            train_mean=('inner_train_accuracy', 'mean'),
            validation_mean=('inner_validation_accuracy', 'mean'),
            validation_std=('inner_validation_accuracy', lambda x: x.std(ddof=0)))
        curves.to_csv(out / 'knn_k_scaling_curves.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
        for ax, (name, part) in zip(axes, curves.groupby('model', sort=True)):
            ax.plot(part.k, part.train_mean, 'o-', label='Inner training')
            ax.errorbar(part.k, part.validation_mean, yerr=part.validation_std,
                        fmt='o-', capsize=3, label='Inner validation mean +/- std')
            ax.set(title=name, xlabel='Number of neighbours K', ylabel='Accuracy', ylim=(0.5, 1.02))
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle('Euclidean distance, uniform voting; inner-CV diagnostics')
        fig.tight_layout(); fig.savefig(figs / 'knn_k_scaling.png', dpi=180); plt.close(fig)
    if (grid.model == 'gaussian_nb').any():
        nb = grid[grid.model == 'gaussian_nb'].groupby('var_smoothing', as_index=False).agg(
            train_mean=('inner_train_accuracy', 'mean'), validation_mean=('inner_validation_accuracy', 'mean'),
            validation_std=('inner_validation_accuracy', lambda x: x.std(ddof=0)))
        nb.to_csv(out / 'nb_smoothing_curve.csv', index=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(nb.var_smoothing, nb.train_mean, 'o-', label='Inner training')
        ax.errorbar(nb.var_smoothing, nb.validation_mean, yerr=nb.validation_std, fmt='o-', capsize=3, label='Inner validation')
        ax.set(xscale='log', xlabel='var_smoothing', ylabel='Accuracy', title='GaussianNB: inner-CV smoothing experiment')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(figs / 'nb_smoothing.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ['accuracy', 'f1', 'roc_auc']):
        ax.bar(summary.model, summary[metric + '_mean'], yerr=summary[metric + '_std'], capsize=3)
        ax.set(title=metric, ylim=(0, 1.05)); ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Nested CV: outer-fold means +/- standard deviations')
    fig.tight_layout(); fig.savefig(figs / 'model_comparison.png', dpi=180); plt.close(fig)


def run_experiment(config_factory, family):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    frame = pd.read_csv(DATA)
    X = frame.drop(columns=['PassengerId', 'Survived', 'fold']); y = frame.Survived
    if X.shape != (891, 23) or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Expected dataset B with 891 rows and 23 finite numeric features')
    if not frame.PassengerId.is_unique or set(y.unique()) != {0, 1}:
        raise ValueError('Invalid passenger IDs or target')
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X, y))
    first_fold = np.empty(len(y), dtype=int)
    for k, (_, valid) in enumerate(outer[:5]): first_fold[valid] = k
    if not np.array_equal(first_fold, frame.fold.to_numpy()):
        raise ValueError('Stored fold column differs from common first-repeat split')
    if args.smoke_test: outer = outer[:2]
    out = ROOT / 'team/results/person_b' / family
    figs = ROOT / 'report/figures/person_b' / family
    if args.smoke_test: out, figs = out / 'smoke', figs / 'smoke'
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    configs = config_factory(args.smoke_test)
    folds, tuning, predictions, parameters = [], [], [], []
    # Limit native threads for reproducible, low-overhead small-data fits.
    with threadpool_limits(limits=1):
        for name, (estimator, params) in configs.items():
            for fold, (train, valid) in enumerate(outer, 1):
                start = perf_counter()
                search = GridSearchCV(estimator, params, scoring='accuracy',
                    cv=StratifiedKFold(5, shuffle=True, random_state=1), n_jobs=1,
                    return_train_score=True, error_score='raise')
                search.fit(np.ascontiguousarray(X.iloc[train], dtype=float), y.iloc[train])
                model = search.best_estimator_
                proba = model.predict_proba(np.ascontiguousarray(X.iloc[valid], dtype=float))[:, 1]
                pred = model.predict(np.ascontiguousarray(X.iloc[valid], dtype=float))
                if not np.isfinite(proba).all(): raise ValueError('Non-finite probabilities')
                tn, fp, fn, tp = confusion_matrix(y.iloc[valid], pred, labels=[0, 1]).ravel()
                folds.append(dict(model=name, outer_fold=fold, repeat=(fold-1)//5+1, split=(fold-1)%5+1,
                    n_train=len(train), n_test=len(valid), accuracy=accuracy_score(y.iloc[valid], pred),
                    f1=f1_score(y.iloc[valid], pred, zero_division=0), roc_auc=roc_auc_score(y.iloc[valid], proba),
                    train_accuracy=model.score(np.ascontiguousarray(X.iloc[train], dtype=float), y.iloc[train]),
                    best_inner_accuracy=search.best_score_, selected_params=json.dumps(search.best_params_, sort_keys=True),
                    TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), fit_seconds=perf_counter()-start))
                parameters.append(dict(model=name, outer_fold=fold, **search.best_params_))
                for i, p in enumerate(search.cv_results_['params']):
                    tuning.append(dict(model=name, outer_fold=fold, k=p.get('clf__n_neighbors', np.nan),
                        weights=p.get('clf__weights', ''), var_smoothing=p.get('var_smoothing', np.nan),
                        inner_train_accuracy=search.cv_results_['mean_train_score'][i],
                        inner_validation_accuracy=search.cv_results_['mean_test_score'][i]))
                predictions.extend(dict(model=name, outer_fold=fold, repeat=(fold-1)//5+1,
                    PassengerId=int(frame.PassengerId.iloc[j]), target=int(y.iloc[j]), prediction=int(pr), probability=float(pb))
                    for j, pr, pb in zip(valid, pred, proba))
                print(f'{name} fold {fold:02d}: accuracy={folds[-1]["accuracy"]:.4f}, {search.best_params_}', flush=True)
    results, grid, preds = pd.DataFrame(folds), pd.DataFrame(tuning), pd.DataFrame(predictions)
    expected = len(outer) * len(configs)
    if len(results) != expected or len(preds) != sum(len(v) for _, v in outer)*len(configs):
        raise RuntimeError('Incomplete outer-fold outputs')
    if not args.smoke_test:
        counts = preds.groupby(['model', 'PassengerId']).size()
        if not (counts == 5).all(): raise RuntimeError('Expected one prediction per passenger per repeat')
    summaries = []
    for name, part in results.groupby('model', sort=False):
        row = dict(model=name, n_outer_folds=len(part))
        for metric in ['accuracy', 'f1', 'roc_auc', 'train_accuracy']:
            row[metric + '_mean'] = part[metric].mean(); row[metric + '_std'] = part[metric].std(ddof=0)
        row['train_validation_gap'] = row['train_accuracy_mean'] - row['accuracy_mean']
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    results.to_csv(out / 'folds.csv', index=False); grid.to_csv(out / 'inner_grid.csv', index=False)
    preds.to_csv(out / 'outer_predictions.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(parameters).to_csv(out / 'selected_parameters.csv', index=False)
    results.groupby(['model', 'selected_params']).size().rename('outer_fold_count').reset_index().to_csv(out / 'parameter_counts.csv', index=False)
    paired = None
    if family == 'knn':
        a = results[results.model == 'knn_scaled'].set_index('outer_fold')
        b = results[results.model == 'knn_unscaled'].set_index('outer_fold')
        paired = pd.DataFrame({m + '_scaled_minus_unscaled': a[m]-b[m] for m in ['accuracy', 'f1', 'roc_auc']})
        paired.to_csv(out / 'scaling_paired_differences.csv')
    diagnostics(grid, summary, out, figs)
    labels = {'knn_scaled': r'$k$NN (scaled, nested CV)', 'gaussian_nb': 'Gaussian Naive Bayes (nested CV)'}
    latex = ['% Person B rows only; insert manually into the shared table after review.']
    for _, row in summary.iterrows():
        if row.model in labels:
            latex.append(f'{labels[row.model]} & {row.accuracy_mean:.4f} $\\pm$ {row.accuracy_std:.4f} & {row.f1_mean:.4f} & {row.roc_auc_mean:.4f} ' + r'\\')
    (out / 'report_rows.tex').write_text('\n'.join(latex)+'\n')
    metadata = dict(model_family=family, smoke_test=args.smoke_test, python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__, matplotlib=matplotlib.__version__,
        dataset=str(DATA.relative_to(ROOT)), dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        features=list(X.columns), outer=dict(splits=5, repeats=5, seed=42), actual_outer_folds=len(outer),
        inner=dict(splits=5, shuffle=True, seed=1), grids={n:p for n,(_,p) in configs.items()},
        scoring='accuracy', std_ddof=0, knn_distance='euclidean', knn_algorithm='brute', threads=1,
        notes=['B preprocessing is precomputed on train/test covariates, not refitted in folds.',
               'StandardScaler is fitted only within inner/outer training folds.',
               'Scaled/unscaled KNN are tuned separately on identical outer/inner splits.',
               'K curves hold Euclidean distance and uniform weights fixed; use inner scores only.',
               'GaussianNB receives unscaled features; smoothing is relative to largest feature variance.',
               'Repeated-fold standard deviation is not a confidence interval; no significance test.',
               'No final full-data model or Kaggle submission is generated.',
               'Runtime differs from Person C sklearn 1.6.1; compare descriptively.'])
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# Person B experiment results', '', f'Smoke test: {args.smoke_test}. Dataset B, {len(outer)} outer folds.','',
             '| Model | Accuracy mean +/- std | F1 | AUC | Train accuracy |', '|---|---:|---:|---:|---:|']
    for _, r in summary.iterrows():
        lines.append(f'| {r.model} | {r.accuracy_mean:.4f} +/- {r.accuracy_std:.4f} | {r.f1_mean:.4f} | {r.roc_auc_mean:.4f} | {r.train_accuracy_mean:.4f} |')
    if paired is not None:
        lines += ['', f'Scaled minus unscaled kNN mean accuracy difference: {paired.iloc[:,0].mean():+.4f}.',
            'This compares independently inner-tuned procedures on paired outer folds, not a fixed-hyperparameter effect.',
            'See the uniform-vote K curves for a controlled comparison at identical K and distance.']
    lines += ['', 'Selected parameter frequencies:', '',
              results.groupby(['model','selected_params']).size().to_string(), '',
              'Limitations:', *['- '+n for n in metadata['notes']]]
    (out / 'analysis.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False), flush=True)
