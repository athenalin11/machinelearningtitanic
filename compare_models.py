"""Compare several methods on the processed Titanic data with repeated stratified CV.
Outputs: results.csv, results_table.tex, ablation_table.tex, cv_boxplot.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA

tr = pd.read_csv("train_processed.csv")
y = tr.Survived; X = tr.drop(columns=["PassengerId", "Survived"])
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
inner = StratifiedKFold(5, shuffle=True, random_state=1)

def sc(m): return Pipeline([("s", StandardScaler()), ("m", m)])
def tuned(est, grid):  # nested CV: hyper-parameters are tuned inside every outer training fold
    return GridSearchCV(est, grid, cv=inner, scoring="accuracy")

models = {
    "Logistic Regression (L2, C=0.1)": sc(LogisticRegression(C=0.1, max_iter=2000)),
    "Logistic Regression (L1, C=0.3)": sc(LogisticRegression(C=0.3, penalty="l1", solver="liblinear")),
    "LR + PCA(10)": Pipeline([("s", StandardScaler()), ("p", PCA(10)), ("m", LogisticRegression(C=0.1, max_iter=2000))]),
    "Gaussian Naive Bayes": sc(GaussianNB()),
    "kNN (tuned k)": tuned(sc(KNeighborsClassifier()), {"m__n_neighbors": [5, 9, 15, 25]}),
    "SVM linear (tuned C)": tuned(sc(SVC(kernel="linear")), {"m__C": [0.03, 0.1, 0.3, 1]}),
    "SVM RBF (tuned C,gamma)": tuned(sc(SVC()), {"m__C": [0.3, 1, 3, 10], "m__gamma": [0.01, 0.03, 0.1]}),
    "Decision Tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Random Forest": RandomForestClassifier(300, min_samples_leaf=3, random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
}
# Model-free reference points
class Majority(DecisionTreeClassifier): pass
rows, raw = [], {}
for name, m in models.items():
    r = cross_validate(m, X, y, cv=rcv, scoring=["accuracy", "f1", "roc_auc"])
    raw[name] = r["test_accuracy"]
    rows.append({"Model": name, "Acc": r["test_accuracy"].mean(), "Acc_std": r["test_accuracy"].std(),
                 "F1": r["test_f1"].mean(), "AUC": r["test_roc_auc"].mean()})
    print(f"{name:34s} acc {rows[-1]['Acc']:.4f}±{rows[-1]['Acc_std']:.4f} f1 {rows[-1]['F1']:.4f} auc {rows[-1]['AUC']:.4f}", flush=True)
res = pd.DataFrame(rows).sort_values("Acc", ascending=False)
res.to_csv("results.csv", index=False)

# ---- feature-set ablation for the final LR (accuracy + AUC) ----
allc = list(X.columns)
groups = {"Sex": ["Sex"], "Pclass": [c for c in allc if c.startswith("Pclass_")],
          "Title": [c for c in allc if c.startswith("Title_")], "Age": ["Age", "AgeMissing", "IsChild"],
          "Family": ["FamilySize", "IsAlone", "SibSp", "Parch"], "Fare": ["LogFare"],
          "Cabin": ["HasCabin"], "Embarked": [c for c in allc if c.startswith("Embarked_")]}
lr = lambda: sc(LogisticRegression(C=0.1, max_iter=2000))
def ev(cols):
    r = cross_validate(lr(), X[cols], y, cv=rcv, scoring=["accuracy", "roc_auc"])
    return r["test_accuracy"].mean(), r["test_roc_auc"].mean()
ba, bu = ev(allc)
ab = [("All features", ba, bu, 0.0, 0.0)]
for g, cols in groups.items():
    a, u = ev([c for c in allc if c not in cols]); ab.append((f"$-$ {g}", a, u, a - ba, u - bu))
# add-one-in
base_only = ["Sex"]
a, u = ev(base_only); ab.append(("Only Sex", a, u, a - ba, u - bu))
pd.DataFrame(ab, columns=["Setting", "Acc", "AUC", "dAcc", "dAUC"]).to_csv("ablation.csv", index=False)

# ---- LaTeX tables ----
best = res.Acc.max()
with open("results_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Repeated $5\\times5$ stratified CV on the training set (mean $\\pm$ std of accuracy).}\n"
            "\\label{tab:models}\n\\begin{tabular}{lccc}\n\\toprule\nModel & Accuracy & F1 & AUC \\\\\n\\midrule\n")
    for _, r in res.iterrows():
        acc = f"{r.Acc:.4f} $\\pm$ {r.Acc_std:.4f}"
        if r.Acc == best: acc = "\\textbf{" + acc + "}"
        f.write(f"{r.Model.replace('&','\\&')} & {acc} & {r.F1:.4f} & {r.AUC:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
with open("ablation_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Feature-group ablation for logistic regression (L2, $C=0.1$), repeated $5\\times5$ CV.}\n"
            "\\label{tab:ablation}\n\\begin{tabular}{lcccc}\n\\toprule\nSetting & Acc & $\\Delta$Acc & AUC & $\\Delta$AUC \\\\\n\\midrule\n")
    for s, a, u, da, du in ab:
        f.write(f"{s} & {a:.4f} & {da:+.4f} & {u:.4f} & {du:+.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = list(res.Model)[::-1]
plt.figure(figsize=(7, 4.5)); plt.boxplot([raw[k] for k in order], vert=False, labels=order)
plt.xlabel("CV accuracy (25 folds)"); plt.tight_layout(); plt.savefig("cv_boxplot.png", dpi=200)
print(res.round(4).to_string(index=False)); print(pd.DataFrame(ab).round(4).to_string(index=False))
