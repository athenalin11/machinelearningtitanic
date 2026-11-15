"""Main experiments on dataset B (label-free features): logistic regression, regularisation sweep, feature
ablation, confusion matrix, coefficients. Writes LaTeX tables used by report.tex.
Teammate models (kNN, Naive Bayes, Decision Tree, SVM) are evaluated with team_experiment_template.py."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

NAME = "B_v4_FamilyFlags_AgeTitlePclassParch"
tr = pd.read_csv(f"data/processed/{NAME}_train.csv")
y = tr.Survived; X = tr.drop(columns=["PassengerId", "Survived", "fold"]); allc = list(X.columns)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)   # same protocol for the whole team

def lr(C=0.1, pen="l2"): return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, penalty=pen, solver="liblinear", max_iter=3000))])
def ev(model, cols=allc):
    r = cross_validate(model, X[cols], y, cv=rcv, scoring=["accuracy", "f1", "roc_auc"], return_train_score=True)
    return {k: r[k] for k in ["test_accuracy", "test_f1", "test_roc_auc", "train_accuracy"]}

# ---------- 1. model comparison (my models) ----------
models = {
    "Logistic Regression (L2, $C{=}0.1$)": lr(0.1),
    "Logistic Regression (L1, $C{=}0.3$)": lr(0.3, "l1"),
    "Logistic Regression + PCA(10)": Pipeline([("s", StandardScaler()), ("p", PCA(10)), ("m", LogisticRegression(C=0.1, max_iter=3000))]),
    "Random Forest (reference)": RandomForestClassifier(300, min_samples_leaf=3, random_state=0),
    "Gradient Boosting (reference)": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
}
rows = []
for n, m in models.items():
    r = ev(m); rows.append((n, r["test_accuracy"].mean(), r["test_accuracy"].std(), r["test_f1"].mean(), r["test_roc_auc"].mean()))
    print(f"{n:42s} acc {rows[-1][1]:.4f} ± {rows[-1][2]:.4f} f1 {rows[-1][3]:.4f} auc {rows[-1][4]:.4f}", flush=True)
pd.DataFrame(rows, columns=["Model", "Acc", "Acc_std", "F1", "AUC"]).to_csv("logistic_regression/results/results_B.csv", index=False)
with open("report/tables/results_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\\small\n\\caption{Repeated $5\\times5$ stratified CV on dataset B (mean $\\pm$ std of accuracy).}\n\\label{tab:models}\n"
            "\\begin{tabular}{lccc}\n\\toprule\nModel & Accuracy & F1 & AUC \\\\\n\\midrule\n")
    for n, a, s, f1, u in rows: f.write(f"{n} & {a:.4f} $\\pm$ {s:.4f} & {f1:.4f} & {u:.4f} \\\\\n")
    f.write("\\midrule\n")
    for n in ["$k$NN", "Gaussian Naive Bayes", "Decision Tree", "SVM (linear / RBF)"]:
        f.write(f"{n} & \\todo{{fill}} & \\todo{{fill}} & \\todo{{fill}} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------- 2. regularisation sweep (train vs CV) ----------
reg = []
print("\nC sweep (train vs CV)")
for C in [0.01, 0.03, 0.1, 0.3, 1, 10, 100]:
    r = ev(lr(C)); tra, cva = r["train_accuracy"].mean(), r["test_accuracy"].mean(); reg.append((C, tra, cva, tra - cva)); print(f" C={C:<6} train {tra:.4f} cv {cva:.4f} gap {tra-cva:.4f}")
with open("report/tables/reg_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\\small\n\\caption{Training vs.\\ CV accuracy of $L_2$ logistic regression on dataset B (repeated $5\\times5$ CV).}\n\\label{tab:reg}\n"
            "\\begin{tabular}{rccc}\n\\toprule\n$C$ & Train & CV & Gap \\\\\n\\midrule\n")
    for C, a, b, g in reg: f.write(f"{C} & {a:.4f} & {b:.4f} & {g:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("\nL1 sparsity (full-data fit):")
for C in [0.05, 0.1, 0.3, 1]:
    m = lr(C, "l1").fit(X, y); print(f" L1 C={C}: nonzero {(m[-1].coef_ != 0).sum()}/{len(allc)}  cv {ev(lr(C, 'l1'))['test_accuracy'].mean():.4f}")

# ---------- 3. feature-group ablation ----------
g = {"Sex": ["Female"], "Pclass": [c for c in allc if c.startswith("Pclass_")], "Title": [c for c in allc if c.startswith("Title_")],
     "Age": ["Age", "AgeMissing", "IsChild"], "Family flags": ["HasFamily", "HasChildren", "WithParents", "LargeFamily"],
     "Fare": ["LogFare", "LogFarePP"], "TicketSize": ["TicketSize"], "Cabin": ["HasCabin"], "Embarked": [c for c in allc if c.startswith("Embarked_")]}
assert sorted(sum(g.values(), [])) == sorted(allc), set(allc) ^ set(sum(g.values(), []))
base = ev(lr(0.1)); ba, bu = base["test_accuracy"].mean(), base["test_roc_auc"].mean()
ab = [("All features", ba, bu, 0.0, 0.0)]
for k, cols in g.items():
    r = ev(lr(0.1), [c for c in allc if c not in cols]); a, u = r["test_accuracy"].mean(), r["test_roc_auc"].mean(); ab.append((f"$-$ {k}", a, u, a - ba, u - bu))
r = ev(lr(0.1), ["Female"]); ab.append(("Only Sex", r["test_accuracy"].mean(), r["test_roc_auc"].mean(), r["test_accuracy"].mean() - ba, r["test_roc_auc"].mean() - bu))
pd.DataFrame(ab, columns=["Setting", "Acc", "AUC", "dAcc", "dAUC"]).to_csv("logistic_regression/results/ablation_B.csv", index=False)
print("\nAblation:"); [print(f" {s:16s} acc {a:.4f} ({da:+.4f})  auc {u:.4f} ({du:+.4f})") for s, a, u, da, du in ab]
with open("report/tables/ablation_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\\small\n\\caption{Feature-group ablation, logistic regression ($L_2$, $C{=}0.1$) on dataset B, repeated $5\\times5$ CV.}\n\\label{tab:ablation}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\nSetting & Acc & $\\Delta$Acc & AUC & $\\Delta$AUC \\\\\n\\midrule\n")
    for s, a, u, da, du in ab: f.write(f"{s} & {a:.4f} & {da:+.4f} & {u:.4f} & {du:+.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------- 4. error analysis + coefficients ----------
pred = cross_val_predict(lr(0.1), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print("\nConfusion [[TN FP],[FN TP]]:\n", confusion_matrix(y, pred)); print(classification_report(y, pred, digits=3))
m = lr(0.1).fit(X, y); print("Top coefficients (standardised):\n", pd.Series(m[-1].coef_[0], allc).sort_values(key=abs, ascending=False).head(10).round(2))
