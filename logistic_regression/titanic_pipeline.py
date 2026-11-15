import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

tr = pd.read_csv("data/raw/train.csv"); te = pd.read_csv("data/raw/test.csv")

# ---------- EDA ----------
print("shape", tr.shape, "\nmissing:\n", tr.isna().sum()[lambda s: s > 0])
print("survival rate", tr.Survived.mean().round(3))
for c in ["Sex", "Pclass", "Embarked", "SibSp", "Parch"]:
    print(tr.groupby(c).Survived.agg(["mean", "count"]).round(3), "\n")
print(tr.groupby(pd.cut(tr.Age, [0, 12, 18, 35, 60, 100]), observed=True).Survived.mean().round(3))
print(tr.groupby(pd.qcut(tr.Fare, 4), observed=True).Survived.mean().round(3))
print(tr.groupby(tr.Cabin.notna()).Survived.mean().round(3))

# ---------- Feature engineering + preprocessing ----------
def get_title(names):
    t = names.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    t = t.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return t.where(t.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")

def fe(df, ref):
    d = pd.DataFrame(index=df.index)
    title = get_title(df.Name)
    d["Title"] = title
    d["Sex"] = (df.Sex == "female").astype(int)
    d["Pclass"] = df.Pclass
    med = ref.groupby(get_title(ref.Name)).Age.median()  # train-only statistics
    d["Age"] = df.Age.fillna(title.map(med)).fillna(ref.Age.median())
    d["AgeMissing"] = df.Age.isna().astype(int)
    d["IsChild"] = (d.Age < 13).astype(int)
    d["FamilySize"] = df.SibSp + df.Parch + 1
    d["IsAlone"] = (d.FamilySize == 1).astype(int)
    d["SibSp"] = df.SibSp; d["Parch"] = df.Parch
    d["LogFare"] = np.log1p(df.Fare.fillna(ref.Fare.median()))
    d["HasCabin"] = df.Cabin.notna().astype(int)
    d["Embarked"] = df.Embarked.fillna(ref.Embarked.mode()[0])
    return d

cat = ["Title", "Embarked", "Pclass"]
X = pd.get_dummies(fe(tr, tr), columns=cat, dtype=int); y = tr.Survived
Xte = pd.get_dummies(fe(te, tr), columns=cat, dtype=int).reindex(columns=X.columns, fill_value=0)

# ---------- Model + CV ----------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
def score(cols, C=1.0):
    p = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=2000))])
    s = cross_val_score(p, X[cols], y, cv=cv, scoring="accuracy")
    return s.mean(), s.std()

allc = list(X.columns)
print("\nBaseline (Sex,Pclass,Age):", score(["Sex", "Age"] + [c for c in allc if c.startswith("Pclass_")]))
m, s = score(allc); print("All features: %.4f ± %.4f" % (m, s))

# ---------- Feature ablation (by group) ----------
groups = {
    "Sex": ["Sex"], "Pclass": [c for c in allc if c.startswith("Pclass_")],
    "Title": [c for c in allc if c.startswith("Title_")],
    "Age": ["Age", "AgeMissing", "IsChild"],
    "Family": ["FamilySize", "IsAlone", "SibSp", "Parch"],
    "Fare": ["LogFare"], "Cabin": ["HasCabin"],
    "Embarked": [c for c in allc if c.startswith("Embarked_")],
}
print("\nAblation (drop group -> CV acc, delta vs all):")
for g, cols in groups.items():
    mm, _ = score([c for c in allc if c not in cols])
    print(f"  -{g:9s} {mm:.4f}  {mm-m:+.4f}")

from sklearn.model_selection import cross_validate, cross_val_predict, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

def pipe(C=1.0, penalty="l2"):
    return Pipeline([("s", StandardScaler()),
                     ("m", LogisticRegression(C=C, penalty=penalty, solver="liblinear", max_iter=2000))])

# L3: overfitting / Structural Risk Minimization -> train vs validation gap as regularisation (1/C) varies
print("\nC tuning (L2 = Ridge-style SRM), train vs CV:")
for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]:
    r = cross_validate(pipe(C), X, y, cv=cv, return_train_score=True)
    print(" C=%-5s train %.4f  cv %.4f  gap %.4f" % (C, r["train_score"].mean(), r["test_score"].mean(),
                                                    r["train_score"].mean() - r["test_score"].mean()))
# L3: Occam's razor -> L1 gives a sparser (simpler) model
for C in [0.1, 0.3, 1]:
    pl = pipe(C, "l1").fit(X, y)
    print(" L1 C=%-4s cv %.4f  nonzero coef %d/%d" % (C, cross_val_score(pipe(C, "l1"), X, y, cv=cv).mean(),
                                                       (pl[-1].coef_ != 0).sum(), X.shape[1]))

# L8: metrics beyond accuracy (repeated CV for stability)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
r = cross_validate(pipe(0.1), X, y, cv=rcv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print("\nRepeated 5x5 CV (C=0.1):", {k[5:]: round(v.mean(), 4) for k, v in r.items() if k.startswith("test_")})
pred = cross_val_predict(pipe(0.1), X, y, cv=cv)
print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y, pred))
print(classification_report(y, pred, digits=3))
# ablation re-measured with AUC (threshold-free)
print("Ablation by AUC:")
base_auc = cross_val_score(pipe(0.1), X, y, cv=rcv, scoring="roc_auc").mean()
for g, cols in groups.items():
    a = cross_val_score(pipe(0.1), X[[c for c in allc if c not in cols]], y, cv=rcv, scoring="roc_auc").mean()
    print(f"  -{g:9s} {a:.4f}  {a-base_auc:+.4f}")

p = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(X, y)
print("\nTop coefficients:\n", pd.Series(p[-1].coef_[0], allc).sort_values(key=abs, ascending=False).head(10).round(3))
pd.DataFrame({"PassengerId": te.PassengerId, "Survived": p.predict(Xte)}).to_csv("submissions/history/submission_v1.csv", index=False)

# ---------- Export processed data (unscaled, one-hot encoded) ----------
out_tr = X.copy(); out_tr.insert(0, "Survived", y.values); out_tr.insert(0, "PassengerId", tr.PassengerId.values)
out_te = Xte.copy(); out_te.insert(0, "PassengerId", te.PassengerId.values)
out_tr.to_csv("data/interim/train_processed.csv", index=False); out_te.to_csv("data/interim/test_processed.csv", index=False)
print("\nprocessed", out_tr.shape, out_te.shape, "NaN:", out_tr.isna().sum().sum(), out_te.isna().sum().sum())
