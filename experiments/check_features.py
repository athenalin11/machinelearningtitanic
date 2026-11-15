import sys; sys.path.insert(0, "features")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from improve_experiments import al, tr, ntr, base_frame

y = tr.Survived.values
raw = pd.read_csv("data/raw/train.csv")
print("Fare skew raw %.2f | log1p %.2f | sqrt %.2f" % (raw.Fare.skew(), np.log1p(raw.Fare).skew(), np.sqrt(raw.Fare).skew()))
print("Fare quantiles:", raw.Fare.quantile([.5, .9, .99, 1]).round(1).tolist())
print("corr with Survived: raw %.3f | log %.3f" % (raw.Fare.corr(raw.Survived), np.log1p(raw.Fare).corr(raw.Survived)))
print("Fare==0 rows:", (raw.Fare == 0).sum(), "| survivors among them:", raw[raw.Fare == 0].Survived.sum())
print("Age<1:", (raw.Age < 1).sum(), "survived", raw[raw.Age < 1].Survived.mean().round(2),
      "| estimated (x.5):", ((raw.Age % 1 == 0.5)).sum())
nan = raw[(raw.Age < 13) & (raw.Parch == 0)]; print("children with Parch=0 (nanny/other):", len(nan), "survival", nan.Survived.mean().round(2))
print("Pclass vs median fare:", raw.groupby("Pclass").Fare.median().round(1).to_dict())

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
pipe = lambda C=0.1: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=3000))])
d = al.iloc[:ntr]
def score(X): return cross_val_score(pipe(), X.iloc[:ntr], y, cv=cv).mean()

X0 = base_frame(al, ())
print("\nbase", round(score(X0), 4))
# Fare transform
X = X0.copy(); X["LogFare"] = al.Fare; print("raw Fare instead of log", round(score(X), 4))
X = X0.copy(); X["LogFare"] = np.sqrt(al.Fare); print("sqrt Fare", round(score(X), 4))
# Pclass ordinal
X = X0.drop(columns=[c for c in X0 if c.startswith("Pclass_")]); X["Pclass"] = al.Pclass; print("Pclass ordinal (1 col)", round(score(X), 4))
X = X0.copy(); X["Pclass"] = al.Pclass; print("Pclass one-hot + ordinal", round(score(X), 4))
# Variable-Notes inspired
X = X0.copy(); X["AgeEstimated"] = (al.Age.fillna(0) % 1 == 0.5).astype(int); print("+AgeEstimated", round(score(X), 4))
X = X0.copy(); X["Infant"] = (al.Age < 1).astype(int); print("+Infant(<1)", round(score(X), 4))
X = X0.copy(); X["ChildNoParent"] = ((al.Age < 13) & (al.Parch == 0)).astype(int); print("+ChildNoParent(nanny)", round(score(X), 4))
X = X0.copy(); X["Spouse"] = ((al.Title == "Mrs") & (al.SibSp > 0)).astype(int); print("+WifeAboard(Mrs & SibSp>0)", round(score(X), 4))
X = X0.copy(); X["SibSp"] = al.SibSp; X["Parch"] = al.Parch; print("+SibSp,Parch separately", round(score(X), 4))
X = X0.copy(); X["Fam_small"] = al.FamilySize.between(2, 4).astype(int); X["Fam_large"] = (al.FamilySize >= 5).astype(int); print("+Fam small/large bins", round(score(X), 4))
