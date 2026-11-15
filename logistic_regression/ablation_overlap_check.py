import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
tr = pd.read_csv("data/processed/B_v4_FamilyFlags_AgeTitlePclassParch_train.csv")
y = tr.Survived; X = tr.drop(columns=["PassengerId", "Survived", "fold"])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
m = lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.1, solver="liblinear"))])
fam = ["HasFamily", "HasChildren", "WithParents", "LargeFamily"]; tk = ["TicketSize", "LogFarePP"]
for name, drop in [("all", []), ("-Family flags", fam), ("-TicketSize,LogFarePP", tk), ("-Family flags -TicketSize,LogFarePP", fam + tk),
                   ("-Title -Sex", [c for c in X if c.startswith("Title_")] + ["Female"])]:
    r = cross_validate(m(), X.drop(columns=drop), y, cv=cv, scoring=["accuracy", "roc_auc"])
    print(f"{name:38s} acc {r['test_accuracy'].mean():.4f}  auc {r['test_roc_auc'].mean():.4f}")
