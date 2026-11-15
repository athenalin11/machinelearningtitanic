"""v4 feature builder (importable without running any experiment)."""
import numpy as np, pandas as pd
from sklearn.ensemble import VotingClassifier
from final_v3 import al, tr, te, ntr, y, wc_feat, lr, GB, RF
from improve_experiments import group_feat

raw_age = pd.Series(pd.read_csv("train.csv").Age.tolist() + pd.read_csv("test.csv").Age.tolist(), index=al.index)
has_parch = al.Parch > 0
age4 = (raw_age.fillna(raw_age.groupby([al.Title, al.Pclass, has_parch]).transform("median"))
        .fillna(raw_age.groupby([al.Title, al.Pclass]).transform("median")).fillna(raw_age.median()))
adult = age4 >= 18
B4 = pd.DataFrame({
    "Female": al.Female, "Age": age4, "IsChild": (age4 < 13).astype(int), "AgeMissing": raw_age.isna().astype(int),
    "LogFare": np.log1p(al.Fare), "HasCabin": al.Cabin.notna().astype(int),
    "HasFamily": ((al.SibSp + al.Parch) > 0).astype(int),
    "HasChildren": (has_parch & adult).astype(int), "WithParents": (has_parch & ~adult).astype(int),
    "LargeFamily": (al.FamilySize >= 5).astype(int)})
B4 = pd.concat([B4, pd.get_dummies(al[["Title", "Embarked", "Pclass"]].astype(str), dtype=int)], axis=1)

def make4(tr_idx, va_idx, group=True, wc=True):
    A, B = B4.loc[tr_idx].copy(), B4.loc[va_idx].copy()
    if group:
        for key in ["Ticket", "Surname"]:
            A[key + "_rate"], A[key + "_n"] = group_feat(tr_idx, tr_idx, key)
            B[key + "_rate"], B[key + "_n"] = group_feat(tr_idx, va_idx, key)
    if wc:
        for key in ["Ticket", "Surname"]:
            for mc in ["WomanChild", "AdultMale"]:
                n = f"{key}_{mc}"
                A[n + "_r"], A[n + "_n"] = wc_feat(tr_idx, tr_idx, key, mc)
                B[n + "_r"], B[n + "_n"] = wc_feat(tr_idx, va_idx, key, mc)
    return A, B

ens = lambda: VotingClassifier([("lr", lr(0.1)), ("gb", GB()), ("rf", RF())], voting="soft")
