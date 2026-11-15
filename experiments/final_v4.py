import sys; sys.path.insert(0, "features")
"""v4: simplified family flags + Title x Pclass x HasParch age imputation (+ fold-safe group survival features).
Compares v2 / v3 / v4 under identical CV folds and writes submission_v4a.csv (LR) and submission_v4b.csv (ensemble)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from final_v3 import al, tr, te, ntr, y, make3, wc_feat, lr, GB, RF
from improve_experiments import group_feat

# ---- v4 label-free base features ----
raw_age = pd.Series(pd.read_csv("data/raw/train.csv").Age.tolist() + pd.read_csv("data/raw/test.csv").Age.tolist(), index=al.index)
has_parch = al.Parch > 0
a = raw_age
age4 = a.fillna(a.groupby([al.Title, al.Pclass, has_parch]).transform("median")) \
        .fillna(a.groupby([al.Title, al.Pclass]).transform("median")).fillna(a.median())
adult = age4 >= 18
B4 = pd.DataFrame({
    "Female": al.Female, "Age": age4, "IsChild": (age4 < 13).astype(int), "AgeMissing": raw_age.isna().astype(int),
    "LogFare": np.log1p(al.Fare), "HasCabin": al.Cabin.notna().astype(int),
    "HasFamily": ((al.SibSp + al.Parch) > 0).astype(int),
    "HasChildren": (has_parch & adult).astype(int),      # adult travelling with Parch>0
    "WithParents": (has_parch & ~adult).astype(int),     # minor travelling with Parch>0
    "LargeFamily": (al.FamilySize >= 5).astype(int),
})
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
CANDS = [  # name, feature builder, model
    ("v1-style features, LR(C=.1) [no group feats]", lambda a, b: make3(a, b, ()) , None),
]
cands = {
    "v2  group feats, ensemble":            (lambda a, b: __import__("final_v3").make3(a, b, ()), ens),
    "v3a +wc feats, LR(C=.03)":             (lambda a, b: make3(a, b, ("wc",)), lambda: lr(0.03)),
    "v3b +wc feats, ensemble":              (lambda a, b: make3(a, b, ("wc",)), ens),
    "v4 simple feats, NO group feats, LR":  (lambda a, b: make4(a, b, False, False), lambda: lr(0.1)),
    "v4 simple + group, LR(C=.03)":         (lambda a, b: make4(a, b, True, False), lambda: lr(0.03)),
    "v4a simple + group + wc, LR(C=.03)":   (lambda a, b: make4(a, b, True, True), lambda: lr(0.03)),
    "v4b simple + group + wc, ensemble":    (lambda a, b: make4(a, b, True, True), ens),
}
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)   # identical folds for every candidate
splits = list(rcv.split(np.zeros(ntr), y)); store = {}
for name, (fb, mf) in cands.items():
    accs, aucs = [], []
    for a_, b_ in splits:
        A, B = fb(al.index[a_], al.index[b_]); m = mf().fit(A, y[a_])
        accs.append(accuracy_score(y[b_], m.predict(B))); aucs.append(roc_auc_score(y[b_], m.predict_proba(B)[:, 1]))
    store[name] = np.array(accs)
    print(f"{name:42s} acc {np.mean(accs):.4f} ± {np.std(accs):.4f}  auc {np.mean(aucs):.4f}", flush=True)
# paired comparison v4a vs v2 (same folds)
d = store["v4a simple + group + wc, LR(C=.03)"] - store["v2  group feats, ensemble"]
print(f"paired diff v4a - v2: {d.mean():+.4f}  (std err {d.std()/np.sqrt(len(d)):.4f})")

# ---- final fit + submissions ----
v2 = pd.read_csv("submissions/final/submission_v2.csv"); v3b = pd.read_csv("submissions/history/submission_v3b.csv")
for f, mf in [("submissions/history/submission_v4a.csv", lambda: lr(0.03)), ("submissions/history/submission_v4b.csv", ens)]:
    A, B = make4(al.index[:ntr], al.index[ntr:])
    p = mf().fit(A, y).predict(B)
    s = pd.DataFrame({"PassengerId": te.PassengerId, "Survived": p.astype(int)}); s.to_csv(f, index=False)
    print(f, s.shape, "rate %.3f" % s.Survived.mean(), "| diff v2:", (s.Survived != v2.Survived).sum(), "| diff v3b:", (s.Survived != v3b.Survived).sum())
