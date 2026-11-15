"""Experiments to improve generalisation. Group-survival features are target-derived, so they are
computed strictly inside each CV training fold (leave-one-out for training rows)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
al = pd.concat([tr, te], ignore_index=True)
ntr = len(tr)

def title(n):
    t = n.str.extract(r",\s*([^\.]+)\.")[0].str.strip().replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return t.where(t.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")

# ---------- label-free features, computed on train+test together (unsupervised counts only) ----------
al["Title"] = title(al.Name)
al["Surname"] = al.Name.str.split(",").str[0]
al["FamilySize"] = al.SibSp + al.Parch + 1
al["TicketSize"] = al.groupby("Ticket").Ticket.transform("count")
al["SurnameSize"] = al.groupby("Surname").Surname.transform("count")
al["GroupSize"] = al[["FamilySize", "TicketSize"]].max(axis=1)
al["FarePP"] = al.Fare.fillna(al.Fare.median()) / al.TicketSize
al["Deck"] = al.Cabin.str[0].fillna("U")
al["Age"] = al.Age.fillna(al.groupby(["Title", "Pclass"]).Age.transform("median")).fillna(al.Age.median())
al["Fare"] = al.Fare.fillna(al.groupby("Pclass").Fare.transform("median"))
al["Embarked"] = al.Embarked.fillna("S")
al["IsChild"] = (al.Age < 13).astype(int)
al["Female"] = (al.Sex == "female").astype(int)
al["WomanChild"] = ((al.Female == 1) | (al.IsChild == 1)).astype(int)

def base_frame(d, extra=()):
    f = pd.DataFrame({"Female": d.Female, "Age": d.Age, "IsChild": d.IsChild,
                      "FamilySize": d.FamilySize, "IsAlone": (d.FamilySize == 1).astype(int),
                      "LogFare": np.log1p(d.Fare), "HasCabin": d.Cabin.notna().astype(int)})
    f = pd.concat([f, pd.get_dummies(d[["Title", "Embarked", "Pclass"]].astype(str), dtype=int)], axis=1)
    if "group" in extra:
        f["TicketSize"] = d.TicketSize; f["LogFarePP"] = np.log1p(d.FarePP)
        f["SmallGroup"] = d.GroupSize.between(2, 4).astype(int)
    if "inter" in extra:
        f["Fem_P3"] = d.Female * (d.Pclass == 3); f["Male_P1"] = (1 - d.Female) * (d.Pclass == 1)
        f["Fem_Fam"] = d.Female * d.FamilySize
    if "deck" in extra:
        f = pd.concat([f, pd.get_dummies(d.Deck, prefix="Deck", dtype=int)], axis=1)
    return f

def group_feat(train_idx, target_idx, key):
    """For each row in target_idx: survival mean/count of OTHER rows in train_idx sharing `key`
    among women/children/men separately is overkill; use the plain mean. Leave-one-out for self."""
    t = al.loc[train_idx, [key, "Survived"]]
    s = t.groupby(key).Survived.agg(["sum", "count"])
    d = al.loc[target_idx, [key]].join(s, on=key)
    sm, ct = d["sum"].fillna(0), d["count"].fillna(0)
    own = al.loc[target_idx].index.isin(train_idx)
    y_self = al.loc[target_idx, "Survived"].where(own, 0).fillna(0)
    sm = sm - y_self; ct = ct - own.astype(int)
    prior = t.Survived.mean()
    return ((sm + prior * 2) / (ct + 2)).values, ct.values   # smoothed rate

def make(train_idx, val_idx, extra):
    Xa = base_frame(al, extra)
    cols = Xa.columns
    A, B = Xa.loc[train_idx].copy(), Xa.loc[val_idx].copy()
    if "grp_target" in extra:
        for key in ["Ticket", "Surname"]:
            A[key + "_rate"], A[key + "_n"] = group_feat(train_idx, train_idx, key)
            B[key + "_rate"], B[key + "_n"] = group_feat(train_idx, val_idx, key)
    return A, B

def cv_eval(model_fn, extra, seed_reps=5):
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=seed_reps, random_state=0)
    idx = np.arange(ntr); y = tr.Survived.values; accs = []
    for a, b in rcv.split(idx, y):
        A, B = make(al.index[a], al.index[b], extra)
        m = model_fn().fit(A, y[a]); accs.append(accuracy_score(y[b], m.predict(B)))
    return np.mean(accs), np.std(accs)

LR = lambda C=0.1: (lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=3000))]))
GB = lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
RF = lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=0)

if __name__ == "__main__":
    # Model-free reference: how well does a sex-only rule do on the CV folds vs Kaggle (0.766)?
    print("gender rule on train:", round(accuracy_score(tr.Survived, al.Female[:ntr]), 4))
    for name, extra in [("v1 base", ()), ("+group(size/fare pp)", ("group",)), ("+interactions", ("inter",)),
                        ("+deck", ("deck",)), ("+group+inter", ("group", "inter")),
                        ("+group-survival target enc", ("grp_target",)),
                        ("+group+grp_target", ("group", "grp_target"))]:
        for mn, mf in [("LR", LR()), ("GB", GB), ("RF", RF)]:
            m, s = cv_eval(mf, extra, 3)
            print(f"{name:30s} {mn}  {m:.4f} ± {s:.4f}", flush=True)
