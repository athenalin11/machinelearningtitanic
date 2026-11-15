import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import improve_experiments as ie
from improve_experiments import al, tr, ntr

y = tr.Survived.values
al["AgeRaw"] = pd.read_csv("train.csv").Age.tolist() + pd.read_csv("test.csv").Age.tolist()
known = al.AgeRaw.notna()

# ---------- 1. Can we recover families? ----------
miss = al[~known]
mates_t = al.groupby("Ticket").AgeRaw.transform(lambda s: s.notna().sum())
mates_s = al.groupby("Surname").AgeRaw.transform(lambda s: s.notna().sum())
print("missing age rows:", len(miss))
print(" with >=1 OTHER known-age ticket mate :", int(((mates_t - 0)[~known] >= 1).sum()))
print(" with >=1 OTHER known-age same-surname :", int(((mates_s - 0)[~known] >= 1).sum()))
print(" travelling alone (no SibSp/Parch)    :", int(((al.SibSp + al.Parch)[~known] == 0).sum()))
print("missing age by Pclass:", miss.Pclass.value_counts().to_dict(), "| by Title:", miss.Title.value_counts().to_dict())

# ---------- 2. Impute-quality test: hide known ages, predict, measure MAE ----------
def feats(d, mask_known):
    a = d.AgeRaw.where(mask_known)
    f = pd.DataFrame({"Pclass": d.Pclass, "SibSp": d.SibSp, "Parch": d.Parch, "Female": d.Female,
                      "LogFare": np.log1p(d.Fare), "FamilySize": d.FamilySize})
    f = pd.concat([f, pd.get_dummies(d.Title, dtype=int)], axis=1)
    # family-informed features: mean/min/max age of OTHER members with a known age
    for key in ["Ticket", "Surname"]:
        g = pd.DataFrame({"k": d[key], "a": a})
        s, c = g.groupby("k").a.transform("sum"), g.groupby("k").a.transform("count")
        own = a.fillna(0); oc = a.notna().astype(int)
        f[key + "_meanAge"] = ((s - own) / (c - oc)).replace([np.inf, -np.inf], np.nan)
    return f

def impute_title_pclass(d, mask_known):
    a = d.AgeRaw.where(mask_known)
    return a.fillna(a.groupby([d.Title, d.Pclass]).transform("median")).fillna(a.median())
def impute_title_pclass_parch(d, mask_known):
    a = d.AgeRaw.where(mask_known); k = [d.Title, d.Pclass, (d.Parch > 0)]
    return a.fillna(a.groupby(k).transform("median")).fillna(a.groupby([d.Title, d.Pclass]).transform("median")).fillna(a.median())
def impute_gbr(d, mask_known, train_rows):
    F = feats(d, mask_known); F = F.fillna(-1)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    m.fit(F.loc[train_rows], d.AgeRaw.loc[train_rows])
    return m.predict(F)

rng = np.random.RandomState(0); idx = al.index[known]
res = {"Title x Pclass median": [], "Title x Pclass x HasParch median": [], "GBR + ticket/surname mate ages": []}
for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(idx):
    hide = idx[te_i]; mk = known.copy(); mk[hide] = False       # pretend these ages are missing
    truth = al.AgeRaw[hide]
    res["Title x Pclass median"].append(np.abs(impute_title_pclass(al, mk)[hide] - truth).mean())
    res["Title x Pclass x HasParch median"].append(np.abs(impute_title_pclass_parch(al, mk)[hide] - truth).mean())
    res["GBR + ticket/surname mate ages"].append(np.abs(pd.Series(impute_gbr(al, mk, idx[tr_i]), index=al.index)[hide] - truth).mean())
print("\nAge imputation MAE (years, lower is better); global-median MAE = %.2f" % np.abs(al.AgeRaw[known] - al.AgeRaw.median()).mean())
for k, v in res.items(): print(f"  {k:36s} {np.mean(v):.2f}")

# ---------- 3. Simplified family features vs current ----------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.1, max_iter=3000))])
def sc(X): return cross_val_score(pipe, X.iloc[:ntr], y, cv=cv).mean()
age_t = impute_title_pclass(al, known); age_g = pd.Series(impute_gbr(al, known, al.index[known]), index=al.index).where(~known, None)
age_gbr = al.AgeRaw.fillna(pd.Series(impute_gbr(al, known, al.index[known]), index=al.index))

def build(age, fam="current"):
    d = al.copy(); d["Age"] = age; d["IsChild"] = (age < 13).astype(int)
    f = pd.DataFrame({"Female": d.Female, "Age": d.Age, "IsChild": d.IsChild, "LogFare": np.log1p(d.Fare),
                      "HasCabin": d.Cabin.notna().astype(int)})
    f = pd.concat([f, pd.get_dummies(d[["Title", "Embarked", "Pclass"]].astype(str), dtype=int)], axis=1)
    f["AgeMissing"] = (~known).astype(int)
    if fam == "current":
        f["FamilySize"] = d.FamilySize; f["IsAlone"] = (d.FamilySize == 1).astype(int)
    elif fam == "simple":   # user's idea: has family? has children?
        f["HasFamily"] = ((d.SibSp + d.Parch) > 0).astype(int)
        f["HasChildren"] = ((d.Parch > 0) & (d.Age >= 18)).astype(int)      # adult travelling with Parch>0
        f["WithParents"] = ((d.Parch > 0) & (d.Age < 18)).astype(int)       # minor travelling with Parch>0
    elif fam == "simple+spouse":
        f["HasFamily"] = ((d.SibSp + d.Parch) > 0).astype(int)
        f["HasChildren"] = ((d.Parch > 0) & (d.Age >= 18)).astype(int)
        f["WithParents"] = ((d.Parch > 0) & (d.Age < 18)).astype(int)
        f["HasSibSp"] = (d.SibSp > 0).astype(int)
    elif fam == "simple+size":
        f["HasFamily"] = ((d.SibSp + d.Parch) > 0).astype(int)
        f["HasChildren"] = ((d.Parch > 0) & (d.Age >= 18)).astype(int)
        f["WithParents"] = ((d.Parch > 0) & (d.Age < 18)).astype(int)
        f["LargeFamily"] = (d.FamilySize >= 5).astype(int)
    return f
print("\nLR CV accuracy (10x5 repeated):")
for an, age in [("Title x Pclass median", age_t), ("GBR w/ family ages", age_gbr)]:
    for fam in ["current", "simple", "simple+spouse", "simple+size"]:
        print(f"  age={an:24s} family={fam:14s} {sc(build(age, fam)):.4f}", flush=True)
