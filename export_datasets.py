"""Export the feature-engineered datasets for teammates into ./datasets/.
Every train file has a `fold` column (stratified 5-fold, seed 42) so everybody compares models on identical splits."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from features_v4 import B4, make4, al, tr, te, ntr, y

os.makedirs("datasets", exist_ok=True)
pid_tr, pid_te = tr.PassengerId.values, te.PassengerId.values
fold = np.zeros(ntr, int)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y)): fold[v] = k

def save(name, Xtr, Xte):
    a = Xtr.copy(); a.insert(0, "fold", fold); a.insert(0, "Survived", y); a.insert(0, "PassengerId", pid_tr)
    b = Xte.copy(); b.insert(0, "PassengerId", pid_te)
    a.to_csv(f"datasets/{name}_train.csv", index=False); b.to_csv(f"datasets/{name}_test.csv", index=False)
    print(f"{name:22s} train {a.shape} test {b.shape} NaN {int(a.isna().sum().sum() + b.isna().sum().sum())}")

# A: original first-version dataset (label-free)
a_tr, a_te = pd.read_csv("train_processed.csv"), pd.read_csv("test_processed.csv")
save("A_v1_TitleAge_FamilySize", a_tr.drop(columns=["PassengerId", "Survived"]), a_te.drop(columns=["PassengerId"]))

# B: v4 simplified, label-free (+ ticket-size / per-person fare counts, which use no labels)
extra = pd.DataFrame({"TicketSize": al.TicketSize, "LogFarePP": np.log1p(al.FarePP)}, index=al.index)
Bf = pd.concat([B4, extra], axis=1)
save("B_v4_FamilyFlags_AgeTitlePclassParch", Bf.iloc[:ntr].reset_index(drop=True), Bf.iloc[ntr:].reset_index(drop=True))

# C: B + group-survival target-encoded features (train rows: leave-one-out; test rows: all train labels)
Atr, Ate = make4(al.index[:ntr], al.index[ntr:])
Atr = pd.concat([Atr, extra.iloc[:ntr]], axis=1); Ate = pd.concat([Ate, extra.iloc[ntr:]], axis=1)
save("C_v4_B_plus_GroupSurvival_RecomputeInFold", Atr.reset_index(drop=True), Ate.reset_index(drop=True))
print("\nB columns:", list(Bf.columns)); print("\nC extra columns:", [c for c in Atr.columns if c not in Bf.columns])
