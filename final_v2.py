"""v2: engineered features + group-survival target encoding (fold-safe) + soft-voting ensemble."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from improve_experiments import al, tr, te, ntr, make, LR, GB, RF

EXTRA = ("group", "inter", "grp_target")
ens = lambda: VotingClassifier([("lr", LR()()), ("gb", GB()), ("rf", RF())], voting="soft")

# --- cross-validation (group features recomputed inside every training fold) ---
y = tr.Survived.values
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
accs = []
for a, b in rcv.split(np.zeros(ntr), y):
    A, B = make(al.index[a], al.index[b], EXTRA)
    accs.append(accuracy_score(y[b], ens().fit(A, y[a]).predict(B)))
print(f"Ensemble CV acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# --- final fit on all training rows, predict test ---
A, B = make(al.index[:ntr], al.index[ntr:], EXTRA)
pred = ens().fit(A, y).predict(B)
sub = pd.DataFrame({"PassengerId": te.PassengerId, "Survived": pred.astype(int)})
sub.to_csv("submission_v2.csv", index=False)
old = pd.read_csv("submission.csv"); g = pd.read_csv("gender_submission.csv")
print("v2 survive rate", sub.Survived.mean().round(3), "| differs from v1:", (sub.Survived != old.Survived).sum(),
      "| differs from gender baseline:", (sub.Survived != g.Survived).sum())
