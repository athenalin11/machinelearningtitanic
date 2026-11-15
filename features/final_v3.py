"""v3 experiments: women/children-group survival feature + ensemble tuning, selected by repeated CV only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import improve_experiments as ie
from improve_experiments import al, tr, te, ntr, base_frame, group_feat, LR, GB, RF

y = tr.Survived.values
al["Group"] = al.Ticket  # ticket is the most reliable "travelled together" key

def wc_feat(train_idx, target_idx, key, mask_col):
    """Smoothed survival rate of OTHER training members in the same group who satisfy mask_col
    (women/children for the 'women-and-children' pattern; adult men for the opposite)."""
    t = al.loc[train_idx]; t = t[t[mask_col] == 1]
    s = t.groupby(key).Survived.agg(["sum", "count"])
    d = al.loc[target_idx, [key]].join(s, on=key)
    sm, ct = d["sum"].fillna(0).values, d["count"].fillna(0).values
    own = np.isin(al.loc[target_idx].index, t.index)
    ys = np.where(own, al.loc[target_idx, "Survived"].fillna(0).values, 0)
    sm, ct = sm - ys, ct - own.astype(int)
    return (sm + 0.384 * 2) / (ct + 2), ct

al["AdultMale"] = ((al.Female == 0) & (al.IsChild == 0)).astype(int)

def make3(tr_idx, va_idx, extra):
    Xa = base_frame(al, ("group", "inter"))
    A, B = Xa.loc[tr_idx].copy(), Xa.loc[va_idx].copy()
    for key in ["Ticket", "Surname"]:
        A[key + "_rate"], A[key + "_n"] = group_feat(tr_idx, tr_idx, key)
        B[key + "_rate"], B[key + "_n"] = group_feat(tr_idx, va_idx, key)
    if "wc" in extra:
        for key in ["Ticket", "Surname"]:
            for mc in ["WomanChild", "AdultMale"]:
                n = f"{key}_{mc}"
                A[n + "_r"], A[n + "_n"] = wc_feat(tr_idx, tr_idx, key, mc)
                B[n + "_r"], B[n + "_n"] = wc_feat(tr_idx, va_idx, key, mc)
    return A, B

def lr(C): return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=3000))])
def gb(**k): return GradientBoostingClassifier(random_state=0, **k)
CANDS = {
    "v2 ensemble (LR.1,GB,RF)": ((), lambda: VotingClassifier([("lr", lr(0.1)), ("gb", GB()), ("rf", RF())], voting="soft")),
    "v2 feats, LR C=0.03":       ((), lambda: lr(0.03)),
    "+wc feats, ensemble":       (("wc",), lambda: VotingClassifier([("lr", lr(0.1)), ("gb", GB()), ("rf", RF())], voting="soft")),
    "+wc feats, LR C=0.03":      (("wc",), lambda: lr(0.03)),
    "+wc, GB shallow":           (("wc",), lambda: gb(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8)),
    "+wc, ensemble(LR.03,GB,RF,ET)": (("wc",), lambda: VotingClassifier(
        [("lr", lr(0.03)), ("gb", gb(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8)),
         ("rf", RandomForestClassifier(500, min_samples_leaf=3, max_features=0.5, random_state=0)),
         ("et", ExtraTreesClassifier(500, min_samples_leaf=3, random_state=0))], voting="soft")),
}

def run(extra, fn, reps=5):
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42)  # same folds for all candidates
    a_ = []
    for a, b in rcv.split(np.zeros(ntr), y):
        A, B = make3(al.index[a], al.index[b], extra)
        a_.append(accuracy_score(y[b], fn().fit(A, y[a]).predict(B)))
    return np.mean(a_), np.std(a_)

if __name__ == "__main__":
    for n, (ex, fn) in CANDS.items():
        m, s = run(ex, fn); print(f"{n:34s} {m:.4f} ± {s:.4f}", flush=True)
