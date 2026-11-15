"""Gender baseline + group-survival override rules (women/children vs adult men), evaluated with fold-safe CV.
Rule idea: women & children in a group where the OTHER women/children all died -> predict die;
adult men in a group where other men / women-children all survived -> predict survive."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import improve_experiments as ie
from final_v3 import al, tr, te, ntr, y

al["Key_TS"] = al.Ticket                      # ticket group
al["Key_SN"] = al.Surname + "_" + al.Pclass.astype(str)   # surname within class (reduces false matches)
A = ((al.Female == 1) | (al.IsChild == 1)).astype(int)   # women & children
BM = 1 - A                                                # adult men

def mates(train_idx, target_idx, key, sel):
    """(survivors, count) of OTHER training rows in same `key` group with sel==1."""
    t = al.loc[train_idx]; t = t[sel.loc[train_idx] == 1]
    s = t.groupby(key).Survived.agg(["sum", "count"])
    d = al.loc[target_idx, [key]].join(s, on=key)
    sm, ct = d["sum"].fillna(0).values, d["count"].fillna(0).values
    own = np.isin(al.loc[target_idx].index, t.index)
    ys = np.where(own, al.loc[target_idx, "Survived"].fillna(0).values, 0)
    return sm - ys, ct - own.astype(int)

def predict_rule(train_idx, target_idx, keys, boy_base=0, men_rule="men", flip_women=True, flip_men=True):
    d = al.loc[target_idx]
    pred = d.Female.values.copy()
    if boy_base: pred = np.where(d.IsChild.values == 1, 1, pred)
    isA, isB = A.loc[target_idx].values == 1, BM.loc[target_idx].values == 1
    for key in keys:
        sA, cA = mates(train_idx, target_idx, key, A)
        sB, cB = mates(train_idx, target_idx, key, BM)
        if flip_women:   # other women/children in group all died -> predict die
            pred = np.where(isA & (cA >= 1) & (sA == 0), 0, pred)
        if flip_men:
            if men_rule == "men":      # other adult men all survived -> predict survive
                pred = np.where(isB & (cB >= 1) & (sB == cB), 1, pred)
            elif men_rule == "wc":     # all other women/children survived -> predict survive
                pred = np.where(isB & (cA >= 1) & (sA == cA), 1, pred)
            elif men_rule == "both":
                pred = np.where(isB & (cB >= 1) & (sB == cB) & (cA >= 1) & (sA == cA), 1, pred)
    return pred

variants = {
    "gender only": dict(keys=[], flip_women=False, flip_men=False),
    "gender+children(boys survive)": dict(keys=[], flip_women=False, flip_men=False, boy_base=1),
    "women/kids flip  [ticket]": dict(keys=["Key_TS"], flip_men=False),
    "women/kids flip  [surname]": dict(keys=["Key_SN"], flip_men=False),
    "women/kids flip  [ticket+surname]": dict(keys=["Key_TS", "Key_SN"], flip_men=False),
    "flip both, men rule=men [t+s]": dict(keys=["Key_TS", "Key_SN"], men_rule="men"),
    "flip both, men rule=wc [t+s]": dict(keys=["Key_TS", "Key_SN"], men_rule="wc"),
    "flip both, men rule=both [t+s]": dict(keys=["Key_TS", "Key_SN"], men_rule="both"),
    "kids survive + flips(men) [t+s]": dict(keys=["Key_TS", "Key_SN"], men_rule="men", boy_base=1),
}
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
splits = list(rcv.split(np.zeros(ntr), y))
if __name__ == "__main__":
    for n, kw in variants.items():
        accs = [accuracy_score(y[b], predict_rule(al.index[a], al.index[b], **kw)) for a, b in splits]
        print(f"{n:36s} {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)
