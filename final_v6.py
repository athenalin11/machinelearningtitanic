"""v6: feature selection (filter / wrapper / embedded, all fitted INSIDE each CV fold, per lecture L11) and
an averaged (majority-vote) model. Writes submission_v6_sel.csv, submission_v6_avg.csv, submission_vote5.csv."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from features_v4 import make4, ens, te, ntr, y, al
from rule_model import predict_rule

RULE_KW = dict(keys=["Key_TS", "Key_SN"], men_rule="men", boy_base=1)
LRC = lambda C=0.1, pen="l2": LogisticRegression(C=C, penalty=pen, solver="liblinear", max_iter=3000)
def sel_lr(selector, C=0.1):
    return Pipeline([("s", StandardScaler()), ("sel", selector), ("m", LRC(C))])
selectors = {
    "none (all feats)":            lambda: sel_lr("passthrough"),
    "filter: mutual info, k=15":   lambda: sel_lr(SelectKBest(mutual_info_classif, k=15)),
    "wrapper: RFE-LR, k=15":       lambda: sel_lr(RFE(LRC(0.1), n_features_to_select=15, step=2)),
    "embedded: L1 (C=0.05)":       lambda: sel_lr(SelectFromModel(LRC(0.05, "l1"))),
}

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
res = {k: [] for k in list(selectors) + ["ensemble(all feats)", "rule", "AVERAGE: majority(LR-sel, ens, rule)"]}
kept = {k: Counter() for k in list(selectors)[1:]}
for a, b in rcv.split(np.zeros(ntr), y):
    A, B = make4(al.index[a], al.index[b])
    P = {}
    for n, f in selectors.items():
        m = f().fit(A, y[a]); P[n] = m.predict(B)
        if n in kept:
            mask = m.named_steps["sel"].get_support()
            kept[n].update(A.columns[mask])
    P["ensemble(all feats)"] = ens().fit(A, y[a]).predict(B)
    P["rule"] = predict_rule(al.index[a], al.index[b], **RULE_KW)
    P["AVERAGE: majority(LR-sel, ens, rule)"] = ((P["wrapper: RFE-LR, k=15"] + P["ensemble(all feats)"] + P["rule"]) >= 2).astype(int)
    for k in res: res[k].append(accuracy_score(y[b], P[k]))
print("CV accuracy (5x5 repeated, identical folds):")
for k, v in res.items(): print(f"  {k:40s} {np.mean(v):.4f} ± {np.std(v):.4f}")
print("\nSelection stability (fraction of 25 folds a feature was kept; top / bottom):")
for n, c in kept.items():
    fr = pd.Series({f: c[f] / 25 for f in A.columns}).sort_values(ascending=False)
    print(f" {n}: always kept ({(fr == 1).sum()}):", list(fr[fr == 1].index)[:12], "| never kept:", list(fr[fr == 0].index)[:10])

# ---------- final fits ----------
A, B = make4(al.index[:ntr], al.index[ntr:])
p_sel = selectors["wrapper: RFE-LR, k=15"]().fit(A, y).predict(B)
p_ens = ens().fit(A, y).predict(B)
p_rule = predict_rule(al.index[:ntr], al.index[ntr:], **RULE_KW)
p_avg = ((p_sel + p_ens + p_rule) >= 2).astype(int)
def save(f, p): pd.DataFrame({"PassengerId": te.PassengerId.values, "Survived": np.asarray(p).astype(int)}).to_csv(f, index=False)
save("submission_v6_sel.csv", p_sel); save("submission_v6_avg.csv", p_avg)
subs = {f: pd.read_csv(f).Survived.values for f in ["submission_v2.csv", "submission_v3b.csv", "submission_v4b.csv", "submission_v5.csv"]}
subs["v6_sel"] = p_sel
vote5 = (np.sum(list(subs.values()), axis=0) >= 3).astype(int); save("submission_vote5.csv", vote5)
for n, p in [("v6_sel", p_sel), ("v6_avg", p_avg), ("vote5", vote5)]:
    print(f"{n}: rate {np.mean(p):.3f} | differs from v2: {(p != subs['submission_v2.csv']).sum()}")
