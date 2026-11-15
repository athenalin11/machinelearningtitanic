import sys; sys.path.insert(0, "features")
import pandas as pd
from rule_model import al, te, ntr, predict_rule

kw = dict(keys=["Key_TS", "Key_SN"], men_rule="men", boy_base=1)
p = predict_rule(al.index[:ntr], al.index[ntr:], **kw)
s = pd.DataFrame({"PassengerId": te.PassengerId.values, "Survived": p.astype(int)})
s.to_csv("submissions/history/submission_v5.csv", index=False)
print(s.shape, "rate %.3f" % s.Survived.mean())
for f in ["submissions/final/submission_v2.csv", "submissions/history/submission_v3b.csv", "submissions/history/submission_v4b.csv", "data/raw/gender_submission.csv"]:
    o = pd.read_csv(f); print(f, "differs:", (o.Survived.values != s.Survived.values).sum())
