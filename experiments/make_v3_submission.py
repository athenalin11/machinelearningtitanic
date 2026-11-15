import sys; sys.path.insert(0, "features")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import VotingClassifier
from final_v3 import al, tr, te, ntr, y, make3, lr, GB, RF

A, B = make3(al.index[:ntr], al.index[ntr:], ("wc",))
models = {
    "submissions/history/submission_v3a.csv": lr(0.03),   # simplest model, best CV (0.846)
    "submissions/history/submission_v3b.csv": VotingClassifier([("lr", lr(0.1)), ("gb", GB()), ("rf", RF())], voting="soft"),  # CV 0.845
}
v2 = pd.read_csv("submissions/final/submission_v2.csv")
for f, m in models.items():
    p = m.fit(A, y).predict(B)
    s = pd.DataFrame({"PassengerId": te.PassengerId, "Survived": p.astype(int)}); s.to_csv(f, index=False)
    print(f, s.shape, "survive rate", round(s.Survived.mean(), 3), "| differs from v2:", (s.Survived != v2.Survived).sum())
