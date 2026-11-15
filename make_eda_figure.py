import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
tr = pd.read_csv("train.csv")
fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
def bar(a, s, title, labels=None):
    s = s.round(3); a.bar([str(i) for i in s.index], s.values, color="#4C72B0"); a.set_ylim(0, 1)
    a.axhline(tr.Survived.mean(), color="gray", ls="--", lw=0.8); a.set_title(title, fontsize=9)
    for i, v in enumerate(s.values): a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    a.tick_params(labelsize=7)
bar(ax[0, 0], tr.groupby("Sex").Survived.mean(), "Sex")
bar(ax[0, 1], tr.groupby("Pclass").Survived.mean(), "Pclass")
bar(ax[0, 2], tr.groupby(pd.cut(tr.Age, [0, 12, 18, 35, 60, 100]), observed=True).Survived.mean(), "Age group")
fam = (tr.SibSp + tr.Parch + 1).clip(upper=6)
bar(ax[1, 0], tr.groupby(fam).Survived.mean(), "Family size (6 = 6+)")
bar(ax[1, 1], tr.groupby(pd.qcut(tr.Fare, 4, labels=["Q1", "Q2", "Q3", "Q4"])).Survived.mean(), "Fare quartile")
bar(ax[1, 2], tr.groupby(tr.Cabin.notna().map({True: "known", False: "missing"})).Survived.mean(), "Cabin")
fig.suptitle("Survival rate by feature (dashed: overall 38.4\\%)".replace("\\%", "%"), fontsize=10)
plt.tight_layout(); plt.savefig("eda.png", dpi=200)
