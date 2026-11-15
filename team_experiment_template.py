"""Template for teammates: evaluate ONE model on dataset B with the team's common protocol, so that results are comparable.

Usage:  python3 team_experiment_template.py          (edit MODEL / PARAM_GRID below)
Protocol: repeated stratified 5-fold (5 repeats, random_state=42); accuracy, F1, ROC-AUC (mean +- std over 25 folds).
Hyper-parameters must be tuned INSIDE each outer training fold (nested CV) - never on the whole data set and never
with the Kaggle leaderboard.  Output: results_<NAME>.csv and a LaTeX row to paste into results_table.tex.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# ---- choose your model (uncomment ONE) -------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier      # kNN          (Lecture 7b)
from sklearn.naive_bayes import GaussianNB              # Naive Bayes  (Lecture 7a)
from sklearn.tree import DecisionTreeClassifier         # Decision tree (Lecture 6)
from sklearn.svm import SVC                             # SVM          (Lectures 4-5)

NAME = "kNN"
MODEL = Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())])   # scale for kNN / SVM; not needed for trees
PARAM_GRID = {"clf__n_neighbors": [3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}
# Examples:
#   NAME="NaiveBayes"; MODEL=GaussianNB(); PARAM_GRID={"var_smoothing":[1e-9,1e-7,1e-5,1e-3]}
#   NAME="DecisionTree"; MODEL=DecisionTreeClassifier(random_state=0); PARAM_GRID={"max_depth":[2,3,4,5,6,8],"min_samples_leaf":[1,5,10,20]}
#   NAME="SVM_RBF"; MODEL=Pipeline([("scale",StandardScaler()),("clf",SVC(probability=True))]); PARAM_GRID={"clf__C":[0.3,1,3,10],"clf__gamma":[0.01,0.03,0.1]}

DATA = "B_v4_FamilyFlags_AgeTitlePclassParch"          # label-free dataset: safe for ordinary CV
tr = pd.read_csv(f"datasets/{DATA}_train.csv")
y = tr.Survived; X = tr.drop(columns=["PassengerId", "Survived", "fold"])

inner = StratifiedKFold(5, shuffle=True, random_state=1)
est = GridSearchCV(MODEL, PARAM_GRID, cv=inner, scoring="accuracy") if PARAM_GRID else MODEL      # nested CV
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
r = cross_validate(est, X, y, cv=outer, scoring=["accuracy", "f1", "roc_auc"], return_train_score=True)
res = {"Model": NAME, "Acc": r["test_accuracy"].mean(), "Acc_std": r["test_accuracy"].std(), "F1": r["test_f1"].mean(),
       "AUC": r["test_roc_auc"].mean(), "TrainAcc": r["train_accuracy"].mean()}
print(res); pd.DataFrame([res]).to_csv(f"results_{NAME}.csv", index=False)
print(f"LaTeX row: {NAME} & {res['Acc']:.4f} $\\pm$ {res['Acc_std']:.4f} & {res['F1']:.4f} & {res['AUC']:.4f} \\\\")
# TODO for the report: (1) learning/validation curve over the key hyper-parameter (train vs CV accuracy),
# (2) best hyper-parameters, (3) 2-3 sentences on why the model does / does not suit this data.
