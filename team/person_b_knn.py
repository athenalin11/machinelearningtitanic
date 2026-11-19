"""Person B kNN only: K/weights tuning and scaled vs unscaled comparison."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from person_b_common import run_experiment

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]


def configurations(smoke=False):
    configs = {}
    for scaled in [False, True]:
        name = 'knn_scaled' if scaled else 'knn_unscaled'
        configs[name] = (Pipeline([
            ('scale', StandardScaler() if scaled else 'passthrough'),
            ('clf', KNeighborsClassifier(metric='euclidean', algorithm='brute', n_jobs=1))]),
            {'clf__n_neighbors': [3, 9] if smoke else K_VALUES,
             'clf__weights': ['uniform', 'distance']})
    return configs


if __name__ == '__main__':
    run_experiment(configurations, 'knn')
