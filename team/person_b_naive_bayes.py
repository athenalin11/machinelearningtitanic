"""Person B Gaussian Naive Bayes only: nested variance-smoothing search."""
from sklearn.naive_bayes import GaussianNB
from person_b_common import run_experiment

SMOOTHING = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
             1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def configurations(smoke=False):
    return {'gaussian_nb': (GaussianNB(),
            {'var_smoothing': [1e-9, 1e-3] if smoke else SMOOTHING})}


if __name__ == '__main__':
    run_experiment(configurations, 'naive_bayes')
