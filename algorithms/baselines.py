"""Existing scikit-learn classifiers used as comparison baselines.

The project tree remains our own implementation. These pipelines deliberately
fit feature scaling inside fit, using training data only, and include its cost
in the measured training and prediction phases.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASELINES = ("logistic_regression", "knn", "gaussian_nb")


def make_baseline(name):
    """Construct an unfitted classifier with fixed, reportable hyperparameters."""
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=5000, tol=1e-4, random_state=42))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=5, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=1))
    if name == "gaussian_nb":
        return GaussianNB(var_smoothing=1e-9)
    raise ValueError(f"Unknown baseline {name!r}; choose from {BASELINES}")


def describe_baseline(model):
    """Record all constructor settings, including library defaults, as JSON data."""
    def describe(estimator):
        """Convert one estimator's scalar parameters into a portable record."""
        return dict(implementation=f"{type(estimator).__module__}.{type(estimator).__name__}",
                    parameters=estimator.get_params(deep=False))

    if hasattr(model, "steps"):
        return {name: describe(estimator) for name, estimator in model.steps}
    return describe(model)
