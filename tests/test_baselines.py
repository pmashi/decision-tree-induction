"""Verify baseline integration, shared evaluation and repeat aggregation."""

import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from algorithms.baselines import BASELINES, make_baseline, describe_baseline
from benchmarking.evaluation import evaluate
from benchmarking.reporting import aggregate
from benchmarking.resources import predict_batches


class BaselineTests(unittest.TestCase):
    """Use a simple two-cluster problem without relying on benchmark accuracy."""

    def setUp(self):
        """Create enough examples for the fixed five-neighbor classifier."""
        self.X = np.array([[i, i * 2] for i in range(6)] + [[i, i * 2] for i in range(20, 26)], dtype=float)
        self.y = np.array([0] * 6 + [1] * 6)

    def test_all_baselines_and_batching(self):
        """Batched and direct predictions must agree, including the final short batch."""
        for name in BASELINES:
            model = make_baseline(name).fit(self.X, self.y)
            np.testing.assert_array_equal(predict_batches(model, self.X, 5), model.predict(self.X))
            np.testing.assert_array_equal(model.predict(self.X), self.y)
            self.assertTrue(describe_baseline(model))

    def test_scaling_is_fitted_only_on_training(self):
        """Extreme test values must not change the scaler's stored mean."""
        for name in ('logistic_regression', 'knn'):
            model = make_baseline(name).fit(self.X, self.y)
            model.predict([[10000, 20000]])
            np.testing.assert_allclose(model.steps[0][1].mean_, self.X.mean(axis=0))

    def test_metrics_match_library(self):
        """Our evaluation agrees with reference metrics, including an absent class."""
        y = np.array([0, 0, 1, 1, 2])
        pred = np.array([0, 1, 1, 2, 2])
        metrics = evaluate(y, pred, 4)
        self.assertAlmostEqual(metrics['accuracy'], accuracy_score(y, pred))
        self.assertAlmostEqual(metrics['macro_f1'], f1_score(y, pred, labels=range(4), average='macro', zero_division=0))


if __name__ == '__main__':
    unittest.main()
