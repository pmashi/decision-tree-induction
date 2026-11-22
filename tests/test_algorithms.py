"""Hand-calculated checks and edge cases for tree training and evaluation."""

import unittest

import numpy as np

from algorithms import DecisionTreeClassifier
from algorithms.split_criteria import CRITERIA, entropy, gini_impurity, split_scores
from benchmarking.evaluation import evaluate
from benchmarking.resources import measure_call


class CriteriaTests(unittest.TestCase):
    """Check formulas independently against small known probability distributions."""

    def test_impurities(self):
        """Balanced binary entropy is one bit and Gini is one half."""
        self.assertAlmostEqual(entropy([2, 2]), 1)
        self.assertAlmostEqual(gini_impurity([2, 2]), 0.5)
        self.assertEqual(entropy([4, 0]), 0)
        self.assertEqual(gini_impurity([0, 0]), 0)

    def test_perfect_and_uninformative_splits(self):
        """A perfect split removes all impurity; identical proportions remove none."""
        for criterion in CRITERIA:
            score, _ = split_scores([2, 2], [2, 0], [0, 2], criterion)
            self.assertAlmostEqual(score, 0.5 if criterion == "gini" else 1)
            score, _ = split_scores([2, 2], [1, 1], [1, 1], criterion)
            self.assertAlmostEqual(score, 0)

    def test_unbalanced_gain_ratio(self):
        """Gain ratio divides information gain by binary branch-size entropy."""
        gain = 1 - 0.75 * (-(1 / 3) * np.log2(1 / 3) - (2 / 3) * np.log2(2 / 3))
        branch_entropy = -0.25 * np.log2(0.25) - 0.75 * np.log2(0.75)
        score, reduction = split_scores([2, 2], [1, 0], [1, 2], "gain_ratio")
        self.assertAlmostEqual(reduction, gain)
        self.assertAlmostEqual(score, gain / branch_entropy)


class TreeTests(unittest.TestCase):
    """Validate predictions, stopping rules and exact candidate selection."""

    def test_separable_data_all_criteria(self):
        """Every criterion finds the known separating threshold."""
        for criterion in CRITERIA:
            model = DecisionTreeClassifier(criterion=criterion).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
            self.assertEqual(model.root_.threshold, 1.5)
            self.assertEqual(model.node_count_, 3)
            np.testing.assert_array_equal(model.predict([[0], [3]]), [0, 1])

    def test_constant_features_and_majority_ties(self):
        """Unsplitable conflicting examples become a deterministic majority leaf."""
        model = DecisionTreeClassifier(min_samples_leaf=1).fit([[1], [1]], ["b", "a"])
        self.assertEqual(model.node_count_, 1)
        np.testing.assert_array_equal(model.predict([[1]]), ["a"])

    def test_stopping_rules(self):
        """Depth, leaf size and minimum improvement can prevent splitting."""
        for settings in ({"max_depth": 0}, {"min_samples_leaf": 3}, {"min_impurity_decrease": 2}):
            model = DecisionTreeClassifier(**settings).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
            self.assertEqual(model.leaf_count_, 1)

    def test_adjacent_float32_values(self):
        """Prediction must preserve the float64 midpoint used during training."""
        high = np.float32(1)
        low = np.nextafter(high, np.float32(0))
        X = np.array([[low], [low], [high], [high]], dtype=np.float32)
        model = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
        np.testing.assert_array_equal(model.predict(X), [0, 0, 1, 1])

    def test_split_matches_brute_force(self):
        """Compare the optimized root search to directly counting every partition."""
        rng = np.random.default_rng(9)
        X = rng.integers(0, 8, (40, 3))
        y = rng.integers(0, 3, 40)
        parent = np.bincount(y, minlength=3)
        for criterion in CRITERIA:
            best = (-np.inf, None, None)
            for feature in range(3):
                values = np.unique(X[:, feature])
                for low, high in zip(values[:-1], values[1:]):
                    threshold = (low + high) / 2
                    mask = X[:, feature] <= threshold
                    if min(mask.sum(), (~mask).sum()) < 2:
                        continue
                    score, reduction = split_scores(parent, np.bincount(y[mask], minlength=3),
                                                    np.bincount(y[~mask], minlength=3), criterion)
                    if reduction > 1e-12 and score > best[0] + 1e-12:
                        best = (float(score), feature, threshold)
            model = DecisionTreeClassifier(criterion=criterion, max_depth=1).fit(X, y)
            self.assertEqual((model.root_.feature, model.root_.threshold), best[1:])

    def test_invalid_input_and_refit(self):
        """Reject invalid predictor matrices and reset tree statistics on refit."""
        model = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            model.predict([[0]])
        with self.assertRaises(ValueError):
            model.fit([[np.nan]], [0])
        model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        model.fit([[1], [2]], [1, 1])
        self.assertEqual(model.node_count_, 1)


class BenchmarkTests(unittest.TestCase):
    """Check evaluation arithmetic and basic resource measurement output."""

    def test_known_confusion_matrix(self):
        """Known counts yield accuracy .75 and macro F1 (0.8 + 2/3)/2."""
        result = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), 2)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)

    def test_absent_class(self):
        """Absent classes use zero scores and are still included in macro averages."""
        result = evaluate(np.array([0]), np.array([0]), 2)
        self.assertEqual(result["macro_f1"], 0.5)

    def test_resource_measurement(self):
        """Measurements return the original result and nonnegative timings."""
        result, measurements = measure_call(sum, [1, 2, 3])
        self.assertEqual(result, 6)
        self.assertGreaterEqual(measurements["seconds"], 0)
        self.assertGreaterEqual(measurements["sampled_peak_rss_mib"], measurements["baseline_rss_mib"])


if __name__ == "__main__":
    unittest.main()
