"""Checks for leakage, category handling, splitting and sequence alignment."""

import unittest

import numpy as np

from process_benchmarks import Dataset, ROOT, encode_features, load_dataset, split_indices


class PreprocessingTests(unittest.TestCase):
    """Check preprocessing invariants using small fixtures and the vowel files."""

    def test_train_only_imputation_and_unknown_category(self):
        """Test-only values must not affect medians or the category vocabulary."""
        # Only the first two rows are training data. The test-only 999 must not
        # change the median of 2, and green must use the unknown-category column.
        X = np.array([["1", "red"], ["3", "blue"], ["?", "green"], ["999", "?"]])
        train, test, names, fitted = encode_features(X, ["value", "color"], {1}, np.array([0, 1]), np.array([2, 3]))
        self.assertEqual(fitted[0]["imputation_median"], 2)
        self.assertEqual(fitted[1]["categories"], ["blue", "red"])
        np.testing.assert_array_equal(test[:, 0], [2, 999])
        np.testing.assert_array_equal(test[:, -1], [1, 1])
        np.testing.assert_array_equal(train[:, -1], [0, 0])
        self.assertEqual(len(names), 4)

    def test_all_missing_numeric_fallback(self):
        """An empty training column uses zero without borrowing a test value."""
        train, test, _, _ = encode_features(np.array([["?"], ["5"]]), ["x"], set(), np.array([0]), np.array([1]))
        self.assertEqual(train[0, 0], 0)
        self.assertEqual(test[0, 0], 5)

    def test_stratification_is_disjoint_complete_and_repeatable(self):
        """Every row appears once, rare classes stay trainable, and seeds repeat."""
        dataset = Dataset("test", np.zeros((11, 1)), np.array(["a"] * 5 + ["b"] * 5 + ["rare"]), ["x"])
        train, test, _ = split_indices(dataset, 0.2, 42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(11)))
        self.assertIn(10, train)
        self.assertEqual(set(dataset.y[test]), {"a", "b"})
        np.testing.assert_array_equal(train, split_indices(dataset, 0.2, 42)[0])

    def test_official_and_chronological_splits(self):
        """Honor official boundaries first, then chronological ordering if requested."""
        dataset = Dataset("test", np.zeros((10, 1)), np.array(["a"] * 10), ["x"], train_size=6)
        train, test, method = split_indices(dataset, 0.2, 42)
        np.testing.assert_array_equal(train, np.arange(6))
        np.testing.assert_array_equal(test, np.arange(6, 10))
        self.assertEqual(method, "official")
        dataset.train_size = None
        dataset.chronological = True
        train, test, method = split_indices(dataset, 0.2, 42)
        self.assertLess(train.max(), test.min())
        self.assertEqual(method, "chronological")

    def test_vowel_utterances_and_speaker_labels(self):
        """Check real sequence parsing against the supplied speaker counts."""
        dataset = load_dataset(6, ROOT / "data" / "raw", [20, 100, 200])
        # 640 recordings, each with 12 coefficients x 4 summaries + frame count.
        self.assertEqual(dataset.X.shape, (640, 49))
        self.assertEqual(dataset.train_size, 270)
        np.testing.assert_array_equal(np.unique(dataset.y[:270], return_counts=True)[1], [30] * 9)
        np.testing.assert_array_equal(np.unique(dataset.y[270:], return_counts=True)[1], [31, 35, 88, 44, 29, 24, 40, 50, 29])


if __name__ == "__main__":
    # Support direct execution as well as unittest discovery.
    unittest.main()
