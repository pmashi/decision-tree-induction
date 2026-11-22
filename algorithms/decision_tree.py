"""A deterministic binary decision tree, implemented without ML libraries."""

from dataclasses import dataclass

import numpy as np

from .split_criteria import CRITERIA, split_scores


@dataclass
class Node:
    """One tree node; feature=-1 identifies a leaf with a majority prediction."""
    prediction: int
    samples: int
    feature: int = -1
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None


class DecisionTreeClassifier:
    """Greedily split numeric predictors using gain, gain ratio or Gini decrease.

    Each feature is sorted at a node. Cumulative class counts evaluate every
    boundary between distinct values without repeatedly scanning each subset.
    Ties choose the first feature/threshold, and majority ties choose the first
    sorted class. One-hot columns from preprocessing are treated as numeric.
    """

    def __init__(self, criterion="gain", max_depth=12, min_samples_split=2,
                 min_samples_leaf=2, min_impurity_decrease=0.0):
        """Configure stopping rules; root depth is zero and None allows any depth."""
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a nonnegative integer or None")
        
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        
        if not np.isfinite(min_impurity_decrease) or min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be finite and nonnegative")
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        """Build a tree from finite numeric X and one class label per row."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a nonempty 2D matrix")
        
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one label per row")
        
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numbers; preprocess missing values first")
        
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.node_count_ = self.leaf_count_ = self.depth_ = 0
        
        # Store training arrays only while building, not as part of the final model.
        self._X, self._y = X, encoded
        try:
            self.root_ = self._build(np.arange(len(y)), depth=0)
        finally:
            del self._X, self._y
        return self

    def _best_split(self, indices, parent_counts):
        """Find the highest scoring legal threshold, or None if no split improves."""
        best, best_score = None, -np.inf
        n, classes = len(indices), len(self.classes_)
        
        for feature in range(self.n_features_in_):
            values = self._X[indices, feature]
            order = np.argsort(values, kind="stable")
            values = values[order]
            # A threshold cannot separate equal values; also enforce leaf size.
            positions = np.flatnonzero(values[:-1] < values[1:])
            positions = positions[(positions + 1 >= self.min_samples_leaf) &
                                  (n - positions - 1 >= self.min_samples_leaf)]
            
            if not len(positions):
                continue
            
            counts = np.zeros((n, classes), dtype=np.int64)
            counts[np.arange(n), self._y[indices[order]]] = 1
            np.cumsum(counts, axis=0, out=counts)
            left = counts[positions]
            scores, reductions = split_scores(parent_counts, left, parent_counts - left, self.criterion)
            
            # Numerical noise must not create zero-gain branches. The threshold
            # is a local node impurity reduction, not a root-weighted reduction.
            valid = (reductions > 1e-12) & (reductions >= self.min_impurity_decrease)
            scores = np.where(valid, scores, -np.inf)
            candidate = int(np.argmax(scores))
            if scores[candidate] > best_score + 1e-12:
                best_score = float(scores[candidate])
                position = positions[candidate]
                low, high = float(values[position]), float(values[position + 1])
                threshold = low / 2 + high / 2
                # Adjacent floats may round their midpoint up to high. Using low
                # in that case still separates the observed training values.
                if not low <= threshold < high:
                    threshold = low
                best = feature, threshold
        return best

    def _build(self, indices, depth):
        """Recursively partition rows until a stopping condition produces a leaf."""
        counts = np.bincount(self._y[indices], minlength=len(self.classes_))
        
        node = Node(prediction=int(np.argmax(counts)), samples=len(indices))
        self.node_count_ += 1
        self.depth_ = max(self.depth_, depth)
        stop = (np.count_nonzero(counts) == 1 or
                len(indices) < max(self.min_samples_split, 2 * self.min_samples_leaf) or
                (self.max_depth is not None and depth >= self.max_depth))
        split = None if stop else self._best_split(indices, counts)
        
        if split is None:
            self.leaf_count_ += 1
            return node
        node.feature, node.threshold = split
        goes_left = self._X[indices, node.feature] <= node.threshold
        node.left = self._build(indices[goes_left], depth + 1)
        node.right = self._build(indices[~goes_left], depth + 1)
        return node

    def predict(self, X):
        """Return original class labels, routing batches of rows through the tree."""
        if not hasattr(self, "root_"):
            raise ValueError("Call fit before predict")
        
        # Preserve input precision and avoid a full float64 copy of large test sets.
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X must be finite and match the training feature count")
        predictions = np.empty(len(X), dtype=np.int64)
        pending = [(self.root_, np.arange(len(X)))]
        
        while pending:
            node, indices = pending.pop()
            if node.feature == -1:
                predictions[indices] = node.prediction
            else:
                # Compare in the same precision used during fit, including when
                # a threshold lies between adjacent float32 feature values.
                goes_left = X[indices, node.feature].astype(np.float64) <= node.threshold
                if goes_left.any():
                    pending.append((node.left, indices[goes_left]))
                if (~goes_left).any():
                    pending.append((node.right, indices[~goes_left]))
        return self.classes_[predictions]
