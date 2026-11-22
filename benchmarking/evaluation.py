"""Evaluate classification predictions without depending on an ML library.

Can also be run with: python -m benchmarking.evaluation truth.npy predictions.npy
"""

import argparse
import json

import numpy as np


def evaluate(y_true, y_pred, n_classes):
    """Return accuracy, macro precision/recall/F1, and a confusion matrix.

    Inputs are zero-based integer class IDs. Rows in the confusion matrix are
    true classes; columns are predictions. Undefined per-class metrics are zero,
    and macro averages include all n_classes (including absent test classes).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError("Expected equally sized, nonempty 1D label arrays")
    if not isinstance(n_classes, int) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    for values in (y_true, y_pred):
        if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= n_classes:
            raise ValueError("Labels must be integer IDs in [0, n_classes)")
    matrix = np.bincount(y_true.astype(np.int64) * n_classes + y_pred,
                         minlength=n_classes ** 2).reshape(n_classes, n_classes)
    correct = matrix.diagonal().astype(float)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(correct, predicted, out=np.zeros(n_classes), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(n_classes), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=precision + recall > 0)
    return dict(accuracy=float(correct.sum() / len(y_true)), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                confusion_matrix=matrix.tolist(), precision_per_class=precision.tolist(),
                recall_per_class=recall.tolist(), f1_per_class=f1.tolist(), support=support.tolist())


def main():
    """Print evaluation JSON for two saved NumPy arrays of encoded class IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth")
    parser.add_argument("predictions")
    parser.add_argument("--n-classes", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(np.load(args.truth, allow_pickle=False),
                             np.load(args.predictions, allow_pickle=False), args.n_classes), indent=2))


if __name__ == "__main__":
    main()
