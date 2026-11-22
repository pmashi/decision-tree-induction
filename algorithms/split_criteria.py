"""Impurity and split scores, calculated from class counts using NumPy."""

import numpy as np


CRITERIA = ("gain", "gain_ratio", "gini")


def entropy(counts):
    """Return base-2 entropy along the last axis; empty/pure nodes have zero."""
    counts = np.asarray(counts, dtype=float)
    
    totals = counts.sum(axis=-1, keepdims=True)
    
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)
    
    logs = np.zeros_like(probabilities)
    
    np.log2(probabilities, out=logs, where=probabilities > 0)
    
    return -(probabilities * logs).sum(axis=-1)


def gini_impurity(counts):
    """Return 1 - sum(p_class ** 2), or zero for an empty node."""
    counts = np.asarray(counts, dtype=float)
    
    totals = counts.sum(axis=-1, keepdims=True)
    
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)
    
    return np.where(totals.squeeze(axis=-1) > 0, 1 - (probabilities ** 2).sum(axis=-1), 0)


def split_scores(parent_counts, left_counts, right_counts, criterion):
    """Score one or many binary splits; larger scores are better.

    Child count arrays have shape (candidate_splits, classes), or (classes,).
    Return (selection_score, raw_impurity_reduction). Gain ratio divides entropy
    reduction by the entropy of the two branch sizes. Its raw reduction remains
    information gain, so min_impurity_decrease has a meaningful unnormalized unit.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    
    left_counts = np.asarray(left_counts, dtype=float)
    right_counts = np.asarray(right_counts, dtype=float)
    
    left_n, right_n = left_counts.sum(axis=-1), right_counts.sum(axis=-1)
    
    total = left_n + right_n
    
    left_weight = np.divide(left_n, total, out=np.zeros_like(total), where=total > 0)
    right_weight = np.divide(right_n, total, out=np.zeros_like(total), where=total > 0)
    
    impurity = gini_impurity if criterion == "gini" else entropy
    
    reduction = impurity(parent_counts) - left_weight * impurity(left_counts) - right_weight * impurity(right_counts)
    
    if criterion == "gain_ratio":
        split_information = entropy(np.stack((left_n, right_n), axis=-1))
        score = np.divide(reduction, split_information, out=np.zeros_like(reduction), where=split_information > 0)
    else:
        score = reduction
    return score, reduction
