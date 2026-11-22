"""Replay a small real Wine example and generate its report tables.

Use only the flavonoid measurement for two cultivars, keeping selected training
and test rows in their existing benchmark partitions. This is an illustration,
not a replacement for the full 13-feature, three-class Wine experiment.
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from algorithms import DecisionTreeClassifier
from algorithms.split_criteria import CRITERIA, split_scores
from scripts.process_benchmarks import load_dataset, split_indices


def table(name, header, rows, spec):
    """Save a compact, reproducible LaTeX table fragment."""
    lines = [r'\begin{tabular}{' + spec + '}', r'\toprule', header + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE / 'tables' / name).write_text('\n'.join(lines) + '\n')


def main():
    """Select source rows, enumerate splits, fit all criteria and check predictions."""
    source = HERE.parent / 'data/raw/benchmark9/wine.data'
    source_lines = source.read_text().splitlines()
    raw = np.loadtxt(source, delimiter=',')
    # Replay the actual preprocessing split on all 178 rows, not on the eight
    # illustrative rows. Membership is assigned by a seeded shuffle within class.
    dataset = load_dataset(9, HERE.parent / 'data/raw', [20, 100, 200])
    replay_train, replay_test, method = split_indices(dataset, 0.2, 42)
    with np.load(HERE.parent / 'data/processed/benchmark9/dataset.npz') as data:
        np.testing.assert_array_equal(replay_train, data['train_indices'])
        np.testing.assert_array_equal(replay_test, data['test_indices'])
        train = [int(i) for label in (1, 3) for i in sorted(data['train_indices']) if raw[i, 0] == label]
        train = [i for label in (1, 3) for i in [j for j in train if raw[j, 0] == label][:3]]
        test = [int(next(i for i in sorted(data['test_indices']) if raw[i, 0] == label)) for label in (1, 3)]
    if set(train) & set(test):
        raise ValueError('Illustration partitions must be disjoint')
    # Column 0 is the source label; column 7 is flavonoids, without transformation.
    X, y = raw[train, 7:8], raw[train, 0].astype(int)
    test_X, test_y = raw[test, 7:8], raw[test, 0].astype(int)
    source_rows = [[i + 1, 'Training' if i in train else 'Testing', int(raw[i, 0]), f'{raw[i, 7]:.2f}'] for i in train + test]
    (HERE / 'tables').mkdir(exist_ok=True)
    # Preserve the source text exactly, including decimal formatting and all 14
    # columns, before showing any assigned use or selected predictor columns.
    selected = sorted(train + test)
    verbatim = [r'\begingroup\scriptsize', r'\begin{verbatim}']
    for i in selected:
        verbatim += [f'Source row {i + 1}:', source_lines[i]]
    verbatim += [r'\end{verbatim}', r'\endgroup']
    (HERE / 'tables/wine_example_raw.tex').write_text('\n'.join(verbatim) + '\n')
    (HERE / 'wine_example_raw.csv').write_text('\n'.join(source_lines[i] for i in selected) + '\n')
    table('wine_example_rows.tex', 'Source row & Assigned use & Original cultivar & Flavonoids', source_rows, '@{}rlrr@{}')
    split_counts = [[str(c), str(int(np.sum(raw[:, 0] == c))),
                     str(int(np.sum(raw[replay_train, 0] == c))), str(int(np.sum(raw[replay_test, 0] == c)))]
                    for c in (1, 2, 3)]
    table('wine_example_assignment.tex', 'Original cultivar & All source rows & Training rows & Testing rows', split_counts,
          '@{}rrrr@{}')
    with (HERE / 'wine_example_rows.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_row_1_based', 'partition', 'cultivar', 'flavonoids'])
        writer.writerows(source_rows)
    values = np.unique(X[:, 0])
    candidates, scores_table = [], []
    for low, high in zip(values[:-1], values[1:]):
        threshold = float(low / 2 + high / 2)
        mask = X[:, 0] <= threshold
        if min(mask.sum(), (~mask).sum()) < 2:
            continue
        left = [int(np.sum(y[mask] == c)) for c in (1, 3)]
        right = [int(np.sum(y[~mask] == c)) for c in (1, 3)]
        scores = {c: float(split_scores([3, 3], left, right, c)[0]) for c in CRITERIA}
        candidates.append(dict(threshold=threshold, left_counts=left, right_counts=right, scores=scores))
        scores_table.append([f'{threshold:.3f}', f'({left[0]}, {left[1]})', f'({right[0]}, {right[1]})'] + [f'{scores[c]:.4f}' for c in CRITERIA])
    table('wine_example_splits.tex', r'Threshold & Left counts & Right counts & \shortstack{Information\\gain} & \shortstack{Gain\\ratio} & \shortstack{Gini impurity\\reduction}', scores_table, '@{}rrrrrr@{}')
    results = []
    for criterion in CRITERIA:
        model = DecisionTreeClassifier(criterion=criterion).fit(X, y)
        prediction = model.predict(test_X)
        np.testing.assert_allclose(model.root_.threshold, 2.005)
        np.testing.assert_array_equal(prediction, test_y)
        np.testing.assert_array_equal(model.predict(X), y)
        if model.node_count_ != 3 or model.classes_[model.root_.left.prediction] != 3 or model.classes_[model.root_.right.prediction] != 1:
            raise ValueError('Unexpected illustrative tree')
        results.append(dict(criterion=criterion, threshold=model.root_.threshold, nodes=model.node_count_,
                            train_predictions=model.predict(X).tolist(), test_predictions=prediction.tolist()))
        print(f'{criterion}: flavonoids <= {model.root_.threshold:.3f} -> cultivar 3; otherwise cultivar 1; test predictions {prediction.tolist()}')
    table('wine_example_predictions.tex', 'Source row & Flavonoids & Branch & Prediction & Actual',
          [[i + 1, f'{raw[i, 7]:.2f}', 'Left' if raw[i, 7] <= 2.005 else 'Right', int(label), int(raw[i, 0])]
           for i, label in zip(test, results[0]['test_predictions'])], '@{}rrlrr@{}')
    (HERE / 'wine_example_results.json').write_text(json.dumps(dict(
        source='data/raw/benchmark9/wine.data', feature='flavonoids', raw_column_1_based=8,
        training_source_rows=[i + 1 for i in train], test_source_rows=[i + 1 for i in test],
        split=dict(method=method, seed=42, requested_test_fraction=0.2,
                   full_training_rows=len(replay_train), full_test_rows=len(replay_test),
                   replay_matches_saved_indices=True),
        count_order=[1, 3], candidates=candidates, models=results), indent=2) + '\n')
    print('Replayed the full 178-row split: 142 training and 36 testing rows; saved memberships match exactly.')


if __name__ == '__main__':
    main()
