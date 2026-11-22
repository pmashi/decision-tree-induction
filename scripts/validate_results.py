"""Audit a completed comparison against saved predictions and source datasets.

Usage: python scripts/validate_results.py --results-dir results/comparison
This does not train models or change measurements; it writes validation.json.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    """Read a file incrementally to verify its recorded identity."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    """Fail explicitly even when Python is run with optimization enabled."""
    if not condition:
        raise ValueError(message)


def validate(folder, data_dir):
    """Check run coverage, hashes, labels, metrics and repeat summary arithmetic."""
    manifest = json.loads((folder / 'manifest.json').read_text())
    runs = json.loads((folder / 'results.json').read_text())
    args = manifest['arguments']
    expected = {(b, m, r) for b in set(args['benchmarks']) for m in set(args['models'])
                for r in range(1, args['repeats'] + 1)}
    keys = [(run['benchmark'], run['model'], run['repetition']) for run in runs]
    require(manifest['status'] == 'complete', 'Runner has not finished')
    require(len(keys) == len(expected) and set(keys) == expected, 'Missing or duplicate runs')
    require(manifest['completed_runs'] == manifest['expected_runs'] == len(runs), 'Run counts disagree')
    for source, digest in manifest['source_sha256'].items():
        require(sha256(ROOT / source) == digest, f'Source changed since measurement: {source}')
    for b in sorted(set(args['benchmarks'])):
        dataset = data_dir / f'benchmark{b}'
        digest = sha256(dataset / 'dataset.npz')
        metadata_digest = sha256(dataset / 'metadata.json')
        require(digest == manifest['data_sha256'][f'benchmark{b}'], f'Changed input B{b}')
        with np.load(dataset / 'dataset.npz', allow_pickle=False) as raw:
            test_labels, train_labels = raw['y_test'], raw['y_train']
        for run in (r for r in runs if r['benchmark'] == b):
            require(run['input_sha256'] == digest and run['preprocessing_sha256'] == metadata_digest, 'Input hash mismatch')
            path = folder / 'predictions' / f"benchmark{b}_{run['model']}_r{run['repetition']}.npz"
            with np.load(path, allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions['y_true'], test_labels)
                np.testing.assert_array_equal(predictions['y_train_true'], train_labels)
                for prefix, metrics in (('', run['evaluation']), ('train_', run['training_evaluation'])):
                    truth, pred = predictions[f'y_{prefix}true'], predictions[f'y_{prefix}pred']
                    labels = np.arange(len(run['classes']))
                    matrix = confusion_matrix(truth, pred, labels=labels)
                    np.testing.assert_array_equal(matrix, metrics['confusion_matrix'])
                    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
                    np.testing.assert_allclose(accuracy_score(truth, pred), metrics['accuracy'])
                    for name, values in (('precision', precision), ('recall', recall), ('f1', f1)):
                        np.testing.assert_allclose(values, metrics[f'{name}_per_class'])
                        np.testing.assert_allclose(values.mean(), metrics[f'macro_{name}'])
                    np.testing.assert_array_equal(support, metrics['support'])
            for phase in ('training', 'prediction'):
                measure = run[phase]
                require(measure['seconds'] >= 0, 'Negative time')
                require(measure['sampled_peak_rss_mib'] >= measure['baseline_rss_mib'], 'Invalid memory peak')
        print(f'B{b}: predictions and train/test metrics validated against scikit-learn', flush=True)
    with (folder / 'summary.csv').open(newline='') as stream:
        summary = list(csv.DictReader(stream))
    require(len(summary) == len(expected) // args['repeats'], 'Summary row count mismatch')
    for row in summary:
        group = [r for r in runs if r['benchmark'] == int(row['benchmark']) and r['model'] == row['model']]
        require(len(group) == int(row['repeats']) == args['repeats'], 'Repeat count mismatch')
        for name, values in (
                ('accuracy', [r['evaluation']['accuracy'] for r in group]),
                ('macro_f1', [r['evaluation']['macro_f1'] for r in group]),
                ('training_seconds', [r['training']['seconds'] for r in group]),
                ('prediction_seconds', [r['prediction']['seconds'] for r in group])):
            np.testing.assert_allclose(float(row[name + '_mean']), np.mean(values))
            np.testing.assert_allclose(float(row[name + '_std']), np.std(values, ddof=1) if len(values) > 1 else 0)
    result = dict(status='passed', runs=len(runs), configurations=len(summary),
                  warning_categories=dict(Counter(w['category'] for r in runs for w in r['warnings'])),
                  checks=['coverage', 'source/input hashes', 'true label alignment', 'reference confusion matrices',
                          'reference train/test metrics', 'resource bounds', 'summary means/sample SD'])
    (folder / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    """Select the saved comparison and prepared data directories to audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results' / 'comparison')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    args = parser.parse_args()
    validate(args.results_dir, args.data_dir)


if __name__ == '__main__':
    main()
