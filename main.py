"""Run six classifier configurations and save reproducible comparison artifacts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from algorithms.split_criteria import CRITERIA
from algorithms.baselines import BASELINES
from benchmarking.reporting import save_tables, plot_results, flatten
from benchmarking.resources import file_hash

ROOT = Path(__file__).resolve().parent


def main():
    """Launch sequential fresh-process repetitions and checkpoint completed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', type=int, nargs='+', choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument('--models', '--criteria', nargs='+', choices=CRITERIA + BASELINES, default=list(CRITERIA + BASELINES))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'comparison')
    parser.add_argument('--max-depth', type=int, default=12)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if args.max_depth < 0 or min(args.min_samples_leaf, args.repeats, args.batch_size) < 1:
        parser.error('Invalid depth, leaf size, repetition count or batch size')
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    benchmarks = sorted(set(args.benchmarks))
    models = list(dict.fromkeys(args.models))
    for number in benchmarks:
        for filename in ('dataset.npz', 'metadata.json'):
            if not (data_dir / f'benchmark{number}' / filename).is_file():
                parser.error(f'Missing benchmark{number}/{filename}; run scripts/process_benchmarks.py')
    output.mkdir(parents=True, exist_ok=True)
    # Hash source and inputs once at launch, retaining provenance for the report.
    source_paths = [ROOT / 'main.py', ROOT / 'requirements.txt']
    for directory in ('algorithms', 'benchmarking', 'scripts'):
        source_paths.extend((ROOT / directory).glob('*.py'))
    manifest = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
                    data_sha256={f'benchmark{b}': file_hash(data_dir / f'benchmark{b}' / 'dataset.npz') for b in benchmarks},
                    expected_runs=len(benchmarks) * len(models) * args.repeats,
                    completed_runs=0, status='running')
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    results = []
    # Fix native threading before workers import NumPy or scikit-learn as well as
    # applying runtime threadpoolctl limits inside each measurement worker.
    environment = os.environ.copy()
    environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    print('Benchmark Model                Rep Accuracy MacroF1 Train(s) Predict(s) TrainMiB PredictMiB', flush=True)
    try:
        for number in benchmarks:
            for repetition in range(1, args.repeats + 1):
                # Rotate model order between repetitions to reduce fixed-order bias.
                offset = (repetition - 1) % len(models)
                for model in models[offset:] + models[:offset]:
                    key = f'benchmark{number}_{model}_r{repetition}'
                    print(f'Running {key} ...', flush=True)
                    path = output / 'runs' / f'{key}.json'
                    predictions = output / 'predictions' / f'{key}.npz'
                    command = [sys.executable, '-m', 'benchmarking.resources', '--benchmark', str(number),
                               '--model', model, '--data-dir', str(data_dir), '--output', str(path),
                               '--max-depth', str(args.max_depth), '--min-samples-leaf', str(args.min_samples_leaf),
                               '--repetition', str(repetition), '--batch-size', str(args.batch_size), '--predictions', str(predictions)]
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
                    result = json.loads(path.read_text(encoding='utf-8'))
                    results.append(result)
                    row = flatten(result)
                    print(f"B{number:<8} {model:<20} {repetition:3} {row['accuracy']:8.4f} {row['macro_f1']:7.4f} {row['training_seconds']:8.3f} {row['prediction_seconds']:10.3f} {row['training_sampled_peak_rss_mib']:8.1f} {row['prediction_sampled_peak_rss_mib']:10.1f}", flush=True)
                    if result['warnings']:
                        print(f"  Warnings recorded: {[w['category'] for w in result['warnings']]}", flush=True)
                    save_tables(results, output)
                    manifest['completed_runs'] = len(results)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if not args.no_plot:
            plot_results(results, output)
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'incomplete'
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'\nSaved {len(results)} runs and comparison tables to {output}', flush=True)


if __name__ == '__main__':
    main()
