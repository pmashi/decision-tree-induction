"""Measure training/prediction wall time and sampled resident process memory.

main.py launches this module in a fresh process for each dataset/criterion.
Standalone usage: python -m benchmarking.resources --benchmark 1 --criterion gain
"""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import threading
import warnings
from time import perf_counter

import numpy as np
import psutil
import sklearn
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

from algorithms import DecisionTreeClassifier
from algorithms.split_criteria import CRITERIA
from algorithms.baselines import BASELINES, make_baseline, describe_baseline
from .evaluation import evaluate


ROOT = Path(__file__).resolve().parents[1]
MODELS = CRITERIA + BASELINES


def measure_call(function, *args, interval=0.005):
    """Return function result and elapsed seconds/baseline/peak RSS in MiB.

    A background thread samples resident memory every 5 ms by default. This
    includes Python, loaded datasets, model and temporary arrays, not just model
    storage. Peaks shorter than a sampling interval may be missed. Reading the
    baseline and starting/stopping the sampler are outside the wall-time region.
    """
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        """Observe this process until the measured operation finishes."""
        while not stop.wait(interval):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = perf_counter()
    try:
        result = function(*args)
    finally:
        elapsed = perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        thread.join()
    mib = 1024 ** 2
    return result, dict(seconds=elapsed, baseline_rss_mib=baseline / mib,
                        sampled_peak_rss_mib=peak[0] / mib,
                        sampled_increase_mib=(peak[0] - baseline) / mib)


def predict_batches(model, X, batch_size=2048):
    """Bound prediction working memory while evaluating every supplied row.

    All models use the same batching policy. In particular, brute-force KNN
    never needs a full test-by-training distance matrix for Poker's million rows.
    """
    predictions = np.empty(len(X), dtype=np.int32)
    for start in range(0, len(X), batch_size):
        predictions[start:start + batch_size] = model.predict(X[start:start + batch_size])
    return predictions


def file_hash(path):
    """Hash input bytes so runs can be traced to the exact prepared dataset."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_experiment(data_dir, benchmark, criterion, max_depth=12, min_samples_leaf=2,
                   repetition=1, batch_size=2048, prediction_path=None):
    """Load one prepared dataset, measure fit/predict, then evaluate predictions.

    File loading, scoring and output writing are excluded from timings. Memory
    measurements begin after loading, but total RSS includes the loaded arrays.
    """
    folder = Path(data_dir) / f"benchmark{benchmark}"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    with np.load(folder / "dataset.npz", allow_pickle=False) as data:
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    is_tree = criterion in CRITERIA
    model = (DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
             if is_tree else make_baseline(criterion))
    # Match native thread limits across models. Imports are shared even for tree
    # workers so process RSS has a comparable library/runtime baseline.
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, training = measure_call(model.fit, X_train, y_train)
        predictions, prediction = measure_call(predict_batches, model, X_test, batch_size)
        libraries = threadpool_info()
        # Training scores expose overfitting; their computation is not benchmarked.
        train_predictions = predict_batches(model, X_train, batch_size)
    metrics = evaluate(y_test, predictions, len(metadata["classes"]))
    warning_records = [dict(category=type(w.message).__name__, message=str(w.message)) for w in caught]
    if prediction_path:
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, y_true=y_test, y_pred=predictions,
                            y_train_true=y_train, y_train_pred=train_predictions)
    classifier = model.steps[-1][1] if hasattr(model, "steps") else model
    # A majority-class predictor gives context for imbalanced datasets like Poker.
    majority = int(np.bincount(y_train, minlength=len(metadata["classes"])).argmax())
    return dict(benchmark=benchmark, dataset=metadata["name"], criterion=criterion,
                model=criterion, family="custom_tree" if is_tree else "sklearn_baseline",
                repetition=repetition, warnings=warning_records,
                converged=not any(w["category"] == "ConvergenceWarning" for w in warning_records),
                solver_iterations=getattr(classifier, "n_iter_", np.array([])).tolist(),
                input_sha256=file_hash(folder / "dataset.npz"),
                preprocessing_sha256=file_hash(folder / "metadata.json"),
                predictions_file=str(prediction_path) if prediction_path else None,
                train_rows=len(y_train), test_rows=len(y_test), features=X_train.shape[1],
                classes=metadata["classes"], split=metadata["split"], preprocessing_notes=metadata["notes"],
                config=(dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                            min_samples_split=2, min_impurity_decrease=0.0) if is_tree else describe_baseline(model)),
                tree=(dict(nodes=model.node_count_, leaves=model.leaf_count_, depth=model.depth_) if is_tree else None),
                training=training, prediction=prediction, evaluation=metrics,
                training_evaluation=evaluate(y_train, train_predictions, len(metadata["classes"])),
                majority_baseline_accuracy=float(np.mean(y_test == majority)),
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 psutil=psutil.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, os=platform.platform(),
                                 processor=platform.processor(), logical_cpus=psutil.cpu_count(),
                                 system_memory_gib=psutil.virtual_memory().total / 1024 ** 3),
                measurement=dict(memory="sampled process RSS, includes data and Python runtime",
                                 sample_interval_seconds=0.005, repetitions=1,
                                 batch_size=batch_size, native_threads=1, threadpools=libraries,
                                 timing="perf_counter wall time; includes model-specific scaling and prediction batching; excludes loading, evaluation and writing"),
                completed_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    """Run one isolated experiment, printing JSON or saving it to --output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--criterion", "--model", choices=MODELS, default="gain")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repetition < 1:
        parser.error("batch-size and repetition must be positive")
    result = run_experiment(args.data_dir, args.benchmark, args.criterion, args.max_depth, args.min_samples_leaf,
                            args.repetition, args.batch_size, args.predictions)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
