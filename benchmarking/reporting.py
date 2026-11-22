"""Create CSV tables, report discussion notes and static comparison charts."""

from collections import defaultdict
import csv
import json

import numpy as np


def write_csv(path, rows):
    """Write a nonempty list of dictionaries using its first row's column order."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten(result):
    """Preserve one run's numeric scores, timings, memory and identifying fields."""
    row = dict(benchmark=result["benchmark"], dataset=result["dataset"], model=result["model"],
               repetition=result["repetition"], train_rows=result["train_rows"], test_rows=result["test_rows"],
               features=result["features"], converged=result["converged"], warnings=len(result["warnings"]))
    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        row[name] = result["evaluation"][name]
        row["train_" + name] = result["training_evaluation"][name]
    row["majority_accuracy"] = result["majority_baseline_accuracy"]
    for phase in ("training", "prediction"):
        for name, value in result[phase].items():
            row[f"{phase}_{name}"] = value
    for name in ("nodes", "leaves", "depth"):
        row[name] = result["tree"][name] if result["tree"] else None
    return row


def aggregate(results):
    """Compute per-dataset/model means and sample standard deviations of repeats."""
    groups = defaultdict(list)
    for result in results:
        groups[(result["benchmark"], result["model"])].append(flatten(result))
    aggregated = []
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "train_accuracy", "train_macro_f1",
               "majority_accuracy", "training_seconds", "prediction_seconds",
               "training_baseline_rss_mib", "training_sampled_peak_rss_mib", "training_sampled_increase_mib",
               "prediction_baseline_rss_mib", "prediction_sampled_peak_rss_mib", "prediction_sampled_increase_mib"]
    for (benchmark, model), rows in sorted(groups.items()):
        row = dict(benchmark=benchmark, dataset=rows[0]["dataset"], model=model, repeats=len(rows),
                   convergence_warnings=sum(not r["converged"] for r in rows))
        for metric in metrics:
            values = [r[metric] for r in rows]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregated.append(row)
    return aggregated


def save_tables(results, output):
    """Save individual runs, repeat summaries, class scores and report-ready notes."""
    rows = [flatten(result) for result in results]
    summary = aggregate(results)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summary)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    per_class = []
    for result in results:
        for i, label in enumerate(result["classes"]):
            metrics = result["evaluation"]
            per_class.append(dict(benchmark=result["benchmark"], model=result["model"], repetition=result["repetition"],
                                  class_id=i, label=label, support=metrics["support"][i],
                                  precision=metrics["precision_per_class"][i], recall=metrics["recall_per_class"][i],
                                  f1=metrics["f1_per_class"][i]))
    write_csv(output / "per_class.csv", per_class)
    lines = ["# Classification comparison", "",
             "Means and sample standard deviations across repeated fresh-process runs of the same fixed split.",
             "Repeated timings measure runtime variation, not uncertainty across different train/test splits.", "",
             "| B | Dataset | Model | Runs | Accuracy | Macro F1 | Train seconds ± SD | Predict seconds ± SD | Train peak RSS MiB | Predict peak RSS MiB |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['benchmark']} | {row['dataset']} | {row['model']} | {row['repeats']} | {row['accuracy_mean']:.4f} | {row['macro_f1_mean']:.4f} | {row['training_seconds_mean']:.4f} ± {row['training_seconds_std']:.4f} | {row['prediction_seconds_mean']:.4f} ± {row['prediction_seconds_std']:.4f} | {row['training_sampled_peak_rss_mib_mean']:.1f} | {row['prediction_sampled_peak_rss_mib_mean']:.1f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_discussion(results, summary, output)


def save_discussion(results, summary, output):
    """Summarize observed tradeoffs without treating test winners as tuned models."""
    models = sorted({r["model"] for r in summary})
    overall = []
    for model in models:
        rows = [r for r in summary if r["model"] == model]
        overall.append(dict(model=model, benchmarks=len(rows),
                            mean_accuracy=float(np.mean([r["accuracy_mean"] for r in rows])),
                            mean_macro_f1=float(np.mean([r["macro_f1_mean"] for r in rows])),
                            total_train_seconds=sum(r["training_seconds_mean"] for r in rows),
                            total_predict_seconds=sum(r["prediction_seconds_mean"] for r in rows)))
    overall.sort(key=lambda r: r["mean_macro_f1"], reverse=True)
    write_csv(output / "overall.csv", overall)
    lines = ["# Report discussion notes", "", "## Overall comparison", "",
             "Each dataset has equal weight in the averages below; Poker's million test rows do not dominate them.",
             "Compare models only once they cover the same datasets. These are descriptive averages, not significance tests.", "",
             "| Model | Datasets | Mean accuracy | Mean macro F1 | Total train seconds | Total predict seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in overall:
        lines.append(f"| {row['model']} | {row['benchmarks']} | {row['mean_accuracy']:.4f} | {row['mean_macro_f1']:.4f} | {row['total_train_seconds']:.3f} | {row['total_predict_seconds']:.3f} |")
    lines.extend(["", "## Dataset observations", ""])
    for benchmark in sorted({r["benchmark"] for r in summary}):
        rows = [r for r in summary if r["benchmark"] == benchmark]
        best_f1 = max(r["macro_f1_mean"] for r in rows)
        best_acc = max(r["accuracy_mean"] for r in rows)
        f1_winners = ", ".join(r["model"] for r in rows if abs(r["macro_f1_mean"] - best_f1) < 1e-10)
        acc_winners = ", ".join(r["model"] for r in rows if abs(r["accuracy_mean"] - best_acc) < 1e-10)
        lines.append(f"- B{benchmark} ({rows[0]['dataset']}): highest observed macro F1 {best_f1:.4f} ({f1_winners}); highest accuracy {best_acc:.4f} ({acc_winners}). Training-majority baseline accuracy: {rows[0]['majority_accuracy_mean']:.4f}.")
    warnings = [(r["benchmark"], r["model"], r["repetition"], w) for r in results for w in r["warnings"]]
    lines.extend(["", "## Convergence and warnings", ""])
    if not warnings:
        lines.append("No warnings were emitted by the measured model operations.")
    for benchmark, model, repetition, warning in warnings:
        lines.append(f"- B{benchmark} / {model} / repetition {repetition}: {warning['category']}: {warning['message']}")
    lines.extend(["", "## Method and limitations", "",
        "- Baselines use existing scikit-learn implementations: logistic regression (StandardScaler, lbfgs, C=1, max_iter=5000), KNN (StandardScaler, k=5, uniform Euclidean brute-force search), GaussianNB (var_smoothing=1e-9). Full defaults and package versions are recorded in each run JSON.",
        "- The custom trees use gain, gain ratio or Gini, maximum depth 12 and minimum leaf size 2 unless overridden; consult each run config for the actual settings. Direct binary gain ratio is not full C4.5.",
        "- All models use exactly the same prepared partitions. Scaling is fitted on training rows only. Hyperparameters are fixed in advance; no test-set tuning is performed. Per-dataset winners are retrospective descriptions, not deployable model-selection rules.",
        "- One fresh subprocess per repetition/configuration; native BLAS/OpenMP pools limited to one thread. Timings include baseline scaling and shared batched prediction. Data loading, evaluation, prediction-file writing and training-score computation are excluded.",
        "- Memory is process RSS sampled every 5 ms, including Python, common library imports, data, model and temporary allocations. Peaks shorter than the interval can be missed. Baseline and increase over baseline are also in runs.csv; these are not model-only storage estimates.",
        "- The same deterministic split is repeated. Timing SD is not a confidence interval for accuracy. Cross-validation or additional independent splits would be needed to assess generalization variability.",
        "- Accuracy can hide poor minority-class recall. Use macro F1, per_class.csv and confusion matrices for the imbalanced cervical, Poker and quality datasets. Macro averages include all defined classes; undefined class metrics are zero.",
        "- Training accuracy and training macro F1 are recorded to help identify overfitting. KNN's training score includes each training example as its own neighbor and should be interpreted accordingly.",
        "- Hardware is adapted from regression into fixed PRP bands; Vowels uses per-utterance statistics; Steel uses a chronological split and same-time measurements. Source duplicates remain. These choices limit comparison with published benchmark scores.",
        "- GaussianNB's Gaussian feature assumption is imperfect for one-hot columns. KNN uses Euclidean distances on scaled one-hot and numeric features. Discuss these modeling choices rather than interpreting rankings as universal algorithm superiority.",
        "- Library implementations use optimized native routines while the custom tree orchestrates searches in Python/NumPy; timing differences reflect implementation as well as algorithm.",
        "", "## Reproducibility artifacts", "",
        "runs.csv retains each measurement; summary.csv holds repeat means/SD; overall.csv holds equal-dataset averages; per_class.csv holds class-level scores. results.json and individual run JSONs retain confusion matrices, full settings, versions, hardware, input SHA-256 hashes, warning text and iteration counts. predictions/ retains true/predicted training and test labels. manifest.json records command arguments and source/data hashes."])
    (output / "discussion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(results, output):
    """Save legible heatmaps for six-model comparisons over all ten datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rows = aggregate(results)
    models = list(dict.fromkeys(r["model"] for r in results))
    benchmarks = sorted({r["benchmark"] for r in rows})
    lookup = {(r["benchmark"], r["model"]): r for r in rows}
    names = [next(r["dataset"] for r in rows if r["benchmark"] == b) for b in benchmarks]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), layout="constrained")
    for axis, metric, title in zip(axes.flat,
            ("accuracy_mean", "macro_f1_mean", "training_seconds_mean", "prediction_seconds_mean"),
            ("Test accuracy", "Test macro F1", "Training seconds (log colors)", "Prediction seconds (log colors)")):
        values = np.array([[lookup[(b, model)][metric] for model in models] for b in benchmarks])
        is_time = "seconds" in metric
        norm = LogNorm(vmin=max(values.min(), 1e-6), vmax=max(values.max(), 1e-5)) if is_time else None
        picture = axis.imshow(values, aspect="auto", cmap="viridis", norm=norm,
                              **({} if is_time else dict(vmin=0, vmax=1)))
        axis.set_xticks(range(len(models)), models, rotation=25, ha="right")
        axis.set_yticks(range(len(benchmarks)), [f"B{b}: {n}" for b, n in zip(benchmarks, names)])
        axis.set_title(title)
        for i in range(len(benchmarks)):
            for j in range(len(models)):
                text = f"{values[i, j]:.3g}" if is_time else f"{values[i, j]:.3f}"
                normalized = picture.norm(values[i, j])
                axis.text(j, i, text, ha="center", va="center", color="black" if normalized > 0.55 else "white", fontsize=8)
        fig.colorbar(picture, ax=axis, shrink=0.8)
    fig.suptitle("Classifier comparison: fixed splits, repeated fresh-process measurements")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
