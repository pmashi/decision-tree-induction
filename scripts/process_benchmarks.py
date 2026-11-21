"""Prepare the ten local datasets for classification. Requires only NumPy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


# Resolve default paths from this file, regardless of the terminal's directory.
ROOT = Path(__file__).resolve().parents[1]
# Source files use either empty cells or question marks for missing values.
MISSING = {"", "?"}


@dataclass
class Dataset:
    """Common representation used by all dataset loaders before encoding.

    X holds raw predictors (rows = examples, columns = features); y holds
    original class labels. features names the columns, and categorical stores
    their zero-based indices when they should be one-hot encoded.

    train_size marks the boundary when official training rows precede test rows.
    Otherwise, chronological selects a time-ordered split instead of a random
    stratified split. Optional groups contains subject IDs, never predictors.
    name, target and notes describe the task in the saved metadata.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    features: list[str]
    categorical: set[int] = field(default_factory=set)
    target: str = "class"
    notes: list[str] = field(default_factory=list)
    train_size: int | None = None
    chronological: bool = False
    groups: np.ndarray | None = None


def read_csv(path: Path, header: bool = False, delimiter: str = ","):
    """Return optional column names and a rectangular array of string cells.

    Keep cells as strings so dataset-specific code can distinguish categorical
    values from numbers. Reject inconsistent row widths before selecting columns.
    utf-8-sig also accepts files containing a UTF-8 byte-order mark.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [[value.strip() for value in row] for row in csv.reader(stream, delimiter=delimiter) if row]
    names = rows.pop(0) if header else None
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"Empty or ragged CSV: {path}")
    if names is not None and len(names) != len(rows[0]):
        raise ValueError(f"Header width mismatch: {path}")
    return names, np.asarray(rows)


def load_vowels(folder: Path) -> Dataset:
    """Convert variable-length utterances into 49 features and speaker labels.

    Preserve the supplied train/test boundary. Each utterance becomes one row:
    four statistics for each of 12 coefficients, plus its number of frames.
    """
    matrices, labels = [], []
    for split in ("train", "test"):
        blocks, block = [], []
        # Blank lines delimit recordings; the extra blank flushes the final one.
        for line in (folder / f"ae.{split}").read_text().splitlines() + [""]:
            if line.strip():
                values = [float(value) for value in line.split()]
                if len(values) != 12:
                    raise ValueError("Expected 12 coefficients per vowel frame")
                block.append(values)
            elif block:
                blocks.append(np.asarray(block))
                block = []
        # Recordings are grouped by speaker; size files give each group's count.
        counts = np.loadtxt(folder / f"size_ae.{split}", dtype=int).reshape(-1)
        if len(counts) != 9 or counts.sum() != len(blocks):
            raise ValueError("Vowel speaker counts do not match utterances")
        # Aggregate within each recording, without mixing examples or partitions.
        matrices.append(np.asarray([
            np.concatenate((b.mean(axis=0), b.std(axis=0), b.min(axis=0), b.max(axis=0), [len(b)]))
            for b in blocks
        ]))
        labels.append(np.repeat(np.arange(1, 10), counts).astype(str))
    features = [f"lpc_{i}_{stat}" for stat in ("mean", "std", "min", "max") for i in range(1, 13)]
    return Dataset("Japanese vowels", np.vstack(matrices), np.concatenate(labels), features + ["frames"],
                   target="speaker", train_size=len(labels[0]), notes=[
                       "One example per utterance; speaker labels follow the supplied size files.",
                       "Each coefficient is summarized by mean, population standard deviation, minimum and maximum; frame count is included. Temporal order is lost."])


def load_dataset(number: int, root: Path, hardware_bins: list[float]) -> Dataset:
    """Read one benchmark from root and separate its predictors from its target.

    Each branch handles a source format and records its modeling choices.
    Learned transformations (medians and category vocabularies) happen later,
    after the training/test split. hardware_bins defines fixed PRP class bounds.
    """
    folder = root / f"benchmark{number}"
    if number == 1:
        # Breast cancer: first column is the label; remaining values are categories.
        _, rows = read_csv(folder / "breast-cancer.data")
        return Dataset("Breast cancer", rows[:, 1:], rows[:, 0],
                       ["age", "menopause", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"],
                       set(range(9)), target="recurrence", notes=["All original feature values, including interval labels, are treated as categories."])
    if number == 2:
        # Cervical cancer: predict Biopsy without other diagnostic-result fields.
        names, rows = read_csv(folder / "risk_factors_cervical_cancer.csv", header=True)
        excluded = {"Hinselmann", "Schiller", "Citology", "Biopsy", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"}
        indices = [i for i, name in enumerate(names) if name not in excluded]
        return Dataset("Cervical cancer risk factors", rows[:, indices], rows[:, names.index("Biopsy")],
                       [names[i] for i in indices], target="Biopsy", notes=[
                           "Other screening outcomes and diagnosis columns are excluded to avoid using diagnostic results as risk predictors."])
    if number == 3:
        # Hardware: adapt numeric performance to classes; retain six predictors.
        _, rows = read_csv(folder / "machine.data")
        # Fixed thresholds do not learn anything from held-out targets.
        y = np.searchsorted(hardware_bins, rows[:, 8].astype(float), side="left").astype(str)
        return Dataset("Computer hardware", rows[:, 2:8], y, ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"],
                       target="PRP_band", notes=[
                           f"Adapted regression task: PRP bands have inclusive upper bounds {hardware_bins}; final band is above the last bound. Band IDs start at 0.",
                           "Vendor, model, raw PRP and estimated performance ERP are excluded from predictors. Results are not directly comparable to the original regression task."])
    if number == 4:
        # Contraception: last column is the class; demographic codes are categories.
        _, rows = read_csv(folder / "cmc.data")
        return Dataset("Contraceptive method choice", rows[:, :-1], rows[:, -1],
                       ["wife_age", "wife_education", "husband_education", "children", "wife_religion", "wife_working", "husband_occupation", "living_standard", "media_exposure"],
                       {1, 2, 4, 5, 6, 7, 8}, target="contraceptive_method")
    if number == 5:
        # HAR: read feature, label and subject files from both official partitions.
        folder /= "UCI HAR Dataset"
        # Prefix names with source indices because some feature names repeat.
        features = [f"{index}_{name}" for index, name in
                    (line.split(maxsplit=1) for line in (folder / "features.txt").read_text().splitlines())]
        xs, ys, groups = [], [], []
        for split in ("train", "test"):
            xs.append(np.loadtxt(folder / split / f"X_{split}.txt"))
            ys.append(np.loadtxt(folder / split / f"y_{split}.txt", dtype=int).astype(str))
            groups.append(np.loadtxt(folder / split / f"subject_{split}.txt", dtype=int))
            if not (len(xs[-1]) == len(ys[-1]) == len(groups[-1])):
                raise ValueError("HAR features, labels and subject IDs are misaligned")
        return Dataset("Human activity recognition", np.vstack(xs), np.concatenate(ys), features,
                       target="activity", train_size=len(ys[0]), groups=np.concatenate(groups),
                       notes=["Uses the 561 supplied engineered features; subject IDs are metadata, not predictors. Original feature normalization is retained."])
    if number == 6:
        # Vowels need a sequence parser instead of a standard tabular CSV reader.
        return load_vowels(folder)
    if number == 7:
        # Poker: five suit/rank pairs followed by the class; keep official splits.
        train = np.loadtxt(folder / "poker-hand-training-true.data", delimiter=",", dtype=np.int16)
        test = np.loadtxt(folder / "poker-hand-testing.data", delimiter=",", dtype=np.int16)
        rows = np.vstack((train, test))
        return Dataset("Poker hand", rows[:, :-1], rows[:, -1].astype(str),
                       [f"card_{i}_{kind}" for i in range(1, 6) for kind in ("suit", "rank")],
                       {0, 2, 4, 6, 8}, target="hand_class", train_size=len(train), notes=[
                           "Suits are categorical; ranks are ordered numeric values. Original card positions are retained."])
    if number == 8:
        # Steel: sort timestamps before splitting so training precedes testing.
        names, rows = read_csv(folder / "Steel_industry_data.csv", header=True)
        order = sorted(range(len(rows)), key=lambda i: datetime.strptime(rows[i, 0], "%d/%m/%Y %H:%M"))
        rows = rows[order]
        features = names[1:-1]
        return Dataset("Steel industry", rows[:, 1:-1], rows[:, -1], features,
                       {features.index("WeekStatus"), features.index("Day_of_week")}, target="Load_Type", chronological=True,
                       notes=["Rows sorted by timestamp, then split chronologically. Timestamp excluded; row indices refer to this sorted order.",
                              "This classifies load type using same-time measurements, including Usage_kWh; it is not a future-load forecast."])
    if number == 9:
        # Wine: cultivar is the first column, followed by 13 chemical measurements.
        _, rows = read_csv(folder / "wine.data")
        return Dataset("Wine", rows[:, 1:], rows[:, 0],
                       ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline"], target="cultivar")
    if number == 10:
        # Wine quality: merge both colors and retain color as an extra predictor.
        matrices, labels = [], []
        for color in ("red", "white"):
            names, rows = read_csv(folder / f"winequality-{color}.csv", header=True, delimiter=";")
            matrices.append(np.column_stack((rows[:, :-1], np.repeat(color, len(rows)))))
            labels.append(rows[:, -1])
        return Dataset("Wine quality", np.vstack(matrices), np.concatenate(labels), names[:-1] + ["wine_color"],
                       {len(names) - 1}, target="quality", notes=["Red and white wines are combined with a color feature. Original integer quality scores are classification labels."])
    raise ValueError(f"Unknown benchmark: {number}")


def split_indices(dataset: Dataset, test_fraction: float, seed: int):
    """Return training indices, test indices and the name of the split method.

    Official boundaries take priority, followed by chronological ordering.
    Otherwise split each class independently to preserve class proportions
    approximately. Indices let features, labels and subject IDs stay aligned.
    """
    n = len(dataset.y)
    if dataset.train_size is not None:
        return np.arange(dataset.train_size), np.arange(dataset.train_size, n), "official"
    if dataset.chronological:
        cut = max(1, min(n - 1, int(n * (1 - test_fraction))))
        return np.arange(cut), np.arange(cut, n), "chronological"
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(dataset.y):
        indices = rng.permutation(np.flatnonzero(dataset.y == label))
        # Keep at least one training row per class; singleton classes stay in train.
        count = min(len(indices) - 1, max(1, round(len(indices) * test_fraction))) if len(indices) > 1 else 0
        test.extend(indices[:count])
        train.extend(indices[count:])
    # Shuffle each partition so rows are not left grouped by class.
    return rng.permutation(train).astype(int), rng.permutation(test).astype(int), "stratified"


def encode_features(X, features, categorical, train, test):
    """Fit transformations on training rows and apply them to both partitions.

    X contains raw cells; features names its columns; categorical identifies
    nominal columns; train and test are row-index arrays. Return X_train,
    X_test, expanded feature names and fitted transformation metadata.
    Both output matrices use float32 and support numeric threshold splits.
    """
    train_columns, test_columns, output_names, transforms = [], [], [], []
    for i, name in enumerate(features):
        column = X[:, i]
        if i in categorical:
            # Learn categories from training only; each becomes a binary column.
            values = np.asarray([str(value).strip() for value in column])
            categories = sorted(set(values[train]) - MISSING)
            for category in categories:
                train_columns.append((values[train] == category).astype(np.float32))
                test_columns.append((values[test] == category).astype(np.float32))
                output_names.append(f"{name}={category}")
            # Missing values and categories unseen in training share a fallback.
            train_columns.append((~np.isin(values[train], categories)).astype(np.float32))
            test_columns.append((~np.isin(values[test], categories)).astype(np.float32))
            output_names.append(f"{name}=<missing_or_unknown>")
            transforms.append({"feature": name, "kind": "one_hot", "categories": categories,
                               "unknown_policy": "dedicated missing_or_unknown column"})
        else:
            # Replace missing numeric cells with the training median. A completely
            # missing training column has no median, so use a documented zero.
            values = np.asarray([np.nan if str(v).strip() in MISSING else float(v) for v in column])
            if np.isinf(values).any():
                raise ValueError(f"Infinite numeric values in {name}")
            observed = values[train][~np.isnan(values[train])]
            median = float(np.median(observed)) if len(observed) else 0.0
            train_columns.append(np.nan_to_num(values[train], nan=median).astype(np.float32))
            test_columns.append(np.nan_to_num(values[test], nan=median).astype(np.float32))
            output_names.append(name)
            transforms.append({"feature": name, "kind": "numeric", "imputation_median": median,
                               "all_training_values_missing": not bool(len(observed))})
    return np.column_stack(train_columns), np.column_stack(test_columns), output_names, transforms


def process(number, args):
    """Run and save the complete preprocessing pipeline for one benchmark.

    args supplies paths, split settings and hardware thresholds from the CLI.
    Write model inputs to dataset.npz and human-readable details to metadata.json.
    """
    # Validate raw dimensions and labels before constructing the partitions.
    dataset = load_dataset(number, args.data_dir, args.hardware_bins)
    if dataset.X.shape != (len(dataset.y), len(dataset.features)):
        raise ValueError("Feature/label dimensions do not agree")
    if any(str(label).strip() in MISSING for label in dataset.y):
        raise ValueError("Missing class labels")
    # Split before fitting transformations to keep test data out of training.
    train, test, method = split_indices(dataset, args.test_fraction, args.seed)
    if not len(train) or not len(test):
        raise ValueError("Both training and test sets must be nonempty")
    X_train, X_test, features, transforms = encode_features(
        dataset.X, dataset.features, dataset.categorical, train, test)
    # Map original labels to consecutive integer IDs shared by both partitions.
    # classes[id] in the metadata reverses this mapping for predictions.
    classes = sorted(set(dataset.y[train]))
    unseen = set(dataset.y[test]) - set(classes)
    if unseen:
        raise ValueError(f"Test classes missing from training: {unseen}")
    lookup = {label: i for i, label in enumerate(classes)}
    y_train = np.asarray([lookup[label] for label in dataset.y[train]], dtype=np.int32)
    y_test = np.asarray([lookup[label] for label in dataset.y[test]], dtype=np.int32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Processed features must be finite")
    # Save source row indices for traceability and subject IDs for grouped tasks.
    arrays = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                  train_indices=train, test_indices=test)
    if dataset.groups is not None:
        arrays.update(groups_train=dataset.groups[train], groups_test=dataset.groups[test])
        if set(arrays["groups_train"]) & set(arrays["groups_test"]):
            raise ValueError("Subject overlap between training and test")
    # Record enough context to interpret columns, labels and experimental splits.
    metadata = dict(benchmark=number, name=dataset.name, target=dataset.target, classes=classes,
                    feature_names=features, raw_feature_names=dataset.features, transformations=transforms,
                    split=method, seed=args.seed if method == "stratified" else None,
                    requested_test_fraction=args.test_fraction if method != "official" else None,
                    train_rows=len(train), test_rows=len(test), feature_count=len(features),
                    class_counts={"train": np.bincount(y_train, minlength=len(classes)).tolist(),
                                  "test": np.bincount(y_test, minlength=len(classes)).tolist()},
                    notes=dataset.notes, numpy_version=np.__version__)
    # Compressed NumPy arrays are loaded without pickle; JSON remains inspectable.
    destination = args.output_dir / f"benchmark{number}"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "dataset.npz", **arrays)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"benchmark{number}: {len(train):,} train / {len(test):,} test; {len(features)} features; {len(classes)} classes", flush=True)


def main():
    """Parse CLI options, validate them and process selected benchmarks in order."""
    # Defaults cover all ten datasets; flags allow smaller runs and custom paths.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw",
                        help="Directory containing raw benchmark folders (default: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--benchmarks", type=int, nargs="+", choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--hardware-bins", type=float, nargs="+", default=[20, 100, 200],
                        help="Fixed inclusive upper PRP bounds for the adapted hardware classification task")
    args = parser.parse_args()
    # Reject invalid settings before reading datasets or writing output files.
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if not all(np.isfinite(args.hardware_bins)) or any(a >= b for a, b in zip(args.hardware_bins, args.hardware_bins[1:])):
        parser.error("--hardware-bins must be finite and strictly increasing")
    if args.output_dir.resolve() == args.data_dir.resolve():
        parser.error("Output directory must differ from the raw data directory")
    # Remove repeated selections and use numeric benchmark order.
    for number in sorted(set(args.benchmarks)):
        process(number, args)


if __name__ == "__main__":
    # Importing helpers in tests must not start a full preprocessing run.
    main()
