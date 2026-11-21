# Benchmark preprocessing

From the repository root:

```sh
python -m pip install -r scripts/requirements.txt
python scripts/process_benchmarks.py
```

Process individual benchmarks with `--benchmarks 1 6 10`. Other options are
`--seed 42`, `--test-fraction 0.2`, `--data-dir PATH`, `--output-dir PATH`, and
`--hardware-bins 20 100 200`. Defaults resolve relative to the script, so the
command also works from another working directory. Rerunning replaces generated
outputs for the selected benchmarks; original data is preserved.

Raw inputs live in `data/raw/benchmark1/` through `data/raw/benchmark10/`.
`--data-dir` defaults to `data/raw` and must point to the directory containing
these benchmark folders. Outputs default to `data/processed`; original ZIP
archives remain in `data/zip`.

Each `data/processed/benchmarkN/` contains:

- `dataset.npz`: finite float32 `X_train`, `X_test`; zero-based int32 `y_train`,
  `y_test`; and original zero-based `train_indices`, `test_indices`.
- `metadata.json`: feature names, class labels in encoded ID order, class counts,
  split settings, fitted transformations, and dataset-specific decisions.

```python
import json
import numpy as np

folder = "data/processed/benchmark1"
with np.load(f"{folder}/dataset.npz", allow_pickle=False) as data:
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
with open(f"{folder}/metadata.json", encoding="utf-8") as stream:
    metadata = json.load(stream)
# tree.fit(X_train, y_train)
# predictions = tree.predict(X_test)
# Original class label for prediction i: metadata["classes"][i]
```

The decision tree can use numeric threshold splits for all output columns.
Nominal predictors are one-hot encoded, avoiding artificial category ordering.
Each categorical predictor gets a separate missing/unseen-category column.
Numeric missing values use training medians (zero if the training column is
entirely missing). No scaling is applied. These encodings support all three
requested criteria, but one-hot binary splitting is not the same algorithm as
native multiway categorical splitting; describe that choice in the report.

## Dataset decisions

| Benchmark | Target and preparation | Split |
| --- | --- | --- |
| 1: Breast cancer | Recurrence; all nine predictors treated as categorical | Stratified 80/20 |
| 2: Cervical cancer | Biopsy; excludes other screening outcomes and all four Dx columns | Stratified 80/20 |
| 3: Computer hardware | Adapted PRP classes: <=20, (20,100], (100,200], >200; six hardware predictors only | Stratified 80/20 |
| 4: Contraceptive method choice | Original method class; categorical demographic codes one-hot encoded | Stratified 80/20 |
| 5: Human activity recognition | Activity; supplied 561 engineered features; subject IDs stored separately | Official 7,352/2,947 |
| 6: Japanese vowels | Speaker; mean/std/min/max for each of 12 coefficients plus frame count, one row per utterance | Official 270/370 |
| 7: Poker hand | Original hand class; suits one-hot encoded, ranks numeric | Official 25,010/1,000,000 |
| 8: Steel industry | Load_Type; timestamp removed; calendar categories encoded | Chronological 80/20 |
| 9: Wine | Cultivar; 13 numeric predictors | Stratified 80/20 |
| 10: Wine quality | Original quality score as class; red and white combined with color feature | Stratified 80/20 |

Computer Hardware is originally **regression**, so the fixed classification bands
are a project-specific adaptation, not a standard classification benchmark.
Change thresholds explicitly if your instructor requires different classes.
Japanese Vowels summaries discard temporal order. Steel Industry uses same-time
energy measurements to classify load type; this is not forecasting. Its saved
row indices reference timestamp-sorted rows. For official splits, indices refer
to training rows followed by test rows; Wine Quality uses red then white rows.

All learned preprocessing uses only the training partition. Stratified splits
retain singleton classes in training; per-class rounding can change the overall
test fraction slightly. Official splits ignore seed and test fraction. HAR
subject IDs are saved as `groups_train` and `groups_test`, and overlap is checked.
Original source duplicates are retained; random row splits do not guarantee
that duplicate feature vectors are isolated in a single partition.

Use the same saved partitions for all classifiers. If tuning with cross-validation,
refit preprocessing on each fold's training rows from the raw data; do not use
the held-out test set for tuning. Baselines that require scaling should fit their
scaler on training data only.

Run focused preprocessing checks with:

```sh
python -m unittest discover -s scripts -p "test_*.py"
```
