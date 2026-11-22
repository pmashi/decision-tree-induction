# Report discussion notes

## Overall comparison

Each dataset has equal weight in the averages below; Poker's million test rows do not dominate them.
Compare models only once they cover the same datasets. These are descriptive averages, not significance tests.

| Model | Datasets | Mean accuracy | Mean macro F1 | Total train seconds | Total predict seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| knn | 10 | 0.7503 | 0.6213 | 0.183 | 97.120 |
| logistic_regression | 10 | 0.7482 | 0.5931 | 2.078 | 0.192 |
| gini | 10 | 0.7353 | 0.5908 | 23.763 | 1.953 |
| gain | 10 | 0.7314 | 0.5836 | 23.302 | 3.356 |
| gain_ratio | 10 | 0.7240 | 0.5722 | 24.531 | 1.161 |
| gaussian_nb | 10 | 0.5740 | 0.5172 | 0.162 | 0.525 |

## Dataset observations

- B1 (Breast cancer): highest observed macro F1 0.5778 (knn); highest accuracy 0.7193 (knn). Training-majority baseline accuracy: 0.7018.
- B2 (Cervical cancer risk factors): highest observed macro F1 0.5412 (gain_ratio); highest accuracy 0.9360 (gain, knn, logistic_regression). Training-majority baseline accuracy: 0.9360.
- B3 (Computer hardware): highest observed macro F1 0.7417 (gaussian_nb); highest accuracy 0.8049 (gini). Training-majority baseline accuracy: 0.5854.
- B4 (Contraceptive method choice): highest observed macro F1 0.4953 (logistic_regression); highest accuracy 0.5254 (logistic_regression). Training-majority baseline accuracy: 0.4271.
- B5 (Human activity recognition): highest observed macro F1 0.9553 (logistic_regression); highest accuracy 0.9555 (logistic_regression). Training-majority baseline accuracy: 0.1822.
- B6 (Japanese vowels): highest observed macro F1 0.9733 (logistic_regression); highest accuracy 0.9757 (logistic_regression). Training-majority baseline accuracy: 0.0838.
- B7 (Poker hand): highest observed macro F1 0.1992 (knn); highest accuracy 0.5791 (gain_ratio). Training-majority baseline accuracy: 0.5012.
- B8 (Steel industry): highest observed macro F1 0.6744 (gini); highest accuracy 0.7578 (gini). Training-majority baseline accuracy: 0.5285.
- B9 (Wine): highest observed macro F1 1.0000 (logistic_regression); highest accuracy 1.0000 (logistic_regression). Training-majority baseline accuracy: 0.3889.
- B10 (Wine quality): highest observed macro F1 0.3193 (knn); highest accuracy 0.5746 (gain). Training-majority baseline accuracy: 0.4362.

## Convergence and warnings

No warnings were emitted by the measured model operations.

## Method and limitations

- Baselines use existing scikit-learn implementations: logistic regression (StandardScaler, lbfgs, C=1, max_iter=5000), KNN (StandardScaler, k=5, uniform Euclidean brute-force search), GaussianNB (var_smoothing=1e-9). Full defaults and package versions are recorded in each run JSON.
- The custom trees use gain, gain ratio or Gini, maximum depth 12 and minimum leaf size 2 unless overridden; consult each run config for the actual settings. Direct binary gain ratio is not full C4.5.
- All models use exactly the same prepared partitions. Scaling is fitted on training rows only. Hyperparameters are fixed in advance; no test-set tuning is performed. Per-dataset winners are retrospective descriptions, not deployable model-selection rules.
- One fresh subprocess per repetition/configuration; native BLAS/OpenMP pools limited to one thread. Timings include baseline scaling and shared batched prediction. Data loading, evaluation, prediction-file writing and training-score computation are excluded.
- Memory is process RSS sampled every 5 ms, including Python, common library imports, data, model and temporary allocations. Peaks shorter than the interval can be missed. Baseline and increase over baseline are also in runs.csv; these are not model-only storage estimates.
- The same deterministic split is repeated. Timing SD is not a confidence interval for accuracy. Cross-validation or additional independent splits would be needed to assess generalization variability.
- Accuracy can hide poor minority-class recall. Use macro F1, per_class.csv and confusion matrices for the imbalanced cervical, Poker and quality datasets. Macro averages include all defined classes; undefined class metrics are zero.
- Training accuracy and training macro F1 are recorded to help identify overfitting. KNN's training score includes each training example as its own neighbor and should be interpreted accordingly.
- Hardware is adapted from regression into fixed PRP bands; Vowels uses per-utterance statistics; Steel uses a chronological split and same-time measurements. Source duplicates remain. These choices limit comparison with published benchmark scores.
- GaussianNB's Gaussian feature assumption is imperfect for one-hot columns. KNN uses Euclidean distances on scaled one-hot and numeric features. Discuss these modeling choices rather than interpreting rankings as universal algorithm superiority.
- Library implementations use optimized native routines while the custom tree orchestrates searches in Python/NumPy; timing differences reflect implementation as well as algorithm.

## Reproducibility artifacts

runs.csv retains each measurement; summary.csv holds repeat means/SD; overall.csv holds equal-dataset averages; per_class.csv holds class-level scores. results.json and individual run JSONs retain confusion matrices, full settings, versions, hardware, input SHA-256 hashes, warning text and iteration counts. predictions/ retains true/predicted training and test labels. manifest.json records command arguments and source/data hashes.
