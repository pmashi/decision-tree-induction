# Running the project

Run these commands from the project root in PowerShell; existing processed data can be reused by skipping preprocessing.

On windows 11: 
Set up virtual environment, activate it, install `requirements.txt`, then process benchmarks and run main. 
```cmd prompt
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python scripts/process_benchmarks.py
python main.py
```

## Python files

- 'main.py' : Runs the experiments, prints results, and saves comparison outputs.
- 'algorithms/__init__.py' : Exposes the custom decision-tree classifier from the algorithms package.
- 'algorithms/decision_tree.py' : Implements tree nodes, threshold selection, recursive training, and prediction.
- 'algorithms/split_criteria.py' : Calculates entropy, information gain, gain ratio, and Gini impurity reduction.
- 'algorithms/baselines.py' : Configures logistic regression, five-nearest neighbors, and Gaussian naive Bayes classifiers.
- 'benchmarking/__init__.py' : Marks the benchmarking directory as a Python package.
- 'benchmarking/resources.py' : Measures model training and prediction time and process memory usage.
- 'benchmarking/evaluation.py' : Calculates accuracy, precision, recall, F1 scores, and confusion matrices.
- 'benchmarking/reporting.py' : Combines experiment results into summary tables, charts, and discussion notes.
- 'scripts/process_benchmarks.py' : Loads, cleans, encodes, splits, and saves all ten benchmark datasets.
- 'scripts/test_process_benchmarks.py' : Tests preprocessing, missing-value handling, encoding, and dataset splits.
- 'scripts/validate_results.py' : Audits saved experiment results against predictions, labels, hashes, and summary calculations.
- 'tests/test_algorithms.py' : Tests split calculations, tree behavior, and classification evaluation.
- 'tests/test_baselines.py' : Tests comparison classifiers, feature scaling, and agreement with reference metrics.
- 'report/worked_example.py' : Generates the real-data worked example and checks its tree and predictions.
- 'report/build_assets.py' : Creates LaTeX tables and figures from the saved benchmark results.

## Saved experiment files

Files in `results/comparison/` record the experiments used in the report.

- 'runs.csv' : Stores every individual run's measurements.
- 'summary.csv' : Stores per-benchmark and per-model means and standard deviations.
- 'per_class.csv' : Stores evaluation scores for each class.
- 'overall.csv' : Stores overall comparison scores and total training and prediction times.
- 'results.json' : Stores complete experiment records and model settings.
- 'predictions/' : Stores true and predicted labels for training and testing.
- 'manifest.json' : Records command arguments, source and input hashes, and experiment completion status.
- 'validation.json' : Records the results of the saved-experiment audit.

The report asset builder reads these saved results and copies summary and provenance files into `report/` alongside the generated tables and figures.
