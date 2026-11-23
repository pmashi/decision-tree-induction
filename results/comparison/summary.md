# Classification comparison

Means and sample standard deviations across repeated fresh-process runs of the same fixed split.
Repeated timings measure runtime variation, not uncertainty across different train/test splits.

| B | Dataset | Model | Runs | Accuracy | Macro F1 | Train seconds ± SD | Predict seconds ± SD | Train peak RSS MiB | Predict peak RSS MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Breast cancer | gain | 3 | 0.7018 | 0.5645 | 0.0718 ± 0.0020 | 0.0003 ± 0.0000 | 113.3 | 113.4 |
| 1 | Breast cancer | gain_ratio | 3 | 0.6842 | 0.4543 | 0.0672 ± 0.0062 | 0.0002 ± 0.0000 | 113.4 | 113.4 |
| 1 | Breast cancer | gaussian_nb | 3 | 0.3860 | 0.3852 | 0.0036 ± 0.0003 | 0.0003 ± 0.0000 | 113.4 | 113.4 |
| 1 | Breast cancer | gini | 3 | 0.6667 | 0.5133 | 0.0683 ± 0.0012 | 0.0003 ± 0.0000 | 113.6 | 113.6 |
| 1 | Breast cancer | knn | 3 | 0.7193 | 0.5778 | 0.0039 ± 0.0001 | 0.0032 ± 0.0000 | 113.5 | 114.2 |
| 1 | Breast cancer | logistic_regression | 3 | 0.6140 | 0.4792 | 0.0062 ± 0.0002 | 0.0005 ± 0.0001 | 113.9 | 113.9 |
| 2 | Cervical cancer risk factors | gain | 3 | 0.9360 | 0.4835 | 0.0450 ± 0.0034 | 0.0003 ± 0.0001 | 113.6 | 113.6 |
| 2 | Cervical cancer risk factors | gain_ratio | 3 | 0.9186 | 0.5412 | 0.0422 ± 0.0018 | 0.0002 ± 0.0000 | 113.7 | 113.7 |
| 2 | Cervical cancer risk factors | gaussian_nb | 3 | 0.0930 | 0.0929 | 0.0037 ± 0.0003 | 0.0003 ± 0.0001 | 113.4 | 113.4 |
| 2 | Cervical cancer risk factors | gini | 3 | 0.9186 | 0.4788 | 0.0451 ± 0.0036 | 0.0003 ± 0.0000 | 113.7 | 113.7 |
| 2 | Cervical cancer risk factors | knn | 3 | 0.9360 | 0.4835 | 0.0044 ± 0.0006 | 0.0037 ± 0.0002 | 113.6 | 114.3 |
| 2 | Cervical cancer risk factors | logistic_regression | 3 | 0.9360 | 0.4835 | 0.0072 ± 0.0004 | 0.0004 ± 0.0001 | 113.9 | 113.9 |
| 3 | Computer hardware | gain | 3 | 0.7317 | 0.6538 | 0.0132 ± 0.0005 | 0.0003 ± 0.0000 | 113.1 | 113.2 |
| 3 | Computer hardware | gain_ratio | 3 | 0.7805 | 0.7253 | 0.0150 ± 0.0008 | 0.0002 ± 0.0000 | 113.4 | 113.4 |
| 3 | Computer hardware | gaussian_nb | 3 | 0.7561 | 0.7417 | 0.0036 ± 0.0002 | 0.0003 ± 0.0000 | 113.2 | 113.2 |
| 3 | Computer hardware | gini | 3 | 0.8049 | 0.7338 | 0.0127 ± 0.0002 | 0.0003 ± 0.0000 | 113.2 | 113.2 |
| 3 | Computer hardware | knn | 3 | 0.7805 | 0.7395 | 0.0042 ± 0.0003 | 0.0035 ± 0.0007 | 113.5 | 113.9 |
| 3 | Computer hardware | logistic_regression | 3 | 0.7561 | 0.6598 | 0.0074 ± 0.0005 | 0.0004 ± 0.0000 | 113.5 | 113.6 |
| 4 | Contraceptive method choice | gain | 3 | 0.5051 | 0.4631 | 0.2345 ± 0.0126 | 0.0014 ± 0.0004 | 113.9 | 113.9 |
| 4 | Contraceptive method choice | gain_ratio | 3 | 0.5186 | 0.4950 | 0.1111 ± 0.0027 | 0.0005 ± 0.0000 | 114.0 | 114.0 |
| 4 | Contraceptive method choice | gaussian_nb | 3 | 0.4305 | 0.4298 | 0.0036 ± 0.0001 | 0.0003 ± 0.0000 | 113.8 | 113.9 |
| 4 | Contraceptive method choice | gini | 3 | 0.5051 | 0.4764 | 0.2373 ± 0.0066 | 0.0013 ± 0.0001 | 113.9 | 114.0 |
| 4 | Contraceptive method choice | knn | 3 | 0.4644 | 0.4352 | 0.0045 ± 0.0006 | 0.0048 ± 0.0001 | 113.9 | 114.7 |
| 4 | Contraceptive method choice | logistic_regression | 3 | 0.5254 | 0.4953 | 0.0087 ± 0.0004 | 0.0004 ± 0.0000 | 114.3 | 114.3 |
| 5 | Human activity recognition | gain | 1 | 0.8541 | 0.8507 | 10.2568 ± 0.0000 | 0.0024 ± 0.0000 | 172.9 | 141.4 |
| 5 | Human activity recognition | gain_ratio | 2 | 0.8541 | 0.8500 | 15.1849 ± 2.1935 | 0.0021 ± 0.0001 | 172.0 | 140.6 |
| 5 | Human activity recognition | gaussian_nb | 2 | 0.7703 | 0.7672 | 0.0272 ± 0.0034 | 0.0353 ± 0.0007 | 141.9 | 143.4 |
| 5 | Human activity recognition | gini | 2 | 0.8707 | 0.8679 | 13.7620 ± 3.3187 | 0.0024 ± 0.0000 | 173.0 | 141.8 |
| 5 | Human activity recognition | knn | 2 | 0.8802 | 0.8766 | 0.0417 ± 0.0006 | 0.5414 ± 0.0064 | 164.8 | 159.8 |
| 5 | Human activity recognition | logistic_regression | 2 | 0.9555 | 0.9553 | 1.2595 ± 0.0458 | 0.0049 ± 0.0001 | 153.5 | 137.8 |
