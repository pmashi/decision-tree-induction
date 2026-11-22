# Classification comparison

Means and sample standard deviations across repeated fresh-process runs of the same fixed split.
Repeated timings measure runtime variation, not uncertainty across different train/test splits.

| B | Dataset | Model | Runs | Accuracy | Macro F1 | Train seconds ± SD | Predict seconds ± SD | Train peak RSS MiB | Predict peak RSS MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Breast cancer | gain | 3 | 0.7018 | 0.5645 | 0.0701 ± 0.0009 | 0.0003 ± 0.0000 | 141.1 | 141.1 |
| 1 | Breast cancer | gain_ratio | 3 | 0.6842 | 0.4543 | 0.0722 ± 0.0072 | 0.0003 ± 0.0002 | 141.2 | 141.2 |
| 1 | Breast cancer | gaussian_nb | 3 | 0.3860 | 0.3852 | 0.0145 ± 0.0028 | 0.0003 ± 0.0000 | 141.4 | 141.4 |
| 1 | Breast cancer | gini | 3 | 0.6667 | 0.5133 | 0.0745 ± 0.0041 | 0.0003 ± 0.0000 | 141.3 | 141.3 |
| 1 | Breast cancer | knn | 3 | 0.7193 | 0.5778 | 0.0136 ± 0.0026 | 0.0044 ± 0.0012 | 141.6 | 142.3 |
| 1 | Breast cancer | logistic_regression | 3 | 0.6140 | 0.4792 | 0.0242 ± 0.0104 | 0.0007 ± 0.0002 | 141.8 | 141.8 |
| 2 | Cervical cancer risk factors | gain | 3 | 0.9360 | 0.4835 | 0.0426 ± 0.0017 | 0.0003 ± 0.0000 | 141.2 | 141.2 |
| 2 | Cervical cancer risk factors | gain_ratio | 3 | 0.9186 | 0.5412 | 0.0423 ± 0.0025 | 0.0003 ± 0.0000 | 141.5 | 141.5 |
| 2 | Cervical cancer risk factors | gaussian_nb | 3 | 0.0930 | 0.0929 | 0.0149 ± 0.0029 | 0.0004 ± 0.0001 | 141.5 | 141.5 |
| 2 | Cervical cancer risk factors | gini | 3 | 0.9186 | 0.4788 | 0.0493 ± 0.0068 | 0.0003 ± 0.0000 | 141.4 | 141.5 |
| 2 | Cervical cancer risk factors | knn | 3 | 0.9360 | 0.4835 | 0.0122 ± 0.0003 | 0.0046 ± 0.0001 | 141.2 | 141.9 |
| 2 | Cervical cancer risk factors | logistic_regression | 3 | 0.9360 | 0.4835 | 0.0150 ± 0.0007 | 0.0004 ± 0.0000 | 141.7 | 141.7 |
| 3 | Computer hardware | gain | 3 | 0.7317 | 0.6538 | 0.0145 ± 0.0008 | 0.0003 ± 0.0001 | 141.1 | 141.1 |
| 3 | Computer hardware | gain_ratio | 3 | 0.7805 | 0.7253 | 0.0152 ± 0.0008 | 0.0002 ± 0.0000 | 141.6 | 141.6 |
| 3 | Computer hardware | gaussian_nb | 3 | 0.7561 | 0.7417 | 0.0133 ± 0.0017 | 0.0003 ± 0.0000 | 141.0 | 141.1 |
| 3 | Computer hardware | gini | 3 | 0.8049 | 0.7338 | 0.0138 ± 0.0014 | 0.0003 ± 0.0000 | 141.2 | 141.2 |
| 3 | Computer hardware | knn | 3 | 0.7805 | 0.7395 | 0.0125 ± 0.0006 | 0.0039 ± 0.0001 | 141.1 | 141.5 |
| 3 | Computer hardware | logistic_regression | 3 | 0.7561 | 0.6598 | 0.0179 ± 0.0022 | 0.0005 ± 0.0001 | 141.4 | 141.4 |
| 4 | Contraceptive method choice | gain | 3 | 0.5051 | 0.4631 | 0.2397 ± 0.0082 | 0.0013 ± 0.0002 | 141.9 | 141.9 |
| 4 | Contraceptive method choice | gain_ratio | 3 | 0.5186 | 0.4950 | 0.1110 ± 0.0020 | 0.0005 ± 0.0000 | 141.8 | 141.8 |
| 4 | Contraceptive method choice | gaussian_nb | 3 | 0.4305 | 0.4298 | 0.0131 ± 0.0012 | 0.0004 ± 0.0002 | 141.6 | 141.6 |
| 4 | Contraceptive method choice | gini | 3 | 0.5051 | 0.4764 | 0.2354 ± 0.0031 | 0.0012 ± 0.0000 | 141.7 | 141.7 |
| 4 | Contraceptive method choice | knn | 3 | 0.4644 | 0.4352 | 0.0133 ± 0.0003 | 0.0060 ± 0.0003 | 141.6 | 142.4 |
| 4 | Contraceptive method choice | logistic_regression | 3 | 0.5254 | 0.4953 | 0.0178 ± 0.0002 | 0.0006 ± 0.0001 | 142.2 | 142.2 |
| 5 | Human activity recognition | gain | 3 | 0.8541 | 0.8507 | 19.1524 ± 7.5711 | 0.0039 ± 0.0011 | 200.2 | 168.7 |
| 5 | Human activity recognition | gain_ratio | 3 | 0.8541 | 0.8500 | 22.5292 ± 8.3117 | 0.0029 ± 0.0008 | 200.0 | 168.5 |
| 5 | Human activity recognition | gaussian_nb | 3 | 0.7703 | 0.7672 | 0.0342 ± 0.0024 | 0.0373 ± 0.0023 | 176.1 | 171.6 |
| 5 | Human activity recognition | gini | 3 | 0.8707 | 0.8679 | 20.9246 ± 7.2863 | 0.0042 ± 0.0015 | 200.4 | 169.1 |
| 5 | Human activity recognition | knn | 3 | 0.8802 | 0.8766 | 0.0514 ± 0.0012 | 0.5477 ± 0.0057 | 200.1 | 187.6 |
| 5 | Human activity recognition | logistic_regression | 3 | 0.9555 | 0.9553 | 1.2401 ± 0.0422 | 0.0053 ± 0.0006 | 192.4 | 166.0 |
| 6 | Japanese vowels | gain | 3 | 0.7838 | 0.7820 | 0.1069 ± 0.0114 | 0.0004 ± 0.0002 | 141.5 | 141.5 |
| 6 | Japanese vowels | gain_ratio | 3 | 0.8027 | 0.8012 | 0.1379 ± 0.0130 | 0.0003 ± 0.0000 | 141.2 | 141.2 |
| 6 | Japanese vowels | gaussian_nb | 3 | 0.9568 | 0.9536 | 0.0122 ± 0.0003 | 0.0006 ± 0.0000 | 141.3 | 141.4 |
| 6 | Japanese vowels | gini | 3 | 0.8000 | 0.8045 | 0.0912 ± 0.0020 | 0.0002 ± 0.0000 | 141.2 | 141.3 |
| 6 | Japanese vowels | knn | 3 | 0.9649 | 0.9650 | 0.0125 ± 0.0003 | 0.0051 ± 0.0010 | 141.3 | 142.4 |
| 6 | Japanese vowels | logistic_regression | 3 | 0.9757 | 0.9733 | 0.0167 ± 0.0005 | 0.0005 ± 0.0000 | 141.7 | 141.7 |
| 7 | Poker hand | gain | 3 | 0.5536 | 0.1295 | 2.6687 ± 0.0773 | 3.3419 ± 0.0767 | 272.2 | 269.4 |
| 7 | Poker hand | gain_ratio | 3 | 0.5791 | 0.1486 | 1.2387 ± 0.0552 | 1.1533 ± 0.0115 | 273.2 | 269.4 |
| 7 | Poker hand | gaussian_nb | 3 | 0.4243 | 0.0725 | 0.0182 ± 0.0016 | 0.4829 ± 0.0004 | 264.1 | 267.6 |
| 7 | Poker hand | gini | 3 | 0.5492 | 0.1662 | 1.4001 ± 0.0173 | 1.9376 ± 0.1084 | 272.4 | 269.2 |
| 7 | Poker hand | knn | 3 | 0.4847 | 0.1992 | 0.0201 ± 0.0012 | 95.8470 ± 1.1096 | 267.0 | 271.3 |
| 7 | Poker hand | logistic_regression | 3 | 0.5008 | 0.0670 | 0.4418 ± 0.0137 | 0.1808 ± 0.0129 | 267.9 | 268.7 |
| 8 | Steel industry | gain | 3 | 0.7566 | 0.6723 | 0.3642 ± 0.0061 | 0.0048 ± 0.0004 | 150.8 | 147.0 |
| 8 | Steel industry | gain_ratio | 3 | 0.7493 | 0.6598 | 0.2926 ± 0.0116 | 0.0024 ± 0.0004 | 150.8 | 147.0 |
| 8 | Steel industry | gaussian_nb | 3 | 0.6310 | 0.5726 | 0.0171 ± 0.0012 | 0.0015 ± 0.0000 | 145.5 | 144.9 |
| 8 | Steel industry | gini | 3 | 0.7578 | 0.6744 | 0.3972 ± 0.0237 | 0.0048 ± 0.0001 | 150.8 | 147.0 |
| 8 | Steel industry | knn | 3 | 0.7289 | 0.6449 | 0.0191 ± 0.0012 | 0.6710 ± 0.0080 | 146.4 | 147.4 |
| 8 | Steel industry | logistic_regression | 3 | 0.6799 | 0.5979 | 0.1835 ± 0.0325 | 0.0017 ± 0.0003 | 148.0 | 145.2 |
| 9 | Wine | gain | 3 | 0.9167 | 0.9177 | 0.0068 ± 0.0006 | 0.0001 ± 0.0000 | 141.3 | 141.3 |
| 9 | Wine | gain_ratio | 3 | 0.9167 | 0.9163 | 0.0121 ± 0.0046 | 0.0001 ± 0.0001 | 141.2 | 141.2 |
| 9 | Wine | gaussian_nb | 3 | 0.9722 | 0.9718 | 0.0116 ± 0.0008 | 0.0003 ± 0.0000 | 140.9 | 141.0 |
| 9 | Wine | gini | 3 | 0.9167 | 0.9177 | 0.0070 ± 0.0003 | 0.0001 ± 0.0000 | 141.2 | 141.3 |
| 9 | Wine | knn | 3 | 0.9722 | 0.9718 | 0.0133 ± 0.0011 | 0.0042 ± 0.0007 | 141.0 | 141.4 |
| 9 | Wine | logistic_regression | 3 | 1.0000 | 1.0000 | 0.0159 ± 0.0010 | 0.0005 ± 0.0000 | 141.5 | 141.6 |
| 10 | Wine quality | gain | 3 | 0.5746 | 0.3192 | 0.6364 ± 0.0257 | 0.0033 ± 0.0000 | 144.0 | 144.0 |
| 10 | Wine quality | gain_ratio | 3 | 0.4362 | 0.1304 | 0.0796 ± 0.0018 | 0.0003 ± 0.0000 | 143.6 | 143.6 |
| 10 | Wine quality | gaussian_nb | 3 | 0.3200 | 0.1846 | 0.0128 ± 0.0010 | 0.0006 ± 0.0000 | 142.0 | 142.0 |
| 10 | Wine quality | gini | 3 | 0.5631 | 0.2753 | 0.5702 ± 0.0057 | 0.0039 ± 0.0007 | 143.9 | 143.9 |
| 10 | Wine quality | knn | 3 | 0.5715 | 0.3193 | 0.0153 ± 0.0035 | 0.0262 ± 0.0006 | 142.4 | 143.0 |
| 10 | Wine quality | logistic_regression | 3 | 0.5385 | 0.2194 | 0.1052 ± 0.0020 | 0.0006 ± 0.0000 | 142.7 | 142.7 |
