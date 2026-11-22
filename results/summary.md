# Decision tree benchmark results

Single run per configuration; all criteria use identical prepared partitions and stopping rules.
Maximum depth: 12; minimum samples per leaf: 2.

Memory is sampled resident process memory (RSS), including loaded data and the Python runtime. Short-lived peaks may be missed.
Timings exclude data loading, evaluation and file writing. These are initial measurements, not repeated-trial averages.

| Benchmark | Dataset | Criterion | Accuracy | Macro F1 | Majority accuracy | Train (s) | Predict (s) | Train peak MiB | Predict peak MiB | Nodes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Breast cancer | gain | 0.7018 | 0.5645 | 0.7018 | 0.066 | 0.001 | 31.3 | 31.4 | 101 |
| 1 | Breast cancer | gain_ratio | 0.6842 | 0.4543 | 0.7018 | 0.066 | 0.000 | 31.3 | 31.3 | 59 |
| 1 | Breast cancer | gini | 0.6667 | 0.5133 | 0.7018 | 0.067 | 0.000 | 31.3 | 31.4 | 101 |
| 2 | Cervical cancer risk factors | gain | 0.9360 | 0.4835 | 0.9360 | 0.041 | 0.000 | 31.6 | 31.6 | 73 |
| 2 | Cervical cancer risk factors | gain_ratio | 0.9186 | 0.5412 | 0.9360 | 0.040 | 0.000 | 31.6 | 31.6 | 57 |
| 2 | Cervical cancer risk factors | gini | 0.9186 | 0.4788 | 0.9360 | 0.042 | 0.000 | 31.6 | 31.7 | 77 |
| 3 | Computer hardware | gain | 0.7317 | 0.6538 | 0.5854 | 0.013 | 0.000 | 31.4 | 31.5 | 65 |
| 3 | Computer hardware | gain_ratio | 0.7805 | 0.7253 | 0.5854 | 0.015 | 0.000 | 31.3 | 31.3 | 61 |
| 3 | Computer hardware | gini | 0.8049 | 0.7338 | 0.5854 | 0.013 | 0.000 | 31.3 | 31.4 | 67 |
| 4 | Contraceptive method choice | gain | 0.5051 | 0.4631 | 0.4271 | 0.211 | 0.001 | 31.8 | 31.9 | 513 |
| 4 | Contraceptive method choice | gain_ratio | 0.5186 | 0.4950 | 0.4271 | 0.104 | 0.000 | 31.9 | 31.9 | 153 |
| 4 | Contraceptive method choice | gini | 0.5051 | 0.4764 | 0.4271 | 0.216 | 0.001 | 31.9 | 32.0 | 521 |
| 5 | Human activity recognition | gain | 0.8541 | 0.8507 | 0.1822 | 23.287 | 0.004 | 90.4 | 57.7 | 255 |
| 5 | Human activity recognition | gain_ratio | 0.8541 | 0.8500 | 0.1822 | 27.848 | 0.002 | 91.0 | 57.4 | 115 |
| 5 | Human activity recognition | gini | 0.8707 | 0.8679 | 0.1822 | 25.196 | 0.003 | 91.2 | 58.2 | 265 |
| 6 | Japanese vowels | gain | 0.7838 | 0.7820 | 0.0838 | 0.118 | 0.000 | 31.6 | 31.6 | 51 |
| 6 | Japanese vowels | gain_ratio | 0.8027 | 0.8012 | 0.0838 | 0.120 | 0.000 | 31.4 | 31.5 | 49 |
| 6 | Japanese vowels | gini | 0.8000 | 0.8045 | 0.0838 | 0.092 | 0.000 | 31.5 | 31.5 | 51 |
| 7 | Poker hand | gain | 0.5536 | 0.1295 | 0.5012 | 2.539 | 0.407 | 162.3 | 170.7 | 3427 |
| 7 | Poker hand | gain_ratio | 0.5791 | 0.1486 | 0.5012 | 1.238 | 0.235 | 163.3 | 168.7 | 1003 |
| 7 | Poker hand | gini | 0.5492 | 0.1662 | 0.5012 | 1.367 | 0.260 | 162.2 | 174.3 | 1831 |
| 8 | Steel industry | gain | 0.7566 | 0.6723 | 0.5285 | 0.361 | 0.002 | 40.8 | 36.5 | 693 |
| 8 | Steel industry | gain_ratio | 0.7493 | 0.6598 | 0.5285 | 0.285 | 0.001 | 41.1 | 36.0 | 303 |
| 8 | Steel industry | gini | 0.7578 | 0.6744 | 0.5285 | 0.357 | 0.002 | 40.6 | 36.2 | 741 |
| 9 | Wine | gain | 0.9167 | 0.9177 | 0.3889 | 0.006 | 0.000 | 31.4 | 31.4 | 13 |
| 9 | Wine | gain_ratio | 0.9167 | 0.9163 | 0.3889 | 0.007 | 0.000 | 31.6 | 31.6 | 11 |
| 9 | Wine | gini | 0.9167 | 0.9177 | 0.3889 | 0.007 | 0.000 | 31.3 | 31.3 | 15 |
| 10 | Wine quality | gain | 0.5746 | 0.3192 | 0.4362 | 0.585 | 0.003 | 33.7 | 32.8 | 1393 |
| 10 | Wine quality | gain_ratio | 0.4362 | 0.1304 | 0.4362 | 0.081 | 0.000 | 33.9 | 32.6 | 41 |
| 10 | Wine quality | gini | 0.5631 | 0.2753 | 0.4362 | 0.513 | 0.004 | 33.8 | 32.9 | 1219 |

Computer Hardware uses project-specific PRP classes; Japanese Vowels uses per-utterance summaries.
The majority predictor is a sanity check, not one of the assignment's three additional classification algorithms.
Detailed JSON files contain class-label mappings, confusion matrices (true rows, predicted columns), per-class scores and hardware information.
