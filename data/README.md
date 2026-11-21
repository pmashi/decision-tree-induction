# Benchmark datasets

Raw datasets live in `data/raw/`, ordered alphabetically by name using the
numeric benchmark folder suffix. Folder names below are relative to `data/raw/`.

Run `python scripts/process_benchmarks.py` from the repository root to process
all ten datasets. Generated arrays and metadata are written to
`data/processed/benchmarkN/`. See [preprocessing instructions](../scripts/README.md).

Benchmark Datasets: 
1. [Breast Cancer](https://archive.ics.uci.edu/dataset/14/breast+cancer)
2. [Cervical Cancer (Risk Factors)](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
3. [Computer Hardware](https://archive.ics.uci.edu/dataset/29/computer+hardware)
4. [Contraceptive Method Choice](https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice)
5. [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
6. [Japanese Vowels](https://archive.ics.uci.edu/dataset/128/japanese+vowels)
7. [Poker Hand](https://archive.ics.uci.edu/dataset/158/poker+hand)
8. [Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
9. [Wine](https://archive.ics.uci.edu/dataset/109/wine)
10. [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

| Folder | Dataset | Extraction status |
| --- | --- | --- |
| benchmark1 | Breast cancer | Extracted |
| benchmark2 | Cervical cancer risk factors | Extracted |
| benchmark3 | Computer hardware | Extracted |
| benchmark4 | Contraceptive method choice | Extracted |
| benchmark5 | Human activity recognition using smartphones | Extracted, including nested UCI HAR Dataset.zip |
| benchmark6 | Japanese vowels | Extracted |
| benchmark7 | Poker hand | Extracted |
| benchmark8 | Steel industry | Original CSV retained |
| benchmark9 | Wine | Extracted |
| benchmark10 | Wine quality | Extracted |

Original archives are retained in `zip/`.
