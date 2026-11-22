# LaTeX project report

Open [report.pdf](report.pdf) for the compiled report. Compilation was
checked for unresolved references and overflowing content; all 60 resource rows
were verified in the PDF. Tables and the resource appendix were visually checked.

`report.tex` is the complete report source. Fill in the centered student-name and course fields near the top of the source
before submission. The report uses the validated 180-run comparison in
`results/comparison/`, not the earlier single-run tree results.

From the repository root, regenerate tables and vector figures with:

```powershell
.\.venv\Scripts\python.exe report/worked_example.py
.\.venv\Scripts\python.exe report/build_assets.py
```

The worked-example script copies the original Wine rows into `wine_example_raw.csv` and the report, then replays the full dataset's training/testing split and verifies that it matches the saved assignments. Training/testing describes each row's assigned use; the cultivar class already appears in the first raw column. The script fits tiny trees on six real training rows and checks two held-out rows. It writes the source-row table, candidate split scores, and predictions used in Section 5.1. The asset builder reads existing benchmark results; it does not rerun the full experiments. If changing the
experiments, review the report's narrative numbers and methodological statements
as well as regenerating its tables.

Compile from this folder:

```sh
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively, upload `report.tex`, `tables/`, and `figures/` together to a LaTeX
editor and select `report.tex` as the main document. No bibliography or BibTeX step is needed.

The report includes plain-language split-criteria explanations and algorithm pseudocode, preprocessing decisions,
worked validation examples, test coverage, baseline settings, hardware/software
details, all ten datasets' scores, tree size/training gaps, confusion-matrix
analysis, timing and memory discussion, limitations, and the complete 60-row
resource table with means and sample standard deviations.

`summary.csv`, `overall.csv`, `manifest.json` and `validation.json` are copies of
the numerical/provenance artifacts used to generate the report. Detailed run
records, class scores and saved predictions remain in `results/comparison/`.
